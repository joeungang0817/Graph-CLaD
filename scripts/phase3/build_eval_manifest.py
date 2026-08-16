"""Build the fixed Phase 3B-R1 evaluation manifest.

The manifest separates training selection, validation, natural test, and the
holding challenge view before any model is trained.  It stores stable sample
keys rather than duplicating graph payloads, so the same manifest can be used
by every model and seed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from scripts.phase3.sampling import category_counts, select_with_config
except ModuleNotFoundError:  # standalone Colab bundle fallback
    from sampling import category_counts, select_with_config


DEFAULT_RELATIONS = (
    "left",
    "right",
    "front",
    "behind",
    "above",
    "below",
    "contact",
    "on",
    "holding",
)


def _expand_config_environment(value: Any) -> Any:
    """Expand ${VAR} references for portable Linux/SSH configs."""

    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_config_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_config_environment(item) for key, item in value.items()}
    return value


def _task_family(task_id: int) -> str:
    return f"libero_spatial:{task_id}"


def _task_path(root: Path, task_id: int) -> Path:
    return root / f"task{task_id}" / f"phase2d_task{task_id}_graph_dataset.jsonl.gz"


def _read_task_samples(root: Path, task_id: int) -> list[dict[str, Any]]:
    path = _task_path(root, task_id)
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number} is not an object")
            for raw in payload.get("samples", []):
                if not isinstance(raw, Mapping):
                    continue
                sample = dict(raw)
                sample["suite"] = str(sample.get("suite") or "libero_spatial")
                sample["task_id"] = int(sample.get("task_id", task_id))
                sample["episode_id"] = str(
                    sample.get("episode_id")
                    or f"task{task_id}_{payload.get('demo_id', 'unknown')}"
                )
                rows.append(sample)
    return rows


def load_samples(root: Path, task_ids: Iterable[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        rows.extend(_read_task_samples(root, task_id))
    return rows


def sample_id(sample: Mapping[str, Any]) -> str:
    """Return a stable ID shared by natural and target-aligned releases."""

    required = ("suite", "task_id", "episode_id", "start_step", "target_step", "tau")
    missing = [key for key in required if sample.get(key) is None]
    if missing:
        raise ValueError(f"sample is missing stable key fields {missing}: {sample.get('episode_id')}")
    return "|".join(
        [
            str(sample["suite"]),
            str(sample["task_id"]),
            str(sample["episode_id"]),
            str(sample["start_step"]),
            str(sample["target_step"]),
            str(sample["tau"]),
        ]
    )


def payload_hash(sample: Mapping[str, Any]) -> str:
    """Hash the graph/action/label payload shared by overlapping views."""

    cached = sample.get("_payload_sha256")
    if isinstance(cached, str) and cached:
        return cached
    payload = {
        "graph_t": sample.get("graph_t"),
        "action_window": sample.get("action_window"),
        "graph_target": sample.get("graph_target"),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _row_metadata(sample: Mapping[str, Any], source: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id(sample),
        "source": source,
        "suite": str(sample["suite"]),
        "task_id": int(sample["task_id"]),
        "task_family": f"{sample['suite']}:{sample['task_id']}",
        "episode_id": str(sample["episode_id"]),
        "split": str(sample.get("_evaluation_split", sample.get("split", "unknown"))),
        "start_step": int(sample["start_step"]),
        "target_step": int(sample["target_step"]),
        "tau": int(sample["tau"]),
        "payload_sha256": payload_hash(sample),
        "target_categories": [str(value) for value in sample.get("target_categories", [])],
    }


def _relation_record(edge: Mapping[str, Any], relation: str) -> Mapping[str, Any] | None:
    relations = edge.get("relations", {})
    record = relations.get(relation) if isinstance(relations, Mapping) else None
    return record if isinstance(record, Mapping) else None


def _relation_support(samples: Sequence[Mapping[str, Any]], relations: Sequence[str]) -> dict[str, Any]:
    support: dict[str, dict[str, int]] = {
        relation: {"valid": 0, "positive": 0, "changed": 0} for relation in relations
    }
    for sample in samples:
        current = {
            (str(edge.get("source")), str(edge.get("target"))): edge
            for edge in (sample.get("graph_t", {}) or {}).get("edges", [])
        }
        future = {
            (str(edge.get("source")), str(edge.get("target"))): edge
            for edge in (sample.get("graph_target", {}) or {}).get("edges", [])
        }
        for key in sorted(set(current) & set(future)):
            for relation in relations:
                now = _relation_record(current[key], relation)
                later = _relation_record(future[key], relation)
                if not now or not later or not now.get("valid") or not later.get("valid"):
                    continue
                item = support[relation]
                item["valid"] += 1
                item["positive"] += int(bool(later.get("value")))
                item["changed"] += int(bool(now.get("value")) != bool(later.get("value")))
    return support


def _feature_availability(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observed = {
        "relative_position": False,
        "distance": False,
        "contact_and_validity": False,
        "gripper_qpos_or_closure": False,
        "action_window": False,
        "validity_masks": False,
        "relative_velocity": False,
        "object_following_stability": False,
        "past_graph_history": False,
        "future_graph_present_forbidden_input": False,
    }
    for sample in samples:
        observed["action_window"] |= bool(sample.get("action_window"))
        observed["future_graph_present_forbidden_input"] |= "graph_target" in sample
        observed["past_graph_history"] |= any(
            key in sample for key in ("past_graph", "past_graphs", "graph_history", "history")
        )
        graph = sample.get("graph_t", {}) or {}
        for node in graph.get("nodes", []):
            features = node.get("features", {}) if isinstance(node, Mapping) else {}
            feature_keys = {str(key).lower() for key in features} if isinstance(features, Mapping) else set()
            observed["gripper_qpos_or_closure"] |= any(
                token in key for key in feature_keys for token in ("qpos", "closure", "gripper")
            )
            observed["validity_masks"] |= any("valid" in key or "mask" in key for key in feature_keys)
        for edge in graph.get("edges", []):
            features = edge.get("features", {}) if isinstance(edge, Mapping) else {}
            if isinstance(features, Mapping):
                observed["relative_position"] |= "relative_position" in features
                observed["distance"] |= "distance" in features
                observed["relative_velocity"] |= any(
                    token in str(key).lower() for key in features for token in ("velocity", "delta", "change")
                )
                observed["object_following_stability"] |= any(
                    token in str(key).lower() for key in features for token in ("following", "stability")
                )
                observed["validity_masks"] |= any(
                    "valid" in str(key).lower() or "mask" in str(key).lower() for key in features
                )
            contact = _relation_record(edge, "contact")
            observed["contact_and_validity"] |= bool(contact and "valid" in contact)
    return observed


def _filter(rows: Sequence[Mapping[str, Any]], *, task_ids: Iterable[int], split: str, tau: int) -> list[dict[str, Any]]:
    allowed = set(task_ids)
    return [
        dict(row)
        for row in rows
        if int(row.get("task_id", -1)) in allowed
        and str(row.get("_evaluation_split", row.get("split"))) == split
        and int(row.get("tau", -1)) == tau
    ]


def _load_episode_splits(path: Path) -> dict[tuple[int, str], str]:
    """Load the authoritative in-task split for each episode.

    Dataset samples expose the task-generalization split in ``sample['split']``
    (which is ``train`` for all three task families in this release).  The
    evaluation protocol needs the independent in-task split, so it must come
    from the fixed demo split manifest instead.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[tuple[int, str], str] = {}
    for episode in payload.get("episodes", []):
        task_id = int(episode["task_id"])
        demo_key = str(episode["demo_key"])
        if not demo_key.startswith("demo_"):
            continue
        # Dataset payloads use ``task0_demo0`` while the split manifest uses
        # ``demo_0``; normalize both to the dataset's stable episode key.
        demo_number = demo_key.removeprefix("demo_")
        episode_id = f"task{task_id}_demo{demo_number}"
        result[(task_id, episode_id)] = str(episode["in_task_split"])
    if not result:
        raise ValueError(f"no episode in-task splits found in {path}")
    return result


def _outer_folds(task_ids: Sequence[int]) -> list[dict[str, Any]]:
    families = {_task_family(task_id): task_id for task_id in task_ids}
    folds: list[dict[str, Any]] = []
    for held_out in task_ids:
        held_out_family = _task_family(held_out)
        train_families = [family for family in sorted(families) if family != held_out_family]
        folds.append(
            {
                "name": f"test_task{held_out}",
                "held_out_task": held_out_family,
                "train_task_families": train_families,
                "validation_source_split": "validation",
                "test_source_split": "test",
            }
        )
    return folds


def _check_disjoint(fold: Mapping[str, Any], rows_by_role: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[str]:
    warnings: list[str] = []
    for role, rows in rows_by_role.items():
        ids = [sample_id(row) for row in rows]
        if len(ids) != len(set(ids)):
            warnings.append(
                f"{fold['name']}: duplicate sample IDs in {role}: "
                f"{len(ids) - len(set(ids))}"
            )
    role_episodes = {
        role: {str(row["episode_id"]) for row in rows}
        for role, rows in rows_by_role.items()
        if role in {"train", "validation", "natural_test"}
    }
    roles = sorted(role_episodes)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            overlap = role_episodes[left] & role_episodes[right]
            if overlap:
                warnings.append(f"{fold['name']}: episode overlap {left}/{right}: {sorted(overlap)[:5]}")
    natural_ids = {sample_id(row) for row in rows_by_role.get("natural_test", [])}
    challenge_ids = {sample_id(row) for row in rows_by_role.get("challenge_test", [])}
    if not challenge_ids <= natural_ids:
        missing = sorted(challenge_ids - natural_ids)
        warnings.append(f"{fold['name']}: challenge IDs missing from natural test: {missing[:5]}")
    natural_by_id = {
        sample_id(row): payload_hash(row)
        for row in rows_by_role.get("natural_test", [])
    }
    challenge_by_id = {
        sample_id(row): payload_hash(row)
        for row in rows_by_role.get("challenge_test", [])
    }
    mismatched = sorted(
        identifier
        for identifier in natural_ids & challenge_ids
        if natural_by_id[identifier] != challenge_by_id[identifier]
    )
    if mismatched:
        warnings.append(
            f"{fold['name']}: natural/challenge payload hash mismatch: "
            f"{mismatched[:5]}"
        )
    return warnings


def _overlap_payload_hash_qa(
    natural_rows: Sequence[Mapping[str, Any]],
    challenge_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    natural = {sample_id(row): payload_hash(row) for row in natural_rows}
    challenge = {sample_id(row): payload_hash(row) for row in challenge_rows}
    overlap = sorted(set(natural) & set(challenge))
    mismatched = [key for key in overlap if natural[key] != challenge[key]]
    return {
        "natural_samples": len(natural),
        "challenge_samples": len(challenge),
        "overlap_samples": len(overlap),
        "challenge_subset_of_natural": set(challenge) <= set(natural),
        "payload_hash_match_count": len(overlap) - len(mismatched),
        "payload_hash_mismatch_count": len(mismatched),
        "mismatched_sample_ids": mismatched[:20],
        "hash_fields": ["graph_t", "action_window", "graph_target"],
    }


def build_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    task_ids = [int(value) for value in config.get("task_ids", [0, 1, 2])]
    tau = int(config.get("primary_tau", 6))
    source_roots = config.get("source_roots", {})
    natural_root = Path(str(source_roots["natural"]))
    target_root = Path(str(source_roots["target_aligned"]))
    split_manifest = Path(str(source_roots["split_manifest"]))
    if not natural_root.exists():
        raise FileNotFoundError(f"natural source root does not exist: {natural_root}")
    if not target_root.exists():
        raise FileNotFoundError(f"target source root does not exist: {target_root}")
    if not split_manifest.exists():
        raise FileNotFoundError(f"split manifest does not exist: {split_manifest}")

    natural_rows = load_samples(natural_root, task_ids)
    target_rows = load_samples(target_root, task_ids)
    episode_splits = _load_episode_splits(split_manifest)
    missing_episode_splits: set[tuple[int, str]] = set()
    for row in [*natural_rows, *target_rows]:
        key = (int(row["task_id"]), str(row["episode_id"]))
        split = episode_splits.get(key)
        if split is None:
            missing_episode_splits.add(key)
            continue
        row["_evaluation_split"] = split
    # Payloads are immutable after loading.  Cache one digest on each source
    # row so fold-level QA and metadata export do not repeatedly serialize the
    # same graph/action/label payload.  Natural and target-aligned rows keep
    # separate caches, so a cross-view mismatch cannot be hidden.
    for row in [*natural_rows, *target_rows]:
        if (
            int(row.get("tau", -1)) == tau
            and row.get("_evaluation_split") in {"train", "validation", "test"}
        ):
            row["_payload_sha256"] = payload_hash(row)
    relations = tuple(config.get("relations", DEFAULT_RELATIONS))
    sampling = dict(config.get("training_sampling", {}))
    split_contract = dict(config.get("split_contract", {}))
    validation_source = str(
        split_contract.get("validation_source", "target_aligned")
    )
    if validation_source not in {"natural", "target_aligned"}:
        raise ValueError(
            "split_contract.validation_source must be natural or target_aligned"
        )
    cap = int(sampling.get("cap_per_task", 600))
    task_quota = int(sampling.get("category_quota", 120))
    folds: list[dict[str, Any]] = []
    global_warnings: list[str] = []
    if missing_episode_splits:
        global_warnings.append(
            f"episode split manifest missing {len(missing_episode_splits)} episode IDs"
        )

    for fold in _outer_folds(task_ids):
        held_out_task = int(fold["held_out_task"].split(":")[-1])
        train_tasks = [int(family.split(":")[-1]) for family in fold["train_task_families"]]
        train_rows: list[dict[str, Any]] = []
        for task_id in train_tasks:
            candidates = _filter(target_rows, task_ids=[task_id], split="train", tau=tau)
            selected = select_with_config(
                candidates,
                cap,
                {
                    "method": "category_aware_episode_round_robin_v2",
                    "category_quota": task_quota,
                    "categories": sampling.get(
                        "categories",
                        [
                            "holding_changed",
                            "future_holding_positive",
                            "hard_negative",
                            "background",
                        ],
                    ),
                },
            )
            if len(selected) != cap:
                global_warnings.append(
                    f"{fold['name']}: task{task_id} selected {len(selected)} train rows, expected {cap}"
                )
            train_rows.extend(selected)

        validation_rows = _filter(
            natural_rows if validation_source == "natural" else target_rows,
            task_ids=train_tasks,
            split="validation",
            tau=tau,
        )
        natural_test_rows = _filter(
            natural_rows,
            task_ids=[held_out_task],
            split="test",
            tau=tau,
        )
        challenge_test_rows = _filter(
            target_rows,
            task_ids=[held_out_task],
            split="test",
            tau=tau,
        )
        role_rows = {
            "train": train_rows,
            "validation": validation_rows,
            "natural_test": natural_test_rows,
            "challenge_test": challenge_test_rows,
        }
        role_sources = {
            "train": "target_aligned",
            "validation": validation_source,
            "natural_test": "natural",
            "challenge_test": "target_aligned",
        }
        warnings = _check_disjoint(fold, role_rows)
        global_warnings.extend(warnings)
        category_counts_by_role = {
            role: dict(category_counts(rows)) for role, rows in role_rows.items()
        }
        train_category_counts_by_task = {
            str(task_id): dict(
                category_counts(
                    [row for row in train_rows if int(row["task_id"]) == task_id]
                )
            )
            for task_id in train_tasks
        }
        fold_report = dict(fold)
        fold_report.update(
            {
                "sample_counts": {role: len(rows) for role, rows in role_rows.items()},
                "episode_counts": {
                    role: len({str(row["episode_id"]) for row in rows})
                    for role, rows in role_rows.items()
                },
                "category_counts": category_counts_by_role,
                "train_category_counts_by_task": train_category_counts_by_task,
                "relation_support": {
                    "natural_test": _relation_support(natural_test_rows, relations),
                    "challenge_test": _relation_support(challenge_test_rows, relations),
                },
                "overlap_payload_hash_qa": _overlap_payload_hash_qa(
                    natural_test_rows, challenge_test_rows
                ),
                "sample_keys": {
                    role: [_row_metadata(row, role_sources[role]) for row in rows]
                    for role, rows in role_rows.items()
                },
                "role_sources": role_sources,
                "source_roots": {
                    "natural": str(natural_root),
                    "target_aligned": str(target_root),
                },
            }
        )
        folds.append(fold_report)

    all_feature_rows = target_rows[: min(len(target_rows), 500)]
    required_categories = tuple(
        sampling.get(
            "categories",
            ["holding_changed", "future_holding_positive", "hard_negative", "background"],
        )
    )
    for fold in folds:
        for category in required_categories:
            for task_family in fold["train_task_families"]:
                task_id = int(task_family.split(":")[-1])
                count = fold["train_category_counts_by_task"][str(task_id)].get(
                    category, 0
                )
                if count < task_quota:
                    global_warnings.append(
                        f"{fold['name']}: task{task_id} train category "
                        f"{category} below quota: {count}"
                    )
        for role in ("natural_test", "challenge_test"):
            support = fold["relation_support"][role]["holding"]
            if support["valid"] <= 0 or support["positive"] <= 0:
                global_warnings.append(
                    f"{fold['name']}: {role} lacks holding positive support: {support}"
                )

    return {
        "protocol": str(config.get("protocol", "phase3B-R1-eval-v1")),
        "status": "pass" if not global_warnings else "fail",
        "primary_tau": tau,
        "task_ids": task_ids,
        "relations": list(relations),
        "source_roots": {key: str(value) for key, value in source_roots.items()},
        "training_sampling": {
            "method": "category_aware_episode_round_robin_v2",
            "cap_per_task": cap,
            "category_quota": task_quota,
            "categories": list(required_categories),
            "counts_are_multilabel": True,
        },
        "validation_protocol": {
            "source": validation_source,
            "selection": "all held-in task validation-episode transitions at primary tau",
            "event_enrichment": validation_source != "natural",
        },
        "evaluation_views": {
            "natural_test": "held-out task source-test transitions without category balancing",
            "holding_challenge_test": "target-aligned subset of the same held-out task source-test transitions",
            "challenge_is_subset_of_natural": True,
        },
        "feature_availability": _feature_availability(all_feature_rows),
        "global_source_counts": {
            "natural": dict(Counter(int(row["task_id"]) for row in natural_rows)),
            "target_aligned": dict(Counter(int(row["task_id"]) for row in target_rows)),
        },
        "warnings": global_warnings,
        "folds": folds,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = _expand_config_environment(
        json.loads(args.config.read_text(encoding="utf-8"))
    )
    result = build_manifest(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "warnings": len(result["warnings"]),
        "folds": [
            {"name": fold["name"], "sample_counts": fold["sample_counts"]}
            for fold in result["folds"]
        ],
        "feature_availability": result["feature_availability"],
    }, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
