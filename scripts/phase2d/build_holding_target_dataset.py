"""Build the target-aligned holding dataset used by the Phase 3 experiments.

The source is the input-clean official-demo state-replay release.  Selection
uses only current and target oracle relation labels to form four audit strata:
future holding positive, holding changed, contact-without-holding hard
negative, and background.  Sampling metadata is retained for auditing but is
not part of the model input graph.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


DEFAULT_CAPS = {
    "future_holding_positive": 12,
    "holding_changed": 12,
    "hard_negative": 12,
    "background": 6,
}
TARGET_STRATA = tuple(DEFAULT_CAPS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_name(record: Mapping[str, Any]) -> str:
    split = record.get("split") or {}
    if isinstance(split, Mapping):
        return str(split.get("in_task", "unknown"))
    return str(split)


def _relation_pair_map(
    graph: Mapping[str, Any],
    relation: str = "holding",
    source_id: str = "robot0",
) -> dict[str, dict[str, bool | None]]:
    result: dict[str, dict[str, bool | None]] = {}
    for edge in graph.get("edges", []):
        if str(edge.get("source")) != source_id:
            continue
        record = (edge.get("semantic_relation") or {}).get(relation) or (
            edge.get("relations") or {}
        ).get(relation)
        if not isinstance(record, Mapping):
            continue
        valid = bool(record.get("valid"))
        result[str(edge.get("target"))] = {
            "valid": valid,
            "value": bool(record.get("value")) if valid else None,
        }
    return result


def _event_tags(
    record: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> list[str]:
    tags: set[str] = set()
    start_step = int(sample.get("start_step", 0))
    target_step = int(sample.get("target_step", start_step))
    for event in record.get("events", []):
        if not isinstance(event, Mapping):
            continue
        step = int(event.get("step", -1))
        if not start_step < step <= target_step:
            continue
        if event.get("event") != "holding_state_change":
            continue
        if event.get("to_state") == "holding":
            tags.add("holding_onset")
        if event.get("from_state") == "holding" and event.get("to_state") == "release":
            tags.add("holding_release")
    return sorted(tags)


def classify_target_sample(
    record: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    """Return target-aligned sampling metadata for one graph transition.

    This reproduces the Colab v2 contract: if any robot-object holding pair is
    missing or invalid at either endpoint, the whole transition is marked
    ambiguous and excluded from the target-aligned strata.
    """

    current = _relation_pair_map(sample.get("graph_t") or {})
    future = _relation_pair_map(sample.get("graph_target") or {})
    objects = sorted(set(current) | set(future))
    unknown = False
    future_positive_objects: list[str] = []
    changed_objects: list[str] = []
    for object_id in objects:
        current_label = current.get(object_id)
        future_label = future.get(object_id)
        if (
            current_label is None
            or future_label is None
            or not current_label["valid"]
            or not future_label["valid"]
        ):
            unknown = True
            continue
        if future_label["value"]:
            future_positive_objects.append(object_id)
        if current_label["value"] != future_label["value"]:
            changed_objects.append(object_id)

    contact_objects: set[str] = set()
    for graph in (sample.get("graph_t") or {}, sample.get("graph_target") or {}):
        for object_id, label in _relation_pair_map(graph, relation="contact").items():
            if label["valid"] and label["value"]:
                contact_objects.add(object_id)

    hard_negative_objects: list[str] = []
    if not unknown:
        for object_id in objects:
            current_label = current[object_id]
            future_label = future[object_id]
            holding_false_both = (
                current_label["valid"]
                and future_label["valid"]
                and not current_label["value"]
                and not future_label["value"]
            )
            if (
                object_id in contact_objects
                and holding_false_both
                and object_id not in changed_objects
                and object_id not in future_positive_objects
            ):
                hard_negative_objects.append(object_id)

    if unknown:
        categories = ["ambiguous"]
    else:
        categories: list[str] = []
        if future_positive_objects:
            categories.append("future_holding_positive")
        if changed_objects:
            categories.append("holding_changed")
        if hard_negative_objects:
            categories.append("hard_negative")
        if not categories:
            categories.append("background")

    return {
        "target_categories": categories,
        "target_category": (
            "holding_changed" if "holding_changed" in categories else categories[0]
        ),
        "target_event_tags": _event_tags(record, sample),
        "target_event_objects": sorted(
            set(future_positive_objects + changed_objects + hard_negative_objects)
        ),
        "target_sampling_source": "official_hdf5_state_replay",
    }


def _sample_key(sample: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(sample.get("episode_id")),
        int(sample.get("start_step", 0)),
        int(sample.get("target_step", 0)),
        int(sample.get("tau", sample.get("horizon", 0))),
    )


def select_record_samples(
    record: Mapping[str, Any],
    caps: Mapping[str, int] = DEFAULT_CAPS,
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    """Decorate and cap one demo record by category and horizon."""

    decorated: list[dict[str, Any]] = []
    for original_sample in record.get("samples", []):
        sample = dict(original_sample)
        sample.update(classify_target_sample(record, sample))
        decorated.append(sample)

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    available: Counter[str] = Counter()
    for sample in decorated:
        episode_id = str(sample.get("episode_id", "unknown"))
        horizon = int(sample.get("tau", sample.get("horizon", 0)))
        for category in sample["target_categories"]:
            if category == "ambiguous":
                continue
            available[category] += 1
            grouped[(episode_id, category, horizon)].append(sample)

    selected_by_key: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for (_, category, _), rows in sorted(grouped.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                int(row.get("start_step", 0)),
                int(row.get("target_step", 0)),
                int(row.get("tau", row.get("horizon", 0))),
            ),
        )
        for row in ordered[: int(caps.get(category, 0))]:
            selected_by_key[_sample_key(row)] = row

    selected = sorted(
        selected_by_key.values(),
        key=lambda row: (
            int(row.get("start_step", 0)),
            int(row.get("target_step", 0)),
            int(row.get("tau", row.get("horizon", 0))),
        ),
    )
    selected_counts: Counter[str] = Counter(
        category
        for row in selected
        for category in row.get("target_categories", [])
        if category != "ambiguous"
    )
    return selected, selected_counts, available


def build_target_dataset(
    task_files: Mapping[int, Path],
    output_root: Path,
    caps: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Materialize target-aligned task files, index, config, and manifest."""

    caps = dict(caps or DEFAULT_CAPS)
    output_root.mkdir(parents=True, exist_ok=True)
    all_index_rows: list[dict[str, Any]] = []
    all_coverage: Counter[str] = Counter()
    all_available: Counter[str] = Counter()
    task_reports: dict[str, Any] = {}

    for task_id, source_path in sorted(task_files.items()):
        output_path = (
            output_root
            / f"task{task_id}"
            / f"phase2d_task{task_id}_graph_dataset.jsonl.gz"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        task_coverage: Counter[str] = Counter()
        task_available: Counter[str] = Counter()
        task_records = 0
        task_samples = 0
        with gzip.open(source_path, "rt", encoding="utf-8") as source_file, gzip.open(
            temporary, "wt", encoding="utf-8", compresslevel=1
        ) as output_file:
            for line in source_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                selected, selected_counts, available = select_record_samples(record, caps)
                task_available.update(available)
                all_available.update(available)
                if not selected:
                    continue
                split_name = _split_name(record)
                for sample in selected:
                    for category in sample["target_categories"]:
                        if category == "ambiguous":
                            continue
                        key = f"task{task_id}:{split_name}:{category}"
                        task_coverage[key] += 1
                        all_coverage[key] += 1
                    all_index_rows.append(
                        {
                            "task_id": task_id,
                            "demo_id": record.get("demo_id"),
                            "demo_key": record.get("demo_key"),
                            "episode_id": sample.get("episode_id"),
                            "start_step": sample.get("start_step"),
                            "target_step": sample.get("target_step"),
                            "horizon": sample.get("horizon", sample.get("tau")),
                            "split": split_name,
                            "target_category": sample.get("target_category"),
                            "target_categories": sample.get("target_categories", []),
                            "target_event_tags": sample.get("target_event_tags", []),
                            "target_event_objects": sample.get("target_event_objects", []),
                            "source": "official_hdf5_state_replay",
                        }
                    )
                output_record = dict(record)
                output_record["samples"] = selected
                output_record["target_sampling_version"] = "phase2d-holding-target-v2"
                output_record["target_sample_counts"] = dict(selected_counts)
                output_file.write(
                    json.dumps(
                        output_record,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                task_records += 1
                task_samples += len(selected)
        temporary.replace(output_path)
        task_reports[str(task_id)] = {
            "source": str(source_path),
            "source_sha256": _sha256(source_path),
            "output": str(output_path),
            "records": task_records,
            "samples": task_samples,
            "available_by_category": dict(task_available),
            "selected_by_category": dict(sorted(task_coverage.items())),
            "written_records": task_records,
        }

    index_path = output_root / "phase2d_holding_target_window_index_v1.jsonl.gz"
    index_temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    with gzip.open(index_temporary, "wt", encoding="utf-8", compresslevel=1) as handle:
        for row in all_index_rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    index_temporary.replace(index_path)

    config = {
        "artifact_version": "phase2d-holding-target-v2",
        "source_kind": "official_hdf5_state_replay",
        "holding_relation": "primary",
        "inside_relation": "deferred",
        "unknown_policy": "exclude_transition_if_any_robot_object_holding_pair_is_unknown",
        "fixed_demo_split_preserved": True,
        "caps_per_demo_horizon": caps,
        "selection": "target_aligned_union_strata",
        "target_strata": list(TARGET_STRATA),
    }
    config_text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    (output_root / "generation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    warnings: list[str] = []
    for task_id in sorted(task_files):
        for split in ("train", "validation", "test"):
            for category in ("future_holding_positive", "holding_changed"):
                key = f"task{task_id}:{split}:{category}"
                if all_coverage.get(key, 0) == 0:
                    warnings.append(f"missing_support:{key}")
    manifest = {
        **config,
        "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "tasks": task_reports,
        "coverage": dict(sorted(all_coverage.items())),
        "available_total_by_category": dict(all_available),
        "index": str(index_path),
        "index_rows": len(all_index_rows),
        "warnings": warnings,
        "status": "pass" if not warnings else "pass_with_warnings",
    }
    (output_root / "phase2d_holding_target_dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _task_mapping(values: Iterable[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        task_id, separator, path = value.partition("=")
        if not separator:
            raise ValueError("--task must use TASK_ID=/path/to/input.jsonl.gz")
        result[int(task_id)] = Path(path)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--future-positive-cap", type=int, default=12)
    parser.add_argument("--holding-changed-cap", type=int, default=12)
    parser.add_argument("--hard-negative-cap", type=int, default=12)
    parser.add_argument("--background-cap", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = build_target_dataset(
        _task_mapping(args.task),
        args.output_root,
        caps={
            "future_holding_positive": args.future_positive_cap,
            "holding_changed": args.holding_changed_cap,
            "hard_negative": args.hard_negative_cap,
            "background": args.background_cap,
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
