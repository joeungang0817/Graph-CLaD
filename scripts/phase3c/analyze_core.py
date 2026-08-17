"""Analyze Phase 3C prediction artifacts with paired task bootstrap."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import write_json
from .metrics import evaluate_motion, evaluate_relation_predictions
from .contracts import PRIMARY_RELATIONS, canonical_sha256
from .provenance import runtime_provenance


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_paths(value: Any) -> list[Path]:
    raw_paths = list(value.values()) if isinstance(value, dict) else value
    if isinstance(raw_paths, (str, os.PathLike)):
        raw_paths = [raw_paths]
    if not isinstance(raw_paths, (list, tuple)) or not raw_paths:
        raise ValueError("prediction file entry must be a path, path list, or fold mapping")
    return [
        Path(os.path.expandvars(str(raw_path))).expanduser() for raw_path in raw_paths
    ]


def load_predictions(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    if not rows or rows[0].get("schema") != "phase3c-core-prediction.v2":
        return rows
    grouped: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("schema") != "phase3c-core-prediction.v2":
            raise ValueError("prediction artifact mixes incompatible row schemas")
        sample_id = str(row.get("sample_id"))
        fold = str(row.get("fold", f"test_task{row.get('task_id')}"))
        seed = int(row.get("seed", 0))
        identity = (fold, seed, sample_id)
        relation = str(row.get("relation"))
        if relation not in PRIMARY_RELATIONS:
            raise ValueError(f"unknown prediction relation: {relation}")
        if relation in grouped[identity]:
            raise ValueError(f"duplicate prediction row: {identity}/{relation}")
        grouped[identity][relation] = row
    pivoted: list[dict[str, Any]] = []
    for (fold, seed, sample_id), per_relation in grouped.items():
        missing = [name for name in PRIMARY_RELATIONS if name not in per_relation]
        if missing:
            raise ValueError(f"prediction sample {sample_id} is missing relations {missing}")
        ordered = [per_relation[name] for name in PRIMARY_RELATIONS]
        first = ordered[0]
        for row in ordered[1:]:
            for key in (
                "task_id",
                "episode_id",
                "model_id",
                "prev_step",
                "current_step",
                "target_step",
                "tau",
                "scene_motion",
                "target_scene_motion",
            ):
                if row.get(key) != first.get(key):
                    raise ValueError(f"prediction sample {sample_id} has inconsistent {key}")
        pivoted.append({
            "sample_id": sample_id,
            "paired_key": f"{fold}|seed{seed}|{sample_id}",
            "model_id": first.get("model_id"),
            "fold": fold,
            "seed": seed,
            "task_id": first["task_id"],
            "episode_id": first["episode_id"],
            "prev_step": int(first.get("prev_step", first.get("current_step", 6) - first.get("tau", 6))),
            "current_step": int(first.get("current_step", -1)),
            "target_step": int(first.get("target_step", -1)),
            "tau": int(first.get("tau", 6)),
            "relation_logits": [float(row["logit"]) for row in ordered],
            "target_relation_change": [0 if row["target"] is None else int(row["target"]) for row in ordered],
            "target_relation_mask": [int(bool(row["evaluated"])) for row in ordered],
            "fixed_thresholds": [float(row["threshold"]) for row in ordered],
            "scene_motion": float(first["scene_motion"]),
            "target_scene_motion": float(first["target_scene_motion"]),
            "train_prevalence": (
                [float(row["train_prevalence"]) for row in ordered]
                if all(row.get("train_prevalence") is not None for row in ordered)
                else None
            ),
        })
    return pivoted


def load_prediction_collection(value: Any) -> list[dict[str, Any]]:
    """Load one path, a path list, or a fold-to-path mapping for one model."""

    rows: list[dict[str, Any]] = []
    for path in _prediction_paths(value):
        rows.extend(load_predictions(path))
    keys = [str(row.get("paired_key", row.get("sample_id"))) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("prediction collection contains duplicate fold/seed/sample keys")
    return rows


def _baseline_rows(
    rows: list[dict[str, Any]], *, mode: str
) -> list[dict[str, Any]] | None:
    result: list[dict[str, Any]] = []
    for row in rows:
        if mode == "no_change":
            probability = np.zeros(len(PRIMARY_RELATIONS), dtype=np.float64)
        elif mode == "train_prevalence":
            raw = row.get("train_prevalence")
            if raw is None:
                return None
            probability = np.asarray(raw, dtype=np.float64)
            if probability.shape != (len(PRIMARY_RELATIONS),):
                raise ValueError("train_prevalence must contain one value per relation")
        else:
            raise ValueError(f"unknown trivial baseline: {mode}")
        clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
        baseline = dict(row)
        baseline["relation_logits"] = np.log(clipped / (1.0 - clipped)).tolist()
        baseline["fixed_thresholds"] = [0.5] * len(PRIMARY_RELATIONS)
        result.append(baseline)
    return result


def score_rows(
    rows: list[dict[str, Any]], *, include_baselines: bool = True
) -> dict[str, Any]:
    if not rows:
        return {"status": "empty"}
    fixed_thresholds = None
    threshold_rows = [row.get("fixed_thresholds") for row in rows]
    if any(value is not None for value in threshold_rows):
        if any(value is None for value in threshold_rows):
            raise ValueError("prediction rows mix fixed and optimized thresholds")
        fixed_thresholds = [
            [float(value) for value in threshold_row]
            for threshold_row in threshold_rows
        ]
    result = {
        "status": "completed",
        "rows": len(rows),
        "folds": sorted(set(str(row.get("fold", "unknown")) for row in rows)),
        "seeds": sorted(set(int(row.get("seed", 0)) for row in rows)),
        "relation": evaluate_relation_predictions(
            [row["relation_logits"] for row in rows],
            [row["target_relation_change"] for row in rows],
            [row["target_relation_mask"] for row in rows],
            fixed_thresholds=fixed_thresholds,
        ),
        "motion": evaluate_motion(
            [row["scene_motion"] for row in rows],
            [row["target_scene_motion"] for row in rows],
        ),
    }
    if include_baselines:
        baselines: dict[str, Any] = {}
        for mode in ("no_change", "train_prevalence"):
            baseline = _baseline_rows(rows, mode=mode)
            baselines[mode] = (
                score_rows(baseline, include_baselines=False)
                if baseline is not None
                else {"status": "unavailable", "reason": "missing_train_prevalence"}
            )
        result["trivial_baselines"] = baselines
    return result


def _task_episode_groups(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        task = str(row.get("task_id"))
        episode = row.get("episode_id")
        if episode is None or str(episode) == "":
            raise ValueError("hierarchical bootstrap requires episode_id on every row")
        groups[task][str(episode)].append(row)
    return {
        task: dict(episodes)
        for task, episodes in groups.items()
    }


def hierarchical_bootstrap_difference(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    *,
    metric: str = "macro_pr_auc",
    replicates: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    if int(replicates) <= 0:
        raise ValueError("replicates must be positive")
    def paired_key(row: dict[str, Any]) -> str:
        return str(row.get("paired_key", row["sample_id"]))

    if len({paired_key(row) for row in candidate}) != len(candidate):
        raise ValueError("candidate predictions contain duplicate paired keys")
    if len({paired_key(row) for row in baseline}) != len(baseline):
        raise ValueError("baseline predictions contain duplicate paired keys")
    left = {paired_key(row): row for row in candidate}
    right = {paired_key(row): row for row in baseline}
    left_ids = set(left)
    right_ids = set(right)
    if left_ids != right_ids:
        missing_candidate = sorted(right_ids - left_ids)[:5]
        missing_baseline = sorted(left_ids - right_ids)[:5]
        raise ValueError(
            "paired prediction sample_id sets differ: "
            f"candidate_missing={missing_candidate}, baseline_missing={missing_baseline}"
        )
    paired_ids = sorted(left_ids)
    for sample_id in paired_ids:
        if str(left[sample_id].get("task_id")) != str(right[sample_id].get("task_id")):
            raise ValueError(f"paired prediction task mismatch for sample_id={sample_id}")
        if str(left[sample_id].get("episode_id")) != str(right[sample_id].get("episode_id")):
            raise ValueError(f"paired prediction episode mismatch for sample_id={sample_id}")
        if str(left[sample_id].get("fold")) != str(right[sample_id].get("fold")):
            raise ValueError(f"paired prediction fold mismatch for sample_id={sample_id}")
        if int(left[sample_id].get("seed", 0)) != int(right[sample_id].get("seed", 0)):
            raise ValueError(f"paired prediction seed mismatch for sample_id={sample_id}")
    aligned_left = [left[key] for key in paired_ids]
    aligned_right = [right[key] for key in paired_ids]
    tasks = sorted(set(str(row.get("task_id")) for row in aligned_left))
    rows_left = _task_episode_groups(aligned_left)
    rows_right = _task_episode_groups(aligned_right)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(int(replicates)):
        if not tasks:
            break
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        left_task_scores: list[float] = []
        right_task_scores: list[float] = []
        for task in sampled_tasks:
            task_key = str(task)
            episodes = sorted(rows_left[task_key])
            if set(episodes) != set(rows_right[task_key]):
                raise ValueError(f"paired prediction episode sets differ for task={task_key}")
            if not episodes:
                continue
            sampled_episodes = rng.choice(episodes, size=len(episodes), replace=True)
            sampled_left: list[dict[str, Any]] = []
            sampled_right: list[dict[str, Any]] = []
            for episode in sampled_episodes:
                episode_key = str(episode)
                sampled_left.extend(rows_left[task_key][episode_key])
                sampled_right.extend(rows_right[task_key][episode_key])
            left_score = score_rows(sampled_left, include_baselines=False)["relation"].get(metric)
            right_score = score_rows(sampled_right, include_baselines=False)["relation"].get(metric)
            if left_score is not None and right_score is not None:
                left_task_scores.append(float(left_score))
                right_task_scores.append(float(right_score))
        if left_task_scores and right_task_scores:
            values.append(float(np.mean(left_task_scores) - np.mean(right_task_scores)))
    estimate = None
    if aligned_left and aligned_right:
        left_task_scores = [
            score_rows([row for episode in rows_left[task].values() for row in episode], include_baselines=False)["relation"].get(metric)
            for task in tasks
        ]
        right_task_scores = [
            score_rows([row for episode in rows_right[task].values() for row in episode], include_baselines=False)["relation"].get(metric)
            for task in tasks
        ]
        if all(score is not None for score in left_task_scores + right_task_scores):
            estimate = float(np.mean(left_task_scores) - np.mean(right_task_scores))
    return {
        "metric": metric,
        "estimate": estimate,
        "ci_2_5": float(np.quantile(values, 0.025)) if values else None,
        "ci_97_5": float(np.quantile(values, 0.975)) if values else None,
        "replicates": int(replicates),
        "paired_rows": len(paired_ids),
        "common_rows": len(paired_ids),
        "tasks": tasks,
        "episodes_per_task": {task: len(rows_left[task]) for task in tasks},
        "resampling_units": ["task", "episode"],
    }


def analyze(config: dict[str, Any]) -> dict[str, Any]:
    prediction_files = config.get("prediction_files")
    if not isinstance(prediction_files, dict) or not prediction_files:
        raise ValueError("analyze_core requires prediction_files mapping model_id -> path")
    scored: dict[str, Any] = {}
    loaded: dict[str, list[dict[str, Any]]] = {}
    input_artifacts: dict[str, Any] = {}
    for model_id, raw_path in prediction_files.items():
        paths = _prediction_paths(raw_path)
        rows = load_prediction_collection(raw_path)
        observed_model_ids = {
            str(row["model_id"])
            for row in rows
            if row.get("model_id") is not None
        }
        if observed_model_ids and observed_model_ids != {str(model_id)}:
            raise ValueError(
                f"prediction collection model identity mismatch for {model_id}: "
                f"{sorted(observed_model_ids)}"
            )
        loaded[str(model_id)] = rows
        scored[str(model_id)] = score_rows(rows)
        input_artifacts[str(model_id)] = [
            {"path": str(path), "sha256": _sha256_file(path)} for path in paths
        ]
    expected_folds = config.get("expected_folds")
    expected_seeds = config.get("expected_seeds")
    for model_id, rows in loaded.items():
        if expected_folds is not None and {
            str(row.get("fold")) for row in rows
        } != {str(value) for value in expected_folds}:
            raise ValueError(f"prediction collection has incomplete folds for {model_id}")
        if expected_seeds is not None and {
            int(row.get("seed", 0)) for row in rows
        } != {int(value) for value in expected_seeds}:
            raise ValueError(f"prediction collection has incomplete seeds for {model_id}")
    primary_model = str(config.get("primary_model", "C3-RelMPNN-PastAct"))
    baseline_model = str(config.get("baseline_model", "C3-Sem-PastAct"))
    comparator_values = config.get(
        "comparators", [model_id for model_id in loaded if model_id != primary_model]
    )
    if isinstance(comparator_values, (str, os.PathLike)):
        comparator_values = [comparator_values]
    if not isinstance(comparator_values, (list, tuple)):
        raise ValueError("comparators must be a model-id list")
    comparators = [str(value) for value in comparator_values]
    replicates = int(config.get("replicates", 2000))
    seed = int(config.get("seed", 0))

    def comparisons_for(
        rows_by_model: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        reports: dict[str, Any] = {}
        if primary_model not in rows_by_model:
            return reports
        for comparator in comparators:
            if comparator not in rows_by_model or comparator == primary_model:
                continue
            relation_difference = hierarchical_bootstrap_difference(
                rows_by_model[primary_model],
                rows_by_model[comparator],
                metric="macro_pr_auc",
                replicates=replicates,
                seed=seed,
            )
            family_difference = hierarchical_bootstrap_difference(
                rows_by_model[primary_model],
                rows_by_model[comparator],
                metric="family_macro_pr_auc",
                replicates=replicates,
                seed=seed,
            )
            reports[comparator] = {
                "candidate": primary_model,
                "baseline": comparator,
                "relation_macro": relation_difference,
                "family_macro": family_difference,
            }
        return reports

    comparisons = comparisons_for(loaded)
    initial_filtered = {
        model_id: [row for row in rows if int(row.get("prev_step", -1)) != 0]
        for model_id, rows in loaded.items()
    }
    sensitivity = {
        "policy": "exclude_all_prev_step_zero_samples",
        "predicate": "prev_step != 0",
        "removed_rows": {
            model_id: len(loaded[model_id]) - len(rows)
            for model_id, rows in initial_filtered.items()
        },
        "models": {
            model_id: score_rows(rows) for model_id, rows in initial_filtered.items()
        },
        "comparisons": comparisons_for(initial_filtered),
    }
    comparison = comparisons.get(baseline_model)
    result = {
        "schema": "phase3c-core-analysis.v3",
        "status": "completed",
        "config_sha256": canonical_sha256(config),
        "analyzer_source_sha256": _sha256_file(Path(__file__)),
        "runtime_provenance": runtime_provenance(),
        "input_artifacts": input_artifacts,
        "primary_model": primary_model,
        "models": scored,
        "comparisons": comparisons,
        "comparison": comparison,
        "initial_transition_sensitivity": sensitivity,
    }
    output = config.get("output")
    if output:
        write_json(Path(os.path.expandvars(str(output))).expanduser(), result)
    return result


def main() -> None:  # pragma: no cover - SSH analysis CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    from .io import load_json_config

    print(json.dumps(analyze(load_json_config(args.config)), ensure_ascii=False))


if __name__ == "__main__":
    main()
