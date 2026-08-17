"""Analyze Phase 3C prediction artifacts with paired task bootstrap."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import write_json
from .metrics import evaluate_motion, evaluate_relation_predictions
from .contracts import PRIMARY_RELATIONS


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
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("schema") != "phase3c-core-prediction.v2":
            raise ValueError("prediction artifact mixes incompatible row schemas")
        sample_id = str(row.get("sample_id"))
        relation = str(row.get("relation"))
        if relation not in PRIMARY_RELATIONS:
            raise ValueError(f"unknown prediction relation: {relation}")
        if relation in grouped[sample_id]:
            raise ValueError(f"duplicate prediction row: {sample_id}/{relation}")
        grouped[sample_id][relation] = row
    pivoted: list[dict[str, Any]] = []
    for sample_id, per_relation in grouped.items():
        missing = [name for name in PRIMARY_RELATIONS if name not in per_relation]
        if missing:
            raise ValueError(f"prediction sample {sample_id} is missing relations {missing}")
        ordered = [per_relation[name] for name in PRIMARY_RELATIONS]
        first = ordered[0]
        for row in ordered[1:]:
            for key in ("task_id", "episode_id", "scene_motion", "target_scene_motion"):
                if row.get(key) != first.get(key):
                    raise ValueError(f"prediction sample {sample_id} has inconsistent {key}")
        pivoted.append({
            "sample_id": sample_id,
            "task_id": first["task_id"],
            "episode_id": first["episode_id"],
            "relation_logits": [float(row["logit"]) for row in ordered],
            "target_relation_change": [0 if row["target"] is None else int(row["target"]) for row in ordered],
            "target_relation_mask": [int(bool(row["evaluated"])) for row in ordered],
            "fixed_thresholds": [float(row["threshold"]) for row in ordered],
            "scene_motion": float(first["scene_motion"]),
            "target_scene_motion": float(first["target_scene_motion"]),
        })
    return pivoted


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "empty"}
    fixed_thresholds = None
    threshold_rows = [row.get("fixed_thresholds") for row in rows]
    if any(value is not None for value in threshold_rows):
        if any(value is None for value in threshold_rows):
            raise ValueError("prediction rows mix fixed and optimized thresholds")
        first_thresholds = [float(value) for value in threshold_rows[0]]
        if any([float(value) for value in row] != first_thresholds for row in threshold_rows[1:]):
            raise ValueError("fixed thresholds differ across prediction samples")
        fixed_thresholds = first_thresholds
    return {
        "status": "completed",
        "rows": len(rows),
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
    if len({str(row["sample_id"]) for row in candidate}) != len(candidate):
        raise ValueError("candidate predictions contain duplicate sample_id values")
    if len({str(row["sample_id"]) for row in baseline}) != len(baseline):
        raise ValueError("baseline predictions contain duplicate sample_id values")
    left = {str(row["sample_id"]): row for row in candidate}
    right = {str(row["sample_id"]): row for row in baseline}
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
            left_score = score_rows(sampled_left)["relation"].get(metric)
            right_score = score_rows(sampled_right)["relation"].get(metric)
            if left_score is not None and right_score is not None:
                left_task_scores.append(float(left_score))
                right_task_scores.append(float(right_score))
        if left_task_scores and right_task_scores:
            values.append(float(np.mean(left_task_scores) - np.mean(right_task_scores)))
    estimate = None
    if aligned_left and aligned_right:
        left_task_scores = [
            score_rows([row for episode in rows_left[task].values() for row in episode])["relation"].get(metric)
            for task in tasks
        ]
        right_task_scores = [
            score_rows([row for episode in rows_right[task].values() for row in episode])["relation"].get(metric)
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
    for model_id, raw_path in prediction_files.items():
        path = Path(os.path.expandvars(str(raw_path))).expanduser()
        rows = load_predictions(path)
        loaded[str(model_id)] = rows
        scored[str(model_id)] = score_rows(rows)
    primary_model = str(config.get("primary_model", "C3-RelMPNN-PastAct"))
    baseline_model = str(config.get("baseline_model", "C3-Sem-PastAct"))
    comparison = None
    if primary_model in loaded and baseline_model in loaded:
        relation_difference = hierarchical_bootstrap_difference(
            loaded[primary_model], loaded[baseline_model],
            metric="macro_pr_auc", replicates=int(config.get("replicates", 2000)), seed=int(config.get("seed", 0)),
        )
        family_difference = hierarchical_bootstrap_difference(
            loaded[primary_model], loaded[baseline_model],
            metric="family_macro_pr_auc", replicates=int(config.get("replicates", 2000)), seed=int(config.get("seed", 0)),
        )
        comparison = {
            "candidate": primary_model,
            "baseline": baseline_model,
            "relation": relation_difference,
            "relation_macro": relation_difference,
            "family_macro": family_difference,
        }
    result = {"schema": "phase3c-core-analysis.v2", "status": "completed", "models": scored, "comparison": comparison}
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
