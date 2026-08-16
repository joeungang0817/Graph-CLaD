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
    return rows


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "empty"}
    return {
        "status": "completed",
        "rows": len(rows),
        "relation": evaluate_relation_predictions(
            [row["relation_logits"] for row in rows],
            [row["target_relation_change"] for row in rows],
            [row["target_relation_mask"] for row in rows],
        ),
        "motion": evaluate_motion(
            [row["scene_motion"] for row in rows],
            [row["target_scene_motion"] for row in rows],
        ),
    }


def _task_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task = str(row.get("task_id"))
        groups[task].append(row)
    return dict(groups)


def hierarchical_bootstrap_difference(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    *,
    metric: str = "macro_pr_auc",
    replicates: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    left = {str(row["sample_id"]): row for row in candidate}
    right = {str(row["sample_id"]): row for row in baseline}
    common_ids = sorted(set(left) & set(right))
    aligned_left = [left[key] for key in common_ids]
    aligned_right = [right[key] for key in common_ids]
    tasks = sorted(set(str(row.get("task_id")) for row in aligned_left))
    rows_left = _task_groups(aligned_left)
    rows_right = _task_groups(aligned_right)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(int(replicates)):
        if not tasks:
            break
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        left_task_scores: list[float] = []
        right_task_scores: list[float] = []
        for task in sampled_tasks:
            left_score = score_rows(rows_left[str(task)])["relation"].get(metric)
            right_score = score_rows(rows_right[str(task)])["relation"].get(metric)
            if left_score is not None and right_score is not None:
                left_task_scores.append(float(left_score))
                right_task_scores.append(float(right_score))
        if left_task_scores and right_task_scores:
            values.append(float(np.mean(left_task_scores) - np.mean(right_task_scores)))
    estimate = None
    if aligned_left and aligned_right:
        left_task_scores = [score_rows(rows_left[task])["relation"].get(metric) for task in tasks]
        right_task_scores = [score_rows(rows_right[task])["relation"].get(metric) for task in tasks]
        if all(score is not None for score in left_task_scores + right_task_scores):
            estimate = float(np.mean(left_task_scores) - np.mean(right_task_scores))
    return {
        "metric": metric,
        "estimate": estimate,
        "ci_2_5": float(np.quantile(values, 0.025)) if values else None,
        "ci_97_5": float(np.quantile(values, 0.975)) if values else None,
        "replicates": int(replicates),
        "common_rows": len(common_ids),
        "tasks": tasks,
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
        comparison = {
            "candidate": primary_model,
            "baseline": baseline_model,
            "relation": hierarchical_bootstrap_difference(
                loaded[primary_model], loaded[baseline_model],
                metric="macro_pr_auc", replicates=int(config.get("replicates", 2000)), seed=int(config.get("seed", 0)),
            ),
        }
    result = {"schema": "phase3c-core-analysis.v1", "status": "completed", "models": scored, "comparison": comparison}
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
