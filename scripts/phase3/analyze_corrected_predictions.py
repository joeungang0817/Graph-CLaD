"""Analyze corrected Phase 3 prediction artifacts with clustered bootstrap.

Seeds are treated as repeated fits on the same held-out episodes, not as
independent test samples.  The report first stores same-fold/same-seed paired
differences, then averages seeds within each task fold.  Confidence intervals
resample task folds, episodes within a task, and explicit event clusters when
available (otherwise the stable sample ID is recorded as an event proxy).

The legacy ``--result`` mode compares two models recorded in one result JSON.
The ``--left-result``/``--right-result`` mode compares models whose result JSONs
and prediction artifact roots are stored separately, as in the aligned versus
train-shuffled action control.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

import numpy as np


METRICS = (
    "event_pr_auc",
    "event_f1",
    "release_f1",
    "hard_negative_fpr",
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _average_precision(truth: np.ndarray, score: np.ndarray) -> float | None:
    truth = truth.astype(bool)
    if int(truth.sum()) == 0:
        return None
    order = np.argsort(-score.astype(np.float64), kind="mergesort")
    ordered = truth[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].mean())


def _f1(truth: np.ndarray, prediction: np.ndarray) -> float | None:
    if not truth.size:
        return None
    truth = truth.astype(bool)
    prediction = prediction.astype(bool)
    true_positive = int(np.logical_and(truth, prediction).sum())
    false_positive = int(np.logical_and(~truth, prediction).sum())
    false_negative = int(np.logical_and(truth, ~prediction).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return float(2.0 * precision * recall / max(precision + recall, 1e-12))


def _metric(rows: Sequence[Mapping[str, Any]], name: str) -> float | None:
    if not rows:
        return None
    if name == "event_pr_auc":
        return _average_precision(
            np.asarray([row["change_target"] for row in rows]),
            np.asarray(
                [
                    row["conditional_oracle_current_change_probability"]
                    for row in rows
                ]
            ),
        )
    if name == "event_f1":
        return _f1(
            np.asarray([row["change_target"] for row in rows]),
            np.asarray(
                [
                    row["conditional_oracle_current_change_prediction"]
                    for row in rows
                ]
            ),
        )
    if name == "release_f1":
        selected = [row for row in rows if int(row["current_target"]) == 1]
        return _f1(
            np.asarray([row["release_target"] for row in selected]),
            np.asarray([1 - int(row["future_prediction"]) for row in selected]),
        )
    if name == "hard_negative_fpr":
        selected = [row for row in rows if int(row["hard_negative"]) == 1]
        if not selected:
            return None
        return float(
            np.mean([int(row["future_prediction"]) for row in selected])
        )
    raise ValueError(name)


def _paired_rows(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def key(row: Mapping[str, Any]) -> tuple[str, str, str]:
        return (
            str(row["sample_id"]),
            str(row["edge_source"]),
            str(row["edge_target"]),
        )

    left_by_key = {key(row): dict(row) for row in left}
    right_by_key = {key(row): dict(row) for row in right}
    shared = sorted(set(left_by_key) & set(right_by_key))
    if len(shared) != len(left_by_key) or len(shared) != len(right_by_key):
        raise ValueError(
            "paired prediction artifacts do not contain identical sample/edge keys"
        )
    return [left_by_key[item] for item in shared], [right_by_key[item] for item in shared]


def _resample_clustered_pair(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    left, right = _paired_rows(left, right)
    by_episode: dict[str, list[int]] = {}
    for index, row in enumerate(left):
        by_episode.setdefault(str(row["episode_id"]), []).append(index)
    episodes = sorted(by_episode)
    sampled_episodes = rng.choice(episodes, size=len(episodes), replace=True)
    left_output: list[dict[str, Any]] = []
    right_output: list[dict[str, Any]] = []
    for episode in sampled_episodes:
        indices = by_episode[str(episode)]
        by_event: dict[str, list[int]] = {}
        for index in indices:
            by_event.setdefault(
                str(left[index]["event_cluster_id"]), []
            ).append(index)
        events = sorted(by_event)
        sampled_events = rng.choice(events, size=len(events), replace=True)
        for event in sampled_events:
            for index in by_event[str(event)]:
                left_output.append(left[index])
                right_output.append(right[index])
    return left_output, right_output


def _artifact_for(
    result: Mapping[str, Any],
    view: str = "natural_test",
    mode: str = "correct",
    prediction_root: Path | None = None,
) -> Path:
    entry = result["prediction_artifacts"][f"{view}.{mode}"]
    path = Path(str(entry["path"]))
    if prediction_root is None:
        return path
    candidates = (
        prediction_root / path.name,
        prediction_root / "predictions" / path.name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"prediction artifact {path.name!r} was not found under "
        f"{prediction_root}"
    )


def _result_rows(path: Path) -> list[Mapping[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report.get("results", [])
    if not isinstance(rows, list):
        raise ValueError(f"result JSON results must be a list: {path}")
    return [row for row in rows if isinstance(row, Mapping)]


def analyze(
    result_path: Path | None = None,
    *,
    left_result_path: Path | None = None,
    right_result_path: Path | None = None,
    left_prediction_root: Path | None = None,
    right_prediction_root: Path | None = None,
    left_model: str = "G1-late-action",
    right_model: str = "B1-v2",
    replicates: int = 2000,
    seed: int = 20260812,
) -> dict[str, Any]:
    if left_result_path is None:
        left_result_path = result_path
    if right_result_path is None:
        right_result_path = result_path
    if left_result_path is None or right_result_path is None:
        raise ValueError(
            "provide --result or both --left-result and --right-result"
        )

    left_results = _result_rows(left_result_path)
    right_results = _result_rows(right_result_path)
    left_by_run = {
        (str(row["fold"]), int(row["seed"]), str(row["comparison_id"])): row
        for row in left_results
    }
    right_by_run = {
        (str(row["fold"]), int(row["seed"]), str(row["comparison_id"])): row
        for row in right_results
    }
    folds = sorted(
        {
            str(row["fold"])
            for row in left_results
            if str(row["comparison_id"]) == left_model
        }
        | {
            str(row["fold"])
            for row in right_results
            if str(row["comparison_id"]) == right_model
        }
    )
    seeds_by_fold = {
        fold: sorted(
            {
                int(row["seed"])
                for row in left_results
                if str(row["fold"]) == fold
                and str(row["comparison_id"]) == left_model
            }
        )
        for fold in folds
    }
    paired: dict[tuple[str, int], tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    run_differences: dict[str, Any] = {}
    event_cluster_sources: set[str] = set()
    for fold in folds:
        for run_seed in seeds_by_fold[fold]:
            left_result = left_by_run.get((fold, run_seed, left_model))
            right_result = right_by_run.get((fold, run_seed, right_model))
            if left_result is None or right_result is None:
                continue
            left_rows, right_rows = _paired_rows(
                _read_rows(
                    _artifact_for(
                        left_result,
                        prediction_root=left_prediction_root,
                    )
                ),
                _read_rows(
                    _artifact_for(
                        right_result,
                        prediction_root=right_prediction_root,
                    )
                ),
            )
            paired[(fold, run_seed)] = (left_rows, right_rows)
            event_cluster_sources.update(
                str(row.get("event_cluster_source")) for row in left_rows
            )
            run_differences[f"{fold}.seed{run_seed}"] = {
                metric: (
                    None
                    if _metric(left_rows, metric) is None
                    or _metric(right_rows, metric) is None
                    else _metric(left_rows, metric) - _metric(right_rows, metric)
                )
                for metric in METRICS
            }

    if not paired:
        raise ValueError(
            "no same-fold/seed paired runs found for "
            f"{left_model!r} and {right_model!r}"
        )

    fold_seed_means: dict[str, Any] = {}
    for fold in folds:
        fold_seed_means[fold] = {}
        for metric in METRICS:
            values = [
                run_differences[f"{fold}.seed{run_seed}"][metric]
                for run_seed in seeds_by_fold[fold]
                if f"{fold}.seed{run_seed}" in run_differences
                and run_differences[f"{fold}.seed{run_seed}"][metric] is not None
            ]
            fold_seed_means[fold][metric] = mean(values) if values else None

    rng = np.random.default_rng(seed)
    bootstrap: dict[str, list[float]] = {metric: [] for metric in METRICS}
    usable_folds = [fold for fold in folds if any((fold, value) in paired for value in seeds_by_fold[fold])]
    for _ in range(int(replicates)):
        sampled_folds = rng.choice(
            usable_folds, size=len(usable_folds), replace=True
        )
        replicate_fold_values: dict[str, list[float]] = {
            metric: [] for metric in METRICS
        }
        for fold in sampled_folds:
            seed_values: dict[str, list[float]] = {
                metric: [] for metric in METRICS
            }
            for run_seed in seeds_by_fold[str(fold)]:
                pair = paired.get((str(fold), run_seed))
                if pair is None:
                    continue
                left_rows, right_rows = _resample_clustered_pair(
                    pair[0], pair[1], rng
                )
                for metric in METRICS:
                    left_value = _metric(left_rows, metric)
                    right_value = _metric(right_rows, metric)
                    if left_value is not None and right_value is not None:
                        seed_values[metric].append(left_value - right_value)
            for metric in METRICS:
                if seed_values[metric]:
                    replicate_fold_values[metric].append(
                        mean(seed_values[metric])
                    )
        for metric in METRICS:
            if replicate_fold_values[metric]:
                bootstrap[metric].append(mean(replicate_fold_values[metric]))

    intervals = {
        metric: {
            "estimate": mean(
                [
                    values[metric]
                    for values in fold_seed_means.values()
                    if values[metric] is not None
                ]
            )
            if any(values[metric] is not None for values in fold_seed_means.values())
            else None,
            "ci_2_5": (
                float(np.quantile(bootstrap[metric], 0.025))
                if bootstrap[metric]
                else None
            ),
            "ci_97_5": (
                float(np.quantile(bootstrap[metric], 0.975))
                if bootstrap[metric]
                else None
            ),
            "replicates": len(bootstrap[metric]),
        }
        for metric in METRICS
    }
    return {
        "protocol": "phase3B-R1-corrected-hierarchical-bootstrap-v1",
        "source_result": str(result_path) if result_path is not None else None,
        "left_result": str(left_result_path),
        "right_result": str(right_result_path),
        "left_prediction_root": (
            str(left_prediction_root) if left_prediction_root is not None else None
        ),
        "right_prediction_root": (
            str(right_prediction_root) if right_prediction_root is not None else None
        ),
        "comparison": f"{left_model}_minus_{right_model}",
        "view": "natural_test",
        "mode": "correct",
        "run_paired_differences": run_differences,
        "task_fold_seed_means": fold_seed_means,
        "hierarchical_bootstrap": intervals,
        "resampling_hierarchy": ["task_fold", "episode", "event_cluster"],
        "seed_handling": "metric differences are averaged within task fold; seeds are not independent test units",
        "event_cluster_sources": sorted(event_cluster_sources),
        "event_cluster_limitation": (
            "sample_id_proxy is used when the dataset has no explicit event_id; "
            "this does not collapse overlapping windows from one physical event"
            if "sample_id_proxy" in event_cluster_sources
            else None
        ),
        "bootstrap_seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        help="Legacy result JSON containing both comparison models.",
    )
    parser.add_argument(
        "--left-result",
        type=Path,
        help="Result JSON for the left model when reports are separate.",
    )
    parser.add_argument(
        "--right-result",
        type=Path,
        help="Result JSON for the right model when reports are separate.",
    )
    parser.add_argument(
        "--left-prediction-root",
        type=Path,
        help="Root containing the left prediction artifact, or its predictions/ directory.",
    )
    parser.add_argument(
        "--right-prediction-root",
        type=Path,
        help="Root containing the right prediction artifact, or its predictions/ directory.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left-model", default="G1-late-action")
    parser.add_argument("--right-model", default="B1-v2")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    separate_results = args.left_result is not None or args.right_result is not None
    if args.result is not None and separate_results:
        parser.error("use either --result or --left-result/--right-result")
    if args.result is None and (
        args.left_result is None or args.right_result is None
    ):
        parser.error("provide --result or both --left-result and --right-result")
    output = analyze(
        args.result,
        left_result_path=args.left_result,
        right_result_path=args.right_result,
        left_prediction_root=args.left_prediction_root,
        right_prediction_root=args.right_prediction_root,
        left_model=args.left_model,
        right_model=args.right_model,
        replicates=args.replicates,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "comparison": output["comparison"],
                "output": str(args.output),
                "hierarchical_bootstrap": output["hierarchical_bootstrap"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
