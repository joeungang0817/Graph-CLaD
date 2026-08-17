"""Dependency-light metrics for masked relation-change predictions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


RELATION_FAMILIES = {
    "horizontal": ("left", "right"),
    "depth": ("front", "behind"),
    "vertical": ("above", "below"),
    "contact": ("contact",),
    "support": ("on",),
}


def aggregate_relation_families(per_relation: dict[str, Any]) -> dict[str, Any]:
    """Average inverse relations once per semantic family.

    The ordinary relation macro is retained for compatibility.  This report
    prevents left/right, front/behind, and above/below from being presented as
    six independent families of evidence.
    """

    per_family: dict[str, Any] = {}
    metric_names = ("pr_auc", "f1", "brier", "ece_10bin")
    for family, members in RELATION_FAMILIES.items():
        present = [name for name in members if name in per_relation]
        if not present:
            continue
        entry: dict[str, Any] = {
            "relations": present,
            "evaluable_relations": [],
            "relation_valid_count_sum": int(
                sum(int(per_relation[name].get("valid_count", 0)) for name in present)
            ),
        }
        for metric in metric_names:
            values = [
                float(per_relation[name][metric])
                for name in present
                if per_relation[name].get(metric) is not None
            ]
            entry[metric] = float(np.mean(values)) if values else None
        entry["evaluable_relations"] = [
            name for name in present if per_relation[name].get("pr_auc") is not None
        ]
        per_family[family] = entry

    def family_macro(metric: str) -> float | None:
        values = [
            float(entry[metric])
            for entry in per_family.values()
            if entry.get(metric) is not None
        ]
        return float(np.mean(values)) if values else None

    return {
        "per_family": per_family,
        "family_macro_pr_auc": family_macro("pr_auc"),
        "family_macro_f1": family_macro("f1"),
        "family_macro_brier": family_macro("brier"),
        "family_macro_ece_10bin": family_macro("ece_10bin"),
        "family_policy": "inverse-pair-mean-then-family-macro",
    }


def _sigmoid(value: Any) -> np.ndarray:
    logits = np.asarray(value, dtype=np.float64)
    if not np.isfinite(logits).all():
        raise ValueError("relation logits contain NaN or Inf")
    # Clipping avoids overflow without changing probabilities at float64
    # precision in the saturated region.
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -700.0, 700.0)))


def _average_precision(y_true: np.ndarray, score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return None
    order = np.argsort(-score, kind="mergesort")
    labels = y_true[order]
    sorted_score = score[order]
    positives = labels == 1
    total_positive = int(positives.sum())
    if total_positive == 0:
        return None
    # Evaluate precision only after each complete equal-score group.  Ranking
    # tied examples by their input order would make AP depend on JSONL order.
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_score)), len(sorted_score) - 1]
    cumulative_positive = np.cumsum(positives)
    true_positive = cumulative_positive[group_ends]
    precision = true_positive / (group_ends + 1)
    recall = true_positive / total_positive
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _best_f1(y_true: np.ndarray, score: np.ndarray, thresholds: Sequence[float]) -> tuple[float | None, float | None]:
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return None, None
    ordered_thresholds = sorted(float(value) for value in thresholds)
    best = (0.0, ordered_thresholds[0])
    for threshold in ordered_thresholds:
        prediction = score >= float(threshold)
        tp = int(np.logical_and(prediction, y_true == 1).sum())
        fp = int(np.logical_and(prediction, y_true == 0).sum())
        fn = int(np.logical_and(~prediction, y_true == 1).sum())
        denominator = 2 * tp + fp + fn
        value = float(2 * tp / denominator) if denominator else 0.0
        # A conservative, deterministic tie break: choose the highest
        # threshold among equally good validation F1 values.
        if value > best[0] or (value == best[0] and float(threshold) > best[1]):
            best = (value, float(threshold))
    return best


def _f1_at_threshold(y_true: np.ndarray, score: np.ndarray, threshold: float) -> float | None:
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return None
    prediction = score >= float(threshold)
    tp = int(np.logical_and(prediction, y_true == 1).sum())
    fp = int(np.logical_and(prediction, y_true == 0).sum())
    fn = int(np.logical_and(~prediction, y_true == 1).sum())
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def _calibration(y_true: np.ndarray, score: np.ndarray, bins: int = 10) -> tuple[float | None, float | None]:
    if y_true.size == 0:
        return None, None
    brier = float(np.mean((score - y_true) ** 2))
    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, int(bins) + 1)
    for index in range(int(bins)):
        lower, upper = boundaries[index], boundaries[index + 1]
        selected = (score >= lower) & (score < upper if index + 1 < bins else score <= upper)
        count = int(selected.sum())
        if count:
            accuracy = float(np.mean(y_true[selected]))
            confidence = float(np.mean(score[selected]))
            ece += count / y_true.size * abs(accuracy - confidence)
    return brier, float(ece)


def evaluate_relation_predictions(
    logits: Any,
    targets: Any,
    masks: Any,
    *,
    thresholds: Sequence[float] | None = None,
    fixed_thresholds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Return per-relation and macro PR-AUC/F1 without treating unknown as 0."""

    score = _sigmoid(logits)
    target = np.asarray(targets, dtype=np.int64)
    mask = np.asarray(masks, dtype=bool)
    if score.ndim != 2 or target.shape != score.shape or mask.shape != score.shape:
        raise ValueError("relation prediction arrays must have shape [N,R]")
    threshold_grid = list(thresholds or np.linspace(0.05, 0.95, 19))
    if not threshold_grid:
        raise ValueError("threshold grid must not be empty")
    if fixed_thresholds is not None and len(fixed_thresholds) != score.shape[1]:
        raise ValueError("fixed_thresholds must contain one value per relation")
    names = ("left", "right", "front", "behind", "above", "below", "contact", "on")
    per_relation: dict[str, Any] = {}
    pr_values: list[float] = []
    f1_values: list[float] = []
    brier_values: list[float] = []
    ece_values: list[float] = []
    for index, name in enumerate(names[: score.shape[1]]):
        valid = mask[:, index]
        y = target[valid, index]
        s = score[valid, index]
        ap = _average_precision(y, s)
        if fixed_thresholds is None:
            f1, threshold = _best_f1(y, s, threshold_grid)
        else:
            threshold = float(fixed_thresholds[index])
            f1 = _f1_at_threshold(y, s, threshold)
        brier, ece = _calibration(y, s)
        entry = {"valid_count": int(valid.sum()), "positive_count": int((y == 1).sum()), "pr_auc": ap, "f1": f1, "threshold": threshold, "brier": brier, "ece_10bin": ece}
        per_relation[name] = entry
        if ap is not None:
            pr_values.append(ap)
        if f1 is not None:
            f1_values.append(f1)
        if brier is not None:
            brier_values.append(brier)
        if ece is not None:
            ece_values.append(ece)
    result = {
        "per_relation": per_relation,
        "macro_pr_auc": float(np.mean(pr_values)) if pr_values else None,
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "macro_brier": float(np.mean(brier_values)) if brier_values else None,
        "macro_ece_10bin": float(np.mean(ece_values)) if ece_values else None,
        "valid_rows": int(mask.sum()),
        "threshold_source": "validation_fixed" if fixed_thresholds is not None else "optimized_on_rows",
    }
    result.update(aggregate_relation_families(per_relation))
    return result


def evaluate_motion(
    prediction: Any,
    target: Any,
    *,
    moving_threshold: float = 0.01,
) -> dict[str, float | int | None]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if prediction.shape != target.shape:
        raise ValueError("motion prediction and target lengths differ")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("motion prediction and target must be finite")
    if float(moving_threshold) < 0.0:
        raise ValueError("moving_threshold must be non-negative")
    if prediction.size == 0:
        return {
            "mae": None,
            "rmse": None,
            "moving_count": 0,
            "moving_mae": None,
            "moving_rmse": None,
            "moving_threshold": float(moving_threshold),
        }
    error = prediction - target
    moving = np.abs(target) > float(moving_threshold)
    moving_error = error[moving]
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "moving_count": int(moving.sum()),
        "moving_mae": float(np.mean(np.abs(moving_error))) if moving_error.size else None,
        "moving_rmse": float(np.sqrt(np.mean(moving_error**2))) if moving_error.size else None,
        "moving_threshold": float(moving_threshold),
    }


def no_change_fpr(logits: Any, targets: Any, masks: Any, thresholds: Sequence[float]) -> dict[str, float | None]:
    score = _sigmoid(logits)
    target = np.asarray(targets, dtype=np.int64)
    mask = np.asarray(masks, dtype=bool)
    if score.ndim != 2 or target.shape != score.shape or mask.shape != score.shape:
        raise ValueError("relation prediction arrays must have shape [N,R]")
    if len(thresholds) != score.shape[1]:
        raise ValueError("thresholds must contain one value per relation")
    values: list[float] = []
    for index, threshold in enumerate(thresholds):
        valid_no_change = np.logical_and(mask[:, index], target[:, index] == 0)
        predicted_change = score[:, index] >= float(threshold)
        denominator = int(valid_no_change.sum())
        if denominator:
            values.append(float(np.logical_and(valid_no_change, predicted_change).sum() / denominator))
    return {"mean": float(np.mean(values)) if values else None}
