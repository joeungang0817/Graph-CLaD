"""Dependency-light metrics for masked relation-change predictions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


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
    positives = labels == 1
    total_positive = int(positives.sum())
    if total_positive == 0:
        return None
    cumulative = np.cumsum(positives)
    precision = cumulative / np.arange(1, len(labels) + 1)
    return float((precision * positives).sum() / total_positive)


def _best_f1(y_true: np.ndarray, score: np.ndarray, thresholds: Sequence[float]) -> tuple[float | None, float | None]:
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return None, None
    best = (0.0, float(thresholds[0]))
    for threshold in thresholds:
        prediction = score >= float(threshold)
        tp = int(np.logical_and(prediction, y_true == 1).sum())
        fp = int(np.logical_and(prediction, y_true == 0).sum())
        fn = int(np.logical_and(~prediction, y_true == 1).sum())
        denominator = 2 * tp + fp + fn
        value = float(2 * tp / denominator) if denominator else 0.0
        if value > best[0]:
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
    return {
        "per_relation": per_relation,
        "macro_pr_auc": float(np.mean(pr_values)) if pr_values else None,
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "macro_brier": float(np.mean(brier_values)) if brier_values else None,
        "macro_ece_10bin": float(np.mean(ece_values)) if ece_values else None,
        "valid_rows": int(mask.sum()),
        "threshold_source": "validation_fixed" if fixed_thresholds is not None else "optimized_on_rows",
    }


def evaluate_motion(prediction: Any, target: Any) -> dict[str, float | None]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if prediction.shape != target.shape:
        raise ValueError("motion prediction and target lengths differ")
    if prediction.size == 0:
        return {"mae": None, "rmse": None}
    error = prediction - target
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error**2)))}


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
