"""Dependency-light metrics for masked relation-change predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


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


def evaluate_relation_predictions(
    logits: Any,
    targets: Any,
    masks: Any,
    *,
    thresholds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Return per-relation and macro PR-AUC/F1 without treating unknown as 0."""

    score = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    target = np.asarray(targets, dtype=np.int64)
    mask = np.asarray(masks, dtype=bool)
    if score.ndim != 2 or target.shape != score.shape or mask.shape != score.shape:
        raise ValueError("relation prediction arrays must have shape [N,R]")
    threshold_grid = list(thresholds or np.linspace(0.05, 0.95, 19))
    names = ("left", "right", "front", "behind", "above", "below", "contact", "on")
    per_relation: dict[str, Any] = {}
    pr_values: list[float] = []
    f1_values: list[float] = []
    for index, name in enumerate(names[: score.shape[1]]):
        valid = mask[:, index]
        y = target[valid, index]
        s = score[valid, index]
        ap = _average_precision(y, s)
        f1, threshold = _best_f1(y, s, threshold_grid)
        entry = {"valid_count": int(valid.sum()), "positive_count": int((y == 1).sum()), "pr_auc": ap, "f1": f1, "threshold": threshold}
        per_relation[name] = entry
        if ap is not None:
            pr_values.append(ap)
        if f1 is not None:
            f1_values.append(f1)
    return {
        "per_relation": per_relation,
        "macro_pr_auc": float(np.mean(pr_values)) if pr_values else None,
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "valid_rows": int(mask.sum()),
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
    score = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    target = np.asarray(targets, dtype=np.int64)
    mask = np.asarray(masks, dtype=bool)
    values: list[float] = []
    for threshold in thresholds:
        valid_no_change = np.logical_and(mask, target == 0)
        predicted_change = score >= float(threshold)
        denominator = int(valid_no_change.sum())
        if denominator:
            values.append(float(np.logical_and(valid_no_change, predicted_change).sum() / denominator))
    return {"mean": float(np.mean(values)) if values else None}
