"""Masked losses for Phase 3C relation and motion targets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Compute BCE only where the relation is valid; unknown is not negative."""

    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError("logits, targets, and mask must have identical shapes")
    logits = logits.float()
    targets = targets.float()
    mask = mask.bool()
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError("masked BCE received non-finite values")
    count = int(mask.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    values = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    return values.masked_select(mask).mean(), count


def smooth_l1_motion(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float()
    target = target.float()
    if prediction.shape != target.shape:
        raise ValueError("motion prediction and target shapes differ")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("motion loss received non-finite values")
    return F.smooth_l1_loss(prediction, target)


def relation_motion_loss(
    predictions: Mapping[str, torch.Tensor],
    batch: Any,
    *,
    relation_weight: float = 1.0,
    motion_weight: float = 0.1,
    pos_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | int]:
    relation_logits = predictions.get("relation_logits")
    scene_motion = predictions.get("scene_motion")
    if relation_logits is None or scene_motion is None:
        raise KeyError("predictions must contain relation_logits and scene_motion")
    target_relation = getattr(batch, "target_relation_change", batch["target_relation_change"] if isinstance(batch, Mapping) else None)
    relation_mask = getattr(batch, "target_relation_mask", batch["target_relation_mask"] if isinstance(batch, Mapping) else None)
    target_motion = getattr(batch, "target_scene_motion", batch["target_scene_motion"] if isinstance(batch, Mapping) else None)
    if target_relation is None or relation_mask is None or target_motion is None:
        raise KeyError("batch is missing Phase 3C relation/motion targets")
    relation_loss, valid_count = masked_bce_with_logits(relation_logits, target_relation, relation_mask, pos_weight=pos_weight)
    motion_loss = smooth_l1_motion(scene_motion, target_motion)
    total = float(relation_weight) * relation_loss + float(motion_weight) * motion_loss
    return {
        "loss": total,
        "loss_relation": relation_loss,
        "loss_motion": motion_loss,
        "relation_valid_count": valid_count,
    }
