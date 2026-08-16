"""Parameter-budget accounting and width selection for the core screen."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


def trainable_parameter_count(module: Any) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def total_parameter_count(module: Any) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def select_width(
    factory: Callable[[int], Any],
    widths: Sequence[int],
    target_parameters: int,
    *,
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Pick the closest candidate and fail when it violates the declared budget."""

    if not widths:
        raise ValueError("width candidate list is empty")
    candidates = []
    for width in widths:
        module = factory(int(width))
        count = trainable_parameter_count(module)
        candidates.append({"width": int(width), "trainable_parameters": count})
    best = min(candidates, key=lambda item: abs(item["trainable_parameters"] - int(target_parameters)))
    relative_error = abs(best["trainable_parameters"] - int(target_parameters)) / max(1, int(target_parameters))
    return {
        "target_parameters": int(target_parameters),
        "tolerance": float(tolerance),
        "selected": best,
        "relative_error": float(relative_error),
        "within_tolerance": bool(relative_error <= float(tolerance)),
        "candidates": candidates,
    }
