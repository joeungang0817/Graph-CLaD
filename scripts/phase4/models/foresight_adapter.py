"""Information-matched Phase 4 residual foresight interface.

Phase 3C selects an architecture, not a checkpoint, for Phase 4.  This module
therefore accepts any causal structured encoder with output shape ``[B,D]``
and keeps the Stage-2-facing tensor contract identical to the semantic-only
baseline.  It does not choose a Phase 3C winner or implement Stage 2.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from scripts.phase3c.models.structured import structured_input_view


def _semantic_branches(
    semantic_foresight: torch.Tensor, hidden_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(semantic_foresight):
        raise TypeError("semantic_foresight must be a torch.Tensor")
    semantic_foresight = semantic_foresight.float()
    if semantic_foresight.ndim != 2 or semantic_foresight.shape[1] != 2 * int(
        hidden_dim
    ):
        raise ValueError(
            f"semantic_foresight must be [B,{2 * int(hidden_dim)}]"
        )
    if not torch.isfinite(semantic_foresight).all():
        raise ValueError("semantic_foresight contains NaN or Inf")
    return semantic_foresight.chunk(2, dim=-1)


def _pack(pred_p: torch.Tensor, pred_s: torch.Tensor) -> dict[str, torch.Tensor]:
    if pred_p.shape != pred_s.shape or pred_p.ndim != 2:
        raise ValueError("Phase 4 foresight branches must be matching rank-2 tensors")
    return {
        "pred_p": pred_p,
        "pred_s": pred_s,
        "foresight": torch.cat([pred_p, pred_s], dim=-1),
    }


class SemanticForesightInterface(nn.Module):
    """Frozen semantic baseline at the exact Stage-2-facing interface."""

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = int(hidden_dim)

    def forward(self, semantic_foresight: torch.Tensor) -> dict[str, torch.Tensor]:
        pred_p, pred_s = _semantic_branches(semantic_foresight, self.hidden_dim)
        return _pack(
            functional.normalize(pred_p, dim=-1, eps=1e-8),
            functional.normalize(pred_s, dim=-1, eps=1e-8),
        )


class ResidualGraphForesightAdapter(nn.Module):
    """Add causal structured residuals to both frozen semantic branches.

    The learned branch gates start at exactly zero.  Consequently a fresh
    adapter is numerically equivalent to :class:`SemanticForesightInterface`.
    Passing ``adapter_enabled=False`` skips the structured encoder entirely,
    which is the policy-level adapter-off control and a physical leakage gate.
    """

    def __init__(
        self,
        structured_encoder: nn.Module,
        *,
        structured_dim: int = 256,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        if int(structured_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("structured_dim and hidden_dim must be positive")
        self.structured_encoder = structured_encoder
        self.structured_dim = int(structured_dim)
        self.hidden_dim = int(hidden_dim)
        self.delta_p = nn.Sequential(
            nn.Linear(self.structured_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.delta_s = nn.Sequential(
            nn.Linear(self.structured_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.alpha = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def forward(
        self,
        batch: Any,
        semantic_foresight: torch.Tensor,
        *,
        adapter_enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        pred_p_sem, pred_s_sem = _semantic_branches(
            semantic_foresight, self.hidden_dim
        )
        if not adapter_enabled:
            return _pack(
                functional.normalize(pred_p_sem, dim=-1, eps=1e-8),
                functional.normalize(pred_s_sem, dim=-1, eps=1e-8),
            )
        structured = self.structured_encoder(structured_input_view(batch)).float()
        if structured.ndim != 2 or structured.shape != (
            semantic_foresight.shape[0],
            self.structured_dim,
        ):
            raise ValueError(
                f"structured encoder output must be [B,{self.structured_dim}]"
            )
        if not torch.isfinite(structured).all():
            raise ValueError("structured encoder output contains NaN or Inf")
        pred_p = functional.normalize(
            pred_p_sem + self.alpha[0] * self.delta_p(structured),
            dim=-1,
            eps=1e-8,
        )
        pred_s = functional.normalize(
            pred_s_sem + self.alpha[1] * self.delta_s(structured),
            dim=-1,
            eps=1e-8,
        )
        return _pack(pred_p, pred_s)
