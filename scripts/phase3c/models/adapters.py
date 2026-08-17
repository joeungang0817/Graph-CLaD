"""Shared Phase 3C fusion and relation-change heads."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .structured import structured_input_view


class SemanticPastActEncoder(nn.Module):
    """Action-only adapter used by the no-graph C3-Sem-PastAct control."""

    def __init__(self, output_dim: int = 256, hidden_dim: int = 128, action_dim: int = 42):
        super().__init__()
        self.output_dim = int(output_dim)
        self.action_dim = int(action_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.action_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.output_dim),
            nn.GELU(),
        )

    def forward(self, batch: Any) -> torch.Tensor:
        action = batch["past_action"] if isinstance(batch, dict) else batch.past_action
        action = action.float()
        if action.ndim == 3:
            action = action.flatten(start_dim=1)
        if action.ndim != 2 or action.shape[1] != self.action_dim:
            raise ValueError(f"past_action must be [B,{self.action_dim}] for semantic adapter")
        if not torch.isfinite(action).all():
            raise ValueError("past_action contains non-finite values")
        return self.encoder(action)


# Backward-compatible import name for pre-run configs; behavior is now the
# planned action-only semantic adapter rather than a constant zero vector.
ZeroStructuredEncoder = SemanticPastActEncoder


class CommonRelationHead(nn.Module):
    """Matched semantic/structured fusion head used by every candidate."""

    def __init__(self, *, semantic_dim: int = 2048, structured_dim: int = 256, latent_dim: int = 256, relation_dim: int = 8):
        super().__init__()
        self.semantic_projector = nn.Sequential(nn.Linear(semantic_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())
        self.structured_projector = nn.Sequential(nn.Linear(structured_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())
        self.adapter_gate = nn.Parameter(torch.zeros(1))
        self.fusion_norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(0.1)
        self.relation_head = nn.Linear(latent_dim, relation_dim)
        self.motion_head = nn.Linear(latent_dim, 1)

    def forward(
        self,
        semantic: torch.Tensor,
        structured: torch.Tensor,
        *,
        adapter_enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        semantic = semantic.float()
        structured = structured.float()
        if semantic.ndim != 2 or semantic.shape[1] != self.semantic_projector[0].in_features:
            raise ValueError("semantic foresight tensor has the wrong shape")
        if structured.ndim != 2 or structured.shape[1] != self.structured_projector[0].in_features:
            raise ValueError("structured representation tensor has the wrong shape")
        if semantic.shape[0] != structured.shape[0]:
            raise ValueError("semantic and structured batch dimensions differ")
        if not torch.isfinite(semantic).all() or not torch.isfinite(structured).all():
            raise ValueError("fusion inputs contain non-finite values")
        base = self.semantic_projector(semantic)
        adapter = self.structured_projector(structured)
        adapter_scale = torch.sigmoid(self.adapter_gate) if adapter_enabled else 0.0
        latent = self.dropout(self.fusion_norm(base + adapter_scale * adapter))
        return {"latent": latent, "relation_logits": self.relation_head(latent), "scene_motion": self.motion_head(latent)}


class Phase3CAdapter(nn.Module):
    """Glue a structured encoder to the common head without future inputs."""

    def __init__(self, structured_encoder: nn.Module, *, semantic_dim: int = 2048, structured_dim: int = 256, relation_dim: int = 8):
        super().__init__()
        self.structured_encoder = structured_encoder
        self.head = CommonRelationHead(
            semantic_dim=semantic_dim, structured_dim=structured_dim, relation_dim=relation_dim
        )

    def forward(
        self,
        batch: Any,
        semantic_foresight: torch.Tensor,
        *,
        adapter_enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        structured = self.structured_encoder(structured_input_view(batch))
        return self.head(
            semantic_foresight,
            structured,
            adapter_enabled=adapter_enabled,
        )
