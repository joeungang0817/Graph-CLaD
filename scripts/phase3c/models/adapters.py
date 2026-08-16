"""Shared Phase 3C fusion and relation-change heads."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ZeroStructuredEncoder(nn.Module):
    """No-graph control used by C3-Sem-PastAct with the common head."""

    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, batch: Any) -> torch.Tensor:
        action = batch["past_action"] if isinstance(batch, dict) else batch.past_action
        return action.new_zeros((action.shape[0], self.output_dim), dtype=torch.float32)


class CommonRelationHead(nn.Module):
    """Matched semantic/structured fusion head used by every candidate."""

    def __init__(self, *, semantic_dim: int = 2048, structured_dim: int = 256, latent_dim: int = 256, relation_dim: int = 8):
        super().__init__()
        self.semantic_projector = nn.Sequential(nn.Linear(semantic_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())
        self.structured_projector = nn.Sequential(nn.Linear(structured_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(), nn.Dropout(0.1)
        )
        self.relation_head = nn.Linear(latent_dim, relation_dim)
        self.motion_head = nn.Linear(latent_dim, 1)

    def forward(self, semantic: torch.Tensor, structured: torch.Tensor) -> dict[str, torch.Tensor]:
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
        latent = self.fusion(torch.cat([self.semantic_projector(semantic), self.structured_projector(structured)], dim=-1))
        return {"latent": latent, "relation_logits": self.relation_head(latent), "scene_motion": self.motion_head(latent)}


class Phase3CAdapter(nn.Module):
    """Glue a structured encoder to the common head without future inputs."""

    def __init__(self, structured_encoder: nn.Module, *, semantic_dim: int = 2048, structured_dim: int = 256, relation_dim: int = 8):
        super().__init__()
        self.structured_encoder = structured_encoder
        self.head = CommonRelationHead(
            semantic_dim=semantic_dim, structured_dim=structured_dim, relation_dim=relation_dim
        )

    def forward(self, batch: Any, semantic_foresight: torch.Tensor) -> dict[str, torch.Tensor]:
        structured = self.structured_encoder(batch)
        return self.head(semantic_foresight, structured)
