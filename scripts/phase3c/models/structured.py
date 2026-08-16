"""Action-conditioned structured encoders for the Phase 3C screen.

All classes consume the same two-snapshot graph contract.  The only intended
differences are whether edge information is omitted, pooled without messages,
or propagated through two residual message-passing layers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


GEOMETRY_EDGE_DIM = 5
CONTACT_EDGE_DIM = 2
RELATION_EDGE_DIM = 14
RELATION_NAMES = ("left", "right", "front", "behind", "above", "below", "on")


@dataclass(frozen=True)
class GraphBatch:
    node_features: torch.Tensor
    node_mask: torch.Tensor
    edge_geometry: torch.Tensor
    edge_contact: torch.Tensor
    edge_relations: torch.Tensor
    edge_mask: torch.Tensor


@dataclass(frozen=True)
class StructuredBatch:
    graph_prev: GraphBatch
    graph_current: GraphBatch
    past_action: torch.Tensor


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise KeyError(f"structured batch is missing {name}")
        return value[name]
    if not hasattr(value, name):
        raise AttributeError(f"structured batch is missing {name}")
    return getattr(value, name)


def _graph(value: Any) -> GraphBatch:
    if isinstance(value, GraphBatch):
        return value
    return GraphBatch(
        node_features=_field(value, "node_features"),
        node_mask=_field(value, "node_mask"),
        edge_geometry=_field(value, "edge_geometry"),
        edge_contact=_field(value, "edge_contact"),
        edge_relations=_field(value, "edge_relations"),
        edge_mask=_field(value, "edge_mask"),
    )


def _finite(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    value = value.float() if not torch.is_floating_point(value) else value
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _validate_graph(graph: GraphBatch, *, node_dim: int, geometry_dim: int, contact_dim: int, relation_dim: int) -> None:
    node = _finite(graph.node_features, "node_features")
    if node.ndim != 3 or node.shape[-1] != node_dim:
        raise ValueError(f"node_features must be [B,N,{node_dim}], got {tuple(node.shape)}")
    batch, nodes = node.shape[:2]
    mask = graph.node_mask.bool()
    if mask.shape != (batch, nodes):
        raise ValueError("node_mask shape does not match node_features")
    geometry = _finite(graph.edge_geometry, "edge_geometry")
    contact = _finite(graph.edge_contact, "edge_contact")
    relations = _finite(graph.edge_relations, "edge_relations")
    edge_mask = graph.edge_mask.bool()
    expected = (batch, nodes, nodes)
    if geometry.shape[:3] != expected or geometry.shape[-1] != geometry_dim:
        raise ValueError("edge_geometry shape does not match graph contract")
    if contact.shape[:3] != expected or contact.shape[-1] != contact_dim:
        raise ValueError("edge_contact shape does not match graph contract")
    if relations.shape[:3] != expected or relations.shape[-1] != relation_dim:
        raise ValueError("edge_relations shape does not match graph contract")
    if edge_mask.shape != expected:
        raise ValueError("edge_mask shape does not match graph contract")
    type_values = node[..., :4]
    valid_type_sums = type_values.sum(dim=-1)[mask]
    if valid_type_sums.numel() and not torch.allclose(
        valid_type_sums, torch.ones_like(valid_type_sums), atol=1e-6, rtol=0.0
    ):
        raise ValueError("valid nodes must have exactly one active node-type channel")
    if ((type_values < -1e-6) | (type_values > 1.0 + 1e-6)).any():
        raise ValueError("node-type channels must be binary")
    valid_endpoints = mask[:, :, None] & mask[:, None, :]
    if torch.logical_and(edge_mask, ~valid_endpoints).any():
        raise ValueError("edge_mask references a padded or invalid node")
    diagonal = torch.eye(nodes, dtype=torch.bool, device=edge_mask.device).unsqueeze(0)
    if torch.logical_and(edge_mask, diagonal).any():
        raise ValueError("Phase 3C graph contract does not permit self edges")


def _action(action: torch.Tensor, *, action_dim: int = 42) -> torch.Tensor:
    action = _finite(action, "past_action")
    if action.ndim == 3:
        if tuple(action.shape[1:]) != (6, 7):
            raise ValueError("past_action must be [B,6,7]")
        action = action.flatten(start_dim=1)
    if action.ndim != 2 or action.shape[1] != action_dim:
        raise ValueError(f"past_action must be [B,{action_dim}]")
    return action


class MaskedAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(values).squeeze(-1)
        scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * mask.to(weights.dtype)
        normalizer = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return (values * (weights / normalizer).unsqueeze(-1)).sum(dim=1)


class ActionFiLM(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int):
        super().__init__()
        self.action = nn.Sequential(nn.Linear(action_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim * 2))

    def forward(self, values: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.action(action).chunk(2, dim=-1)
        return values * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class _StructuredBase(nn.Module):
    def __init__(
        self,
        *,
        node_feature_dim: int = 8,
        geometry_dim: int = GEOMETRY_EDGE_DIM,
        contact_dim: int = CONTACT_EDGE_DIM,
        relation_dim: int = RELATION_EDGE_DIM,
        action_dim: int = 42,
        hidden_dim: int = 128,
        output_dim: int = 256,
    ):
        super().__init__()
        self.node_feature_dim = int(node_feature_dim)
        self.geometry_dim = int(geometry_dim)
        self.contact_dim = int(contact_dim)
        self.relation_dim = int(relation_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_feature_dim * 3, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.action_film = ActionFiLM(action_dim, hidden_dim)
        self.readout = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim), nn.GELU())
        self.node_pool = MaskedAttentionPool(hidden_dim)

    def _prepare(self, batch: Any) -> tuple[GraphBatch, GraphBatch, torch.Tensor, torch.Tensor, torch.Tensor]:
        previous = _graph(_field(batch, "graph_prev"))
        current = _graph(_field(batch, "graph_current"))
        _validate_graph(previous, node_dim=self.node_feature_dim, geometry_dim=self.geometry_dim, contact_dim=self.contact_dim, relation_dim=self.relation_dim)
        _validate_graph(current, node_dim=self.node_feature_dim, geometry_dim=self.geometry_dim, contact_dim=self.contact_dim, relation_dim=self.relation_dim)
        if previous.node_features.shape != current.node_features.shape:
            raise ValueError("graph_prev/current node tensors must have identical padded shapes")
        action = _action(_field(batch, "past_action"), action_dim=self.action_dim)
        if action.shape[0] != previous.node_features.shape[0]:
            raise ValueError("graph and action batch dimensions differ")
        node_mask = previous.node_mask.bool() & current.node_mask.bool()
        temporal = torch.cat(
            [previous.node_features, current.node_features, current.node_features - previous.node_features], dim=-1
        )
        node_hidden = self.node_encoder(temporal)
        node_hidden = self.action_film(node_hidden, action)
        return previous, current, action, node_hidden, node_mask

    def _edge_temporal(self, previous: GraphBatch, current: GraphBatch, *, include_relations: bool) -> tuple[torch.Tensor, torch.Tensor]:
        parts = []
        for left, right in (
            (previous.edge_geometry, current.edge_geometry),
            (previous.edge_contact, current.edge_contact),
        ):
            parts.extend((left, right, right - left))
        if include_relations:
            parts.extend((previous.edge_relations, current.edge_relations, current.edge_relations - previous.edge_relations))
        return torch.cat(parts, dim=-1), previous.edge_mask.bool() & current.edge_mask.bool()


class SceneSetPastAct(_StructuredBase):
    """Full-scene temporal set control with no pairwise edge messages."""

    def forward(self, batch: Any) -> torch.Tensor:
        _, _, _, node_hidden, node_mask = self._prepare(batch)
        return self.readout(self.node_pool(node_hidden, node_mask))


class PairPastAct(_StructuredBase):
    """Target-centric robot-to-entity pair encoder without object-object edges."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        pair_dim = self.hidden_dim * 3
        self.pair_encoder = nn.Sequential(nn.Linear(pair_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.GELU())
        self.pair_pool = MaskedAttentionPool(self.hidden_dim)

    def forward(self, batch: Any) -> torch.Tensor:
        previous, current, _, node_hidden, node_mask = self._prepare(batch)
        robot_flags = (
            previous.node_features[..., 0] > 0.5
        ) & (
            current.node_features[..., 0] > 0.5
        ) & node_mask
        if not torch.all(robot_flags.sum(dim=1) == 1) or not torch.all(robot_flags[:, 0]):
            raise ValueError("PairPastAct requires exactly one robot at node index 0")
        robot = node_hidden[:, :1]
        candidates = node_hidden[:, 1:]
        pair = torch.cat([robot.expand_as(candidates), candidates, candidates - robot], dim=-1)
        pair_mask = node_mask[:, 1:]
        if pair.shape[1] == 0:
            pooled = node_hidden.new_zeros((node_hidden.shape[0], self.hidden_dim))
        else:
            pooled = self.pair_pool(self.pair_encoder(pair), pair_mask)
        return self.readout(pooled)


class _EdgeModel(_StructuredBase):
    def __init__(self, *, include_relations: bool, message_passing: bool, layers: int = 2, **kwargs: Any):
        super().__init__(**kwargs)
        self.include_relations = bool(include_relations)
        self.message_passing = bool(message_passing)
        edge_dim = (self.geometry_dim + self.contact_dim) * 3
        if self.include_relations:
            edge_dim += self.relation_dim * 3
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + edge_dim + self.action_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU(),
        )
        self.message_layers = nn.ModuleList()
        self.node_update_layers = nn.ModuleList()
        for _ in range(int(layers)):
            self.message_layers.append(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU()))
            self.node_update_layers.append(
                nn.Sequential(nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.GELU())
            )
        self.edge_pool = MaskedAttentionPool(self.hidden_dim)

    def _edge_tokens(self, previous: GraphBatch, current: GraphBatch, action: torch.Tensor, node_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        edge_features, edge_mask = self._edge_temporal(previous, current, include_relations=self.include_relations)
        left = node_hidden.unsqueeze(2).expand(-1, -1, node_hidden.shape[1], -1)
        right = node_hidden.unsqueeze(1).expand(-1, node_hidden.shape[1], -1, -1)
        action_grid = action[:, None, None, :].expand(-1, node_hidden.shape[1], node_hidden.shape[1], -1)
        tokens = self.edge_encoder(torch.cat([left, right, edge_features, action_grid], dim=-1))
        return tokens, edge_mask

    def forward(self, batch: Any) -> torch.Tensor:
        previous, current, action, node_hidden, node_mask = self._prepare(batch)
        tokens, edge_mask = self._edge_tokens(previous, current, action, node_hidden)
        if self.message_passing:
            hidden = node_hidden
            for message_layer, update_layer in zip(self.message_layers, self.node_update_layers):
                messages = message_layer(tokens)
                weights = edge_mask.to(messages.dtype).unsqueeze(-1)
                # Edge axes are [source, target].  Update each target from
                # incoming source messages rather than aggregating outgoing
                # edges back into the source node.
                aggregated = (messages * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
                hidden = hidden + update_layer(torch.cat([hidden, aggregated], dim=-1))
                hidden = hidden.masked_fill(~node_mask.unsqueeze(-1), 0.0)
                # Recompute edge tokens so each layer has the same raw edge
                # input but updated endpoint states.
                left = hidden.unsqueeze(2).expand(-1, -1, hidden.shape[1], -1)
                right = hidden.unsqueeze(1).expand(-1, hidden.shape[1], -1, -1)
                edge_features, edge_mask = self._edge_temporal(previous, current, include_relations=self.include_relations)
                action_grid = action[:, None, None, :].expand(-1, hidden.shape[1], hidden.shape[1], -1)
                tokens = self.edge_encoder(torch.cat([left, right, edge_features, action_grid], dim=-1))
            pooled = self.node_pool(hidden, node_mask)
            # The node readout includes edge information through messages.
            return self.readout(pooled)
        edge_values = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
        flat_mask = edge_mask.reshape(edge_mask.shape[0], -1)
        return self.readout(self.edge_pool(edge_values, flat_mask))


class GeomMPNNPastAct(_EdgeModel):
    def __init__(self, **kwargs: Any):
        super().__init__(include_relations=False, message_passing=True, **kwargs)


class RelPoolPastAct(_EdgeModel):
    def __init__(self, **kwargs: Any):
        super().__init__(include_relations=True, message_passing=False, **kwargs)


class RelMPNNPastAct(_EdgeModel):
    def __init__(self, **kwargs: Any):
        super().__init__(include_relations=True, message_passing=True, **kwargs)


MODEL_REGISTRY = {
    "C3-SceneSet-PastAct": SceneSetPastAct,
    "C3-Pair-PastAct": PairPastAct,
    "C3-GeomMPNN-PastAct": GeomMPNNPastAct,
    "C3-RelPool-PastAct": RelPoolPastAct,
    "C3-RelMPNN-PastAct": RelMPNNPastAct,
}


def build_structured_model(model_id: str, **kwargs: Any) -> nn.Module:
    try:
        factory = MODEL_REGISTRY[str(model_id)]
    except KeyError as exc:
        raise ValueError(f"unknown Phase 3C structured model: {model_id}") from exc
    return factory(**kwargs)
