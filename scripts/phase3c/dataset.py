"""Tensorization for the Phase 3C joined manifest and semantic store."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contracts import PRIMARY_RELATIONS, parse_action_window
from .io import iter_json_objects
from .models.structured import GraphBatch


NODE_TYPE_RANK = {"robot": 0, "object": 1, "fixture": 2, "site": 3}
NODE_TYPES = tuple(NODE_TYPE_RANK)
NODE_FEATURE_DIM = 8  # type one-hot(4) + position(3) + position-valid(1)
PROPRIO_DIM = 16  # robot joint position(7) + velocity(7) + gripper qpos(2)
GEOMETRY_DIM = 5  # relative xyz, distance, distance-valid
CONTACT_DIM = 2  # value/valid for one snapshot; temporal deltas are built by the model
RELATION_EDGE_DIM = 14  # seven non-contact relations x (value/valid) for one snapshot
NODE_CONTINUOUS_INDICES = (4, 5, 6)


@dataclass(frozen=True)
class Phase3CBatch:
    sample_ids: tuple[str, ...]
    task_ids: torch.Tensor
    episode_ids: tuple[str, ...]
    v_history: torch.Tensor
    p_history: torch.Tensor
    past_action: torch.Tensor
    language: torch.Tensor
    graph_prev: GraphBatch
    graph_current: GraphBatch
    target_v: torch.Tensor
    target_p: torch.Tensor
    target_relation_change: torch.Tensor
    target_relation_mask: torch.Tensor
    target_scene_motion: torch.Tensor


@dataclass(frozen=True)
class NormalizationStats:
    """Train-only normalization for continuous graph/proprio channels."""

    node_mean: tuple[float, ...]
    node_std: tuple[float, ...]
    proprio_mean: tuple[float, ...]
    proprio_std: tuple[float, ...]
    edge_geometry_mean: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    edge_geometry_std: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    continuous_node_indices: tuple[int, ...] = NODE_CONTINUOUS_INDICES

    def __post_init__(self) -> None:
        expected = (
            ("node_mean", self.node_mean, NODE_FEATURE_DIM),
            ("node_std", self.node_std, NODE_FEATURE_DIM),
            ("proprio_mean", self.proprio_mean, PROPRIO_DIM),
            ("proprio_std", self.proprio_std, PROPRIO_DIM),
            ("edge_geometry_mean", self.edge_geometry_mean, GEOMETRY_DIM),
            ("edge_geometry_std", self.edge_geometry_std, GEOMETRY_DIM),
        )
        for name, values, dimension in expected:
            if len(values) != dimension or not all(math.isfinite(float(item)) for item in values):
                raise ValueError(f"{name} must contain {dimension} finite values")
        if any(float(item) <= 0.0 for item in self.node_std + self.proprio_std + self.edge_geometry_std):
            raise ValueError("normalization standard deviations must be positive")
        if tuple(self.continuous_node_indices) != NODE_CONTINUOUS_INDICES:
            raise ValueError("normalization continuous node indices do not match Phase 3C schema")

    def apply_graph(self, graph: GraphBatch) -> GraphBatch:
        values = graph.node_features.clone()
        mean = torch.tensor(self.node_mean, dtype=values.dtype, device=values.device)
        std = torch.tensor(self.node_std, dtype=values.dtype, device=values.device)
        indices = torch.tensor(self.continuous_node_indices, dtype=torch.long, device=values.device)
        normalized_node = (values[..., indices] - mean[indices]) / std[indices]
        position_valid = values[..., 7:8] > 0.5
        values[..., indices] = torch.where(
            position_valid, normalized_node, torch.zeros_like(normalized_node)
        )
        geometry = graph.edge_geometry.clone()
        geometry_mean = torch.tensor(self.edge_geometry_mean, dtype=geometry.dtype, device=geometry.device)
        geometry_std = torch.tensor(self.edge_geometry_std, dtype=geometry.dtype, device=geometry.device)
        normalized_geometry = (geometry[..., :4] - geometry_mean[:4]) / geometry_std[:4]
        geometry[..., :4] = torch.where(
            geometry[..., 4:5] > 0.5,
            normalized_geometry,
            torch.zeros_like(normalized_geometry),
        )
        return GraphBatch(values, graph.node_mask, geometry, graph.edge_contact, graph.edge_relations, graph.edge_mask)

    def apply_proprio(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.proprio_mean, dtype=value.dtype, device=value.device)
        std = torch.tensor(self.proprio_std, dtype=value.dtype, device=value.device)
        return (value - mean) / std

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_mean": list(self.node_mean), "node_std": list(self.node_std),
            "proprio_mean": list(self.proprio_mean), "proprio_std": list(self.proprio_std),
            "edge_geometry_mean": list(self.edge_geometry_mean), "edge_geometry_std": list(self.edge_geometry_std),
            "continuous_node_indices": list(self.continuous_node_indices),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NormalizationStats":
        return cls(
            tuple(float(item) for item in value["node_mean"]), tuple(float(item) for item in value["node_std"]),
            tuple(float(item) for item in value["proprio_mean"]), tuple(float(item) for item in value["proprio_std"]),
            tuple(float(item) for item in value.get("edge_geometry_mean", (0.0, 0.0, 0.0, 0.0, 0.0))),
            tuple(float(item) for item in value.get("edge_geometry_std", (1.0, 1.0, 1.0, 1.0, 1.0))),
            tuple(int(item) for item in value.get("continuous_node_indices", NODE_CONTINUOUS_INDICES)),
        )


class SemanticFeatureStore:
    """Read-only index over the per-demo `.npz` shards produced by Milestone 2."""

    def __init__(self, root: Path, *, max_open_shards: int = 32):
        self.root = Path(root)
        with (self.root / "manifest.json").open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema") != "phase3c-semantic-feature-store.v2":
            raise ValueError("unsupported semantic feature-store schema")
        self.manifest = manifest
        self.index = manifest.get("index", {})
        if not isinstance(self.index, Mapping):
            raise ValueError("semantic feature-store index must be an object")
        if int(max_open_shards) <= 0:
            raise ValueError("max_open_shards must be positive")
        self.max_open_shards = int(max_open_shards)
        self._shards: OrderedDict[str, Any] = OrderedDict()

    @property
    def feature_dim(self) -> int:
        return int(self.manifest["decisionnce"]["feature_dim"])

    def _entry(self, key: str) -> Mapping[str, Any]:
        value = self.index.get(key)
        if not isinstance(value, Mapping):
            raise KeyError(f"semantic feature is absent: {key}")
        return value

    def _shard(self, name: str) -> Any:
        if name in self._shards:
            value = self._shards.pop(name)
            self._shards[name] = value
            return value
        if name not in self._shards:
            shard = (self.root / name).resolve()
            if self.root.resolve() not in shard.parents:
                raise ValueError(f"feature-store shard escapes root: {name}")
            if not shard.exists():
                raise FileNotFoundError(shard)
            self._shards[name] = np.load(shard, allow_pickle=False)
            while len(self._shards) > self.max_open_shards:
                _, old = self._shards.popitem(last=False)
                old.close()
        return self._shards[name]

    def close(self) -> None:
        for shard in self._shards.values():
            shard.close()
        self._shards.clear()

    def __enter__(self) -> "SemanticFeatureStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - defensive interpreter cleanup
        shards = getattr(self, "_shards", None)
        if shards is not None:
            self.close()

    def image(self, task_id: int, demo_key: str, step: int, view: int) -> np.ndarray:
        if int(view) not in (0, 1):
            raise ValueError(f"semantic view must be 0 or 1, got {view}")
        key = f"{int(task_id)}/{demo_key}/{int(step)}/view{int(view)}"
        entry = self._entry(key)
        values = self._shard(str(entry["shard"]))[f"view{int(view)}"]
        row = int(entry["row"])
        if row < 0 or row >= int(values.shape[0]):
            raise IndexError(f"semantic feature row is out of range at {key}: {row}")
        array = np.asarray(values[row], dtype=np.float32)
        if array.ndim != 1 or array.shape[0] != self.feature_dim or not np.isfinite(array).all():
            raise ValueError(f"invalid semantic feature shape/value at {key}")
        return array

    def language(self, task_id: int, demo_key: str) -> np.ndarray:
        key = f"{int(task_id)}/{demo_key}/language"
        entry = self._entry(key)
        array = np.asarray(self._shard(str(entry["shard"]))["language"], dtype=np.float32)
        if array.ndim != 1 or array.shape[0] != self.feature_dim or not np.isfinite(array).all():
            raise ValueError(f"invalid language feature shape/value at {key}")
        return array


def _node_map(graph: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for node in graph.get("nodes", []):
        if not isinstance(node, Mapping) or "node_id" not in node:
            raise ValueError("graph contains an invalid node")
        node_id = str(node["node_id"])
        if node_id in result:
            raise ValueError(f"duplicate graph node_id: {node_id}")
        result[node_id] = node
    return result


def _edge_map(graph: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for edge in graph.get("edges", []):
        if not isinstance(edge, Mapping) or "source" not in edge or "target" not in edge:
            raise ValueError("graph contains an invalid edge")
        key = (str(edge["source"]), str(edge["target"]))
        if key in result:
            raise ValueError(f"duplicate graph edge: {key[0]}->{key[1]}")
        result[key] = edge
    return result


def _node_order(*graphs: Mapping[str, Any]) -> list[str]:
    nodes: dict[str, str] = {}
    for graph in graphs:
        for node in _node_map(graph).values():
            node_id = str(node["node_id"])
            node_type = str(node.get("node_type", "unknown"))
            if node_id in nodes and nodes[node_id] != node_type:
                raise ValueError(
                    f"node type changed across snapshots: {node_id} {nodes[node_id]}->{node_type}"
                )
            nodes[node_id] = node_type
    return sorted(nodes, key=lambda node_id: (NODE_TYPE_RANK.get(nodes[node_id], 99), node_id))


def _node_vector(node: Mapping[str, Any], source_feature_dim: int = 24) -> list[float]:
    """Build the Phase 3C node schema while auditing the Phase 2D source."""

    vector = node.get("feature_vector")
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)) or len(vector) != source_feature_dim:
        raise ValueError(f"node {node.get('node_id')} has no {source_feature_dim}-dimensional feature_vector")
    source_values = [float(item) for item in vector]
    if not all(math.isfinite(item) for item in source_values):
        raise ValueError(f"node {node.get('node_id')} feature_vector contains NaN or Inf")
    # This is the Phase 2D input-clean invariant, not a task label.
    if abs(source_values[4]) > 1e-8:
        raise ValueError(f"task-derived node feature slot is non-zero for {node.get('node_id')}")
    node_type = str(node.get("node_type"))
    if node_type not in NODE_TYPE_RANK:
        raise ValueError(f"unsupported node type for {node.get('node_id')}: {node_type}")
    features = node.get("features")
    if not isinstance(features, Mapping):
        raise ValueError(f"node {node.get('node_id')} has no grouped features")
    position_valid = features.get("position_valid") == 1
    raw_position = features.get("position")
    if not isinstance(raw_position, Sequence) or isinstance(raw_position, (str, bytes)) or len(raw_position) != 3:
        if position_valid:
            raise ValueError(f"node {node.get('node_id')} has invalid position")
        position = [0.0, 0.0, 0.0]
    else:
        position = [float(item) for item in raw_position]
    if not all(math.isfinite(item) for item in position):
        raise ValueError(f"node {node.get('node_id')} position contains NaN or Inf")
    if not position_valid:
        position = [0.0, 0.0, 0.0]
    one_hot = [float(node_type == candidate) for candidate in NODE_TYPES]
    return one_hot + position + [float(position_valid)]


def _proprio(graph: Mapping[str, Any]) -> list[float]:
    robots = [node for node in _node_map(graph).values() if str(node.get("node_type")) == "robot"]
    if len(robots) != 1:
        raise ValueError(f"graph must have exactly one robot node, got {len(robots)}")
    robot = robots[0]
    features = robot.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("robot node has no grouped features")
    def vector(name: str, dimension: int) -> list[float]:
        if features.get(f"{name}_valid") != 1:
            raise ValueError(f"robot feature {name} is not valid")
        value = features.get(name)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != dimension:
            raise ValueError(f"robot feature {name} must be length {dimension}")
        result = [float(item) for item in value]
        if not all(math.isfinite(item) for item in result):
            raise ValueError(f"robot feature {name} contains NaN or Inf")
        return result
    return vector("joint_pos", 7) + vector("joint_vel", 7) + vector("gripper_qpos", 2)


def fit_normalization(records: Iterable[Mapping[str, Any]]) -> NormalizationStats:
    """Fit statistics on train records only; validation/test are never read."""

    class Moments:
        def __init__(self, dimension: int):
            self.count = 0
            self.mean = np.zeros(dimension, dtype=np.float64)
            self.m2 = np.zeros(dimension, dtype=np.float64)

        def add(self, value: Sequence[float]) -> None:
            row = np.asarray(value, dtype=np.float64)
            if row.shape != self.mean.shape or not np.isfinite(row).all():
                raise ValueError(f"normalization row must be finite shape {self.mean.shape}")
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (row - self.mean)

        def finish(self) -> tuple[np.ndarray, np.ndarray]:
            if self.count == 0:
                raise ValueError("cannot fit normalization on empty values")
            std = np.sqrt(self.m2 / self.count)
            std[std < 1e-8] = 1.0
            return self.mean, std

    node_values = Moments(NODE_FEATURE_DIM)
    proprio_values = Moments(PROPRIO_DIM)
    geometry_values = Moments(GEOMETRY_DIM)
    for record in records:
        for graph in (record["graph_prev"], record["graph_t"]):
            for node in _node_map(graph).values():
                node_values.add(_node_vector(node))
            proprio_values.add(_proprio(graph))
            for edge in _edge_map(graph).values():
                features = edge.get("features") or {}
                relative = features.get("relative_position", [0.0, 0.0, 0.0])
                if isinstance(relative, Sequence) and len(relative) == 3:
                    geometry_values.add([float(relative[0]), float(relative[1]), float(relative[2]), float(features.get("distance", 0.0) or 0.0), float(bool(features.get("distance_valid", 0)))])
    node_mean, node_std = node_values.finish()
    proprio_mean, proprio_std = proprio_values.finish()
    if geometry_values.count:
        geometry_mean, geometry_std = geometry_values.finish()
    else:
        geometry_mean = np.zeros(5, dtype=np.float64)
        geometry_std = np.ones(5, dtype=np.float64)
    return NormalizationStats(
        tuple(node_mean.tolist()), tuple(node_std.tolist()), tuple(proprio_mean.tolist()), tuple(proprio_std.tolist()),
        tuple(geometry_mean.tolist()), tuple(geometry_std.tolist()),
    )


def _relation_record(edge: Mapping[str, Any], relation: str) -> tuple[float, float]:
    records = edge.get("relations") or edge.get("semantic_relation")
    if not isinstance(records, Mapping):
        return 0.0, 0.0
    record = records.get(relation)
    if not isinstance(record, Mapping) or not bool(record.get("valid")) or not isinstance(record.get("value"), bool):
        return 0.0, 0.0
    return float(bool(record["value"])), 1.0


def graph_tensors(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    max_nodes: int | None = None,
) -> tuple[GraphBatch, GraphBatch, list[float], list[float]]:
    """Convert two graph JSON payloads to deterministic padded tensors."""

    order = _node_order(previous, current)
    if max_nodes is None:
        max_nodes = len(order)
    if len(order) > int(max_nodes):
        raise ValueError(f"graph node count {len(order)} exceeds max_nodes={max_nodes}")
    node_dim = NODE_FEATURE_DIM
    prev_nodes = _node_map(previous)
    curr_nodes = _node_map(current)
    for graph_name, nodes, edges in (
        ("previous", prev_nodes, _edge_map(previous)),
        ("current", curr_nodes, _edge_map(current)),
    ):
        for source, target in edges:
            if source not in nodes or target not in nodes:
                raise ValueError(
                    f"{graph_name} edge {source}->{target} references a missing node"
                )
    prev_node = np.zeros((max_nodes, node_dim), dtype=np.float32)
    curr_node = np.zeros((max_nodes, node_dim), dtype=np.float32)
    prev_mask = np.zeros(max_nodes, dtype=bool)
    curr_mask = np.zeros(max_nodes, dtype=bool)
    for index, node_id in enumerate(order):
        if node_id in prev_nodes:
            prev_node[index] = _node_vector(prev_nodes[node_id])
            prev_mask[index] = True
        if node_id in curr_nodes:
            curr_node[index] = _node_vector(curr_nodes[node_id])
            curr_mask[index] = True
    prev_geometry = np.zeros((max_nodes, max_nodes, GEOMETRY_DIM), dtype=np.float32)
    curr_geometry = np.zeros_like(prev_geometry)
    prev_contact = np.zeros((max_nodes, max_nodes, CONTACT_DIM), dtype=np.float32)
    curr_contact = np.zeros_like(prev_contact)
    prev_rel = np.zeros((max_nodes, max_nodes, RELATION_EDGE_DIM), dtype=np.float32)
    curr_rel = np.zeros_like(prev_rel)
    prev_edge_mask = np.zeros((max_nodes, max_nodes), dtype=bool)
    curr_edge_mask = np.zeros_like(prev_edge_mask)
    prev_edges = _edge_map(previous)
    curr_edges = _edge_map(current)
    relation_names = ("left", "right", "front", "behind", "above", "below", "on")
    for i, source in enumerate(order):
        for j, target in enumerate(order):
            if source == target:
                continue
            for graph_edges, geometry, contact, relations, mask in (
                (prev_edges, prev_geometry, prev_contact, prev_rel, prev_edge_mask),
                (curr_edges, curr_geometry, curr_contact, curr_rel, curr_edge_mask),
            ):
                edge = graph_edges.get((source, target))
                if edge is None:
                    continue
                features = edge.get("features") or {}
                relative = features.get("relative_position", [0.0, 0.0, 0.0])
                if not isinstance(relative, Sequence) or len(relative) != 3:
                    raise ValueError(f"edge {(source, target)} has invalid relative_position")
                geometry[i, j, :3] = [float(value) for value in relative]
                geometry[i, j, 3] = float(features.get("distance", 0.0) or 0.0)
                geometry[i, j, 4] = float(bool(features.get("distance_valid", 0)))
                contact_value, contact_valid = _relation_record(edge, "contact")
                contact[i, j, :2] = [contact_value, contact_valid]
                for relation_index, relation in enumerate(relation_names):
                    value, valid = _relation_record(edge, relation)
                    relations[i, j, relation_index * 2 : relation_index * 2 + 2] = [value, valid]
                mask[i, j] = True
    previous_graph = GraphBatch(
        torch.from_numpy(prev_node), torch.from_numpy(prev_mask), torch.from_numpy(prev_geometry), torch.from_numpy(prev_contact), torch.from_numpy(prev_rel), torch.from_numpy(prev_edge_mask)
    )
    current_graph = GraphBatch(
        torch.from_numpy(curr_node), torch.from_numpy(curr_mask), torch.from_numpy(curr_geometry), torch.from_numpy(curr_contact), torch.from_numpy(curr_rel), torch.from_numpy(curr_edge_mask)
    )
    return previous_graph, current_graph, _proprio(previous), _proprio(current)


def _target_relation(record: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    target = record.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("joined sample has no target payload")
    labels = target.get("relation_any_change")
    valid = target.get("relation_valid")
    if not isinstance(labels, Mapping) or not isinstance(valid, Mapping):
        raise ValueError("joined target relation labels are missing")
    return [float(bool(labels.get(name, 0))) for name in PRIMARY_RELATIONS], [float(bool(valid.get(name, 0))) for name in PRIMARY_RELATIONS]


def collate_phase3c(
    records: Sequence[Mapping[str, Any]],
    store: SemanticFeatureStore,
    *,
    max_nodes: int | None = None,
    normalization: NormalizationStats | None = None,
    device: torch.device | str | None = None,
) -> Phase3CBatch:
    if not records:
        raise ValueError("cannot collate an empty Phase 3C batch")
    if max_nodes is not None and int(max_nodes) <= 0:
        raise ValueError("max_nodes must be positive")
    if max_nodes is None:
        max_nodes = max(
            len(_node_order(record["graph_prev"], record["graph_t"]))
            for record in records
        )
    previous_graphs: list[GraphBatch] = []
    current_graphs: list[GraphBatch] = []
    v_history: list[torch.Tensor] = []
    target_v: list[torch.Tensor] = []
    p_history: list[torch.Tensor] = []
    target_p: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    languages: list[torch.Tensor] = []
    labels: list[list[float]] = []
    masks: list[list[float]] = []
    motion: list[float] = []
    sample_ids: list[str] = []
    task_ids: list[int] = []
    episode_ids: list[str] = []
    for record in records:
        task_id = int(record["task_id"])
        demo_key = str(record["demo_key"])
        prev_step, current_step, target_step = (int(record[name]) for name in ("prev_step", "current_step", "target_step"))
        tau = int(record.get("tau", 6))
        if tau != 6:
            raise ValueError(f"Phase 3C tensor contract requires tau=6, got {tau}")
        if current_step - prev_step != tau or target_step - current_step != tau:
            raise ValueError(
                f"non-uniform temporal steps for sample={record.get('sample_id')}: "
                f"{prev_step}->{current_step}->{target_step}, tau={tau}"
            )
        prev_graph, current_graph, prev_p, current_p = graph_tensors(record["graph_prev"], record["graph_t"], max_nodes=max_nodes)
        target_p_values = _proprio(record["target"]["graph"])
        previous_graphs.append(prev_graph)
        current_graphs.append(current_graph)
        p_history.append(torch.tensor([prev_p, current_p], dtype=torch.float32))
        target_p.append(torch.tensor(target_p_values, dtype=torch.float32))
        # Keep the two view embeddings separate: view0/view1, never a repeated
        # copy of one camera.
        def image_tensor(step: int, view: int) -> torch.Tensor:
            return torch.from_numpy(store.image(task_id, demo_key, step, view))

        v_history.append(torch.stack(
            [torch.stack([image_tensor(prev_step, 0), image_tensor(prev_step, 1)]),
             torch.stack([image_tensor(current_step, 0), image_tensor(current_step, 1)])]
        ))
        target_v.append(torch.stack([image_tensor(target_step, 0), image_tensor(target_step, 1)]))
        actions.append(torch.tensor(parse_action_window(record["past_action_window"], tau=tau), dtype=torch.float32))
        languages.append(torch.tensor(store.language(task_id, demo_key), dtype=torch.float32))
        label, valid = _target_relation(record)
        labels.append(label)
        masks.append(valid)
        motion_value = float(record["target"]["scene_max_displacement_m"])
        if not math.isfinite(motion_value) or motion_value < 0.0:
            raise ValueError("scene_max_displacement_m must be finite and non-negative")
        motion.append(motion_value)
        sample_ids.append(str(record["sample_id"]))
        task_ids.append(task_id)
        episode_ids.append(str(record["episode_id"]))

    def stack_graph(graphs: list[GraphBatch]) -> GraphBatch:
        return GraphBatch(*(torch.stack([getattr(graph, name) for graph in graphs]) for name in GraphBatch.__dataclass_fields__))
    graph_prev_batch = stack_graph(previous_graphs)
    graph_current_batch = stack_graph(current_graphs)
    p_history_batch = torch.stack(p_history)
    target_p_batch = torch.stack(target_p)
    if normalization is not None:
        graph_prev_batch = normalization.apply_graph(graph_prev_batch)
        graph_current_batch = normalization.apply_graph(graph_current_batch)
        p_history_batch = normalization.apply_proprio(p_history_batch)
        target_p_batch = normalization.apply_proprio(target_p_batch)
    batch = Phase3CBatch(
        sample_ids=tuple(sample_ids), task_ids=torch.tensor(task_ids, dtype=torch.long), episode_ids=tuple(episode_ids),
        v_history=torch.stack(v_history), p_history=p_history_batch, past_action=torch.stack(actions),
        language=torch.stack(languages), graph_prev=graph_prev_batch, graph_current=graph_current_batch,
        target_v=torch.stack(target_v), target_p=target_p_batch, target_relation_change=torch.tensor(labels, dtype=torch.float32),
        target_relation_mask=torch.tensor(masks, dtype=torch.float32), target_scene_motion=torch.tensor(motion, dtype=torch.float32).unsqueeze(-1),
    )
    if device is not None:
        def move_graph(graph: GraphBatch) -> GraphBatch:
            return GraphBatch(*(getattr(graph, name).to(device) for name in GraphBatch.__dataclass_fields__))
        batch = Phase3CBatch(
            sample_ids=batch.sample_ids, task_ids=batch.task_ids.to(device), episode_ids=batch.episode_ids,
            v_history=batch.v_history.to(device), p_history=batch.p_history.to(device), past_action=batch.past_action.to(device), language=batch.language.to(device),
            graph_prev=move_graph(batch.graph_prev), graph_current=move_graph(batch.graph_current), target_v=batch.target_v.to(device), target_p=batch.target_p.to(device),
            target_relation_change=batch.target_relation_change.to(device), target_relation_mask=batch.target_relation_mask.to(device), target_scene_motion=batch.target_scene_motion.to(device),
        )
    return batch


def iter_joined_batches(path: Path, batch_size: int) -> Iterable[list[dict[str, Any]]]:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    pending: list[dict[str, Any]] = []
    for record in iter_json_objects(path):
        pending.append(dict(record))
        if len(pending) >= int(batch_size):
            yield pending
            pending = []
    if pending:
        yield pending


def iter_filtered_records(
    path: Path,
    *,
    split: str | None = None,
    exclude_task_id: int | None = None,
    include_task_id: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream a split/task view without materializing the joined artifact."""

    if exclude_task_id is not None and include_task_id is not None:
        raise ValueError("exclude_task_id and include_task_id are mutually exclusive")
    for raw in iter_json_objects(path):
        if split is not None and str(raw.get("split")) != str(split):
            continue
        task_id = int(raw.get("task_id", -1))
        if exclude_task_id is not None and task_id == int(exclude_task_id):
            continue
        if include_task_id is not None and task_id != int(include_task_id):
            continue
        yield dict(raw)


def iter_shuffled_batches(
    path: Path,
    *,
    batch_size: int,
    seed: int,
    split: str | None = None,
    exclude_task_id: int | None = None,
    shuffle_buffer: int = 2048,
) -> Iterator[list[dict[str, Any]]]:
    """Yield deterministic bounded-memory shuffled batches for repeated epochs."""

    import random

    if int(batch_size) <= 0 or int(shuffle_buffer) < int(batch_size):
        raise ValueError("shuffle_buffer must be at least batch_size, and both must be positive")
    rng = random.Random(int(seed))
    pending_batch: list[dict[str, Any]] = []
    while True:
        buffer: list[dict[str, Any]] = []
        seen = 0
        for record in iter_filtered_records(
            path, split=split, exclude_task_id=exclude_task_id
        ):
            seen += 1
            if len(buffer) < int(shuffle_buffer):
                buffer.append(record)
                continue
            index = rng.randrange(len(buffer))
            pending_batch.append(buffer[index])
            buffer[index] = record
            if len(pending_batch) == int(batch_size):
                yield pending_batch
                pending_batch = []
        if seen == 0:
            raise ValueError(f"joined manifest has no matching records for split={split}")
        rng.shuffle(buffer)
        for record in buffer:
            pending_batch.append(record)
            if len(pending_batch) == int(batch_size):
                yield pending_batch
                pending_batch = []
