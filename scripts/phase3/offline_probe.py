"""Run a controlled Phase 3 offline relational-dynamics probe.

The probe consumes the validated Phase 2R dataset and predicts future
pairwise relations from current node features, current geometry, and an action
window.  It intentionally stops before CLaD integration: this isolates the
question of whether relational structure helps latent foresight at all.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


EDGE_RELATIONS = (
    "left",
    "right",
    "front",
    "behind",
    "above",
    "below",
    "contact",
    "on",
    "inside",
    "holding",
)
MODEL_IDS = (
    "p0_flat_mlp",
    "p1_node_no_message",
    "p2_gnn_empty_edge",
    "p3_gnn_geometry",
    "p4_gnn_soft_attention",
    "b1_target_object_mlp",
    "g1_sparse_holder_object_gnn",
    "g3_action_conditioned_holder_object_gnn",
)
EXPERIMENTAL_MODEL_IDS = (
    "b1_pair_feature_mlp_v2",
    "g2_flat_action_holder_object_gnn_v2",
    "g2_structured_action_holder_object_gnn",
    "g3v2_action_film_holder_object_gnn",
    "s0_no_action_holder_object_gnn_v2",
    "c_l_complete_late_action_gnn_v2",
    "c_e_complete_action_film_gnn_v2",
    "h0_pair_local_no_history_no_action",
    "h1_pair_local_history_no_action",
    "h2_pair_local_no_history_action_film",
    "h3_pair_local_history_action_film",
)
CONSTANT_ACTION_MODEL_IDS = (
    "g1_constant_action_holder_object_gnn_v2",
)
V2_MESSAGE_MODEL_IDS = (
    "g2_flat_action_holder_object_gnn_v2",
    "g2_structured_action_holder_object_gnn",
    "g3v2_action_film_holder_object_gnn",
    "s0_no_action_holder_object_gnn_v2",
    "c_l_complete_late_action_gnn_v2",
    "c_e_complete_action_film_gnn_v2",
)
STRUCTURED_ACTION_MODEL_IDS = (
    "g2_structured_action_holder_object_gnn",
    "g3v2_action_film_holder_object_gnn",
    "c_l_complete_late_action_gnn_v2",
    "c_e_complete_action_film_gnn_v2",
    "h2_pair_local_no_history_action_film",
    "h3_pair_local_history_action_film",
)
ACTION_FILM_MODEL_IDS = (
    "g3v2_action_film_holder_object_gnn",
    "c_e_complete_action_film_gnn_v2",
)
NO_ACTION_MODEL_IDS = (
    "s0_no_action_holder_object_gnn_v2",
    "s0_g1_no_action_holder_object_gnn_v2",
    "h0_pair_local_no_history_no_action",
    "h1_pair_local_history_no_action",
)
NODE_FEATURE_CONTRACTS = ("legacy_v1", "holder_object_v2")
EDGE_FEATURE_CONTRACTS = (
    "geometry_v1",
    "holder_object_v2",
    "holder_object_causal_history_v1",
)
ACTION_NORMALIZATION_CONTRACTS = ("flat_position_v1", "channel_v2")
HOLDER_OBJECT_V2_NODE_FEATURES = (
    "is_robot",
    "is_object",
    "is_fixture",
    "is_site",
    "position_valid",
    "gripper_qpos_0",
    "gripper_qpos_1",
    "gripper_qpos_valid",
    "gripper_aperture",
    "joint_velocity_0",
    "joint_velocity_1",
    "joint_velocity_2",
    "joint_velocity_3",
    "joint_velocity_4",
    "joint_velocity_5",
    "joint_velocity_6",
    "joint_velocity_valid",
)
HOLDER_OBJECT_V2_EDGE_FEATURES = (
    "relative_position_x",
    "relative_position_y",
    "relative_position_z",
    "distance",
    "geometry_valid",
    "current_contact",
    "current_contact_valid",
    "robot_to_object",
    "object_to_robot",
)
PAIR_LOCAL_MODEL_IDS = (
    "h0_pair_local_no_history_no_action",
    "h1_pair_local_history_no_action",
    "h2_pair_local_no_history_action_film",
    "h3_pair_local_history_action_film",
)
PAIR_LOCAL_HISTORY_MODEL_IDS = (
    "h1_pair_local_history_no_action",
    "h3_pair_local_history_action_film",
)
PAIR_LOCAL_ACTION_MODEL_IDS = (
    "h2_pair_local_no_history_action_film",
    "h3_pair_local_history_action_film",
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _finite_float_list(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return []
    return result if all(math.isfinite(item) for item in result) else []


def _edge_key(edge: Mapping[str, Any]) -> tuple[str, str]:
    return str(edge["source"]), str(edge["target"])


def _stable_sample_id(sample: Mapping[str, Any]) -> str:
    explicit = sample.get("sample_id")
    if explicit is not None:
        return str(explicit)
    fields = ("suite", "task_id", "episode_id", "start_step", "target_step", "tau")
    if all(sample.get(field) is not None for field in fields):
        return "|".join(str(sample[field]) for field in fields)
    return str(sample.get("episode_id", "unknown"))


def _relation_value(edge: Mapping[str, Any], relation: str) -> tuple[float, float]:
    record = edge.get("relations", {}).get(relation, {})
    if not isinstance(record, Mapping) or not record.get("valid"):
        return 0.0, 0.0
    return float(bool(record.get("value"))), 1.0


def _node_features(
    node: Mapping[str, Any],
    feature_contract: str = "legacy_v1",
) -> list[float]:
    if feature_contract not in NODE_FEATURE_CONTRACTS:
        raise ValueError(f"unknown node feature contract: {feature_contract}")
    node_type = str(node.get("node_type", "unknown"))
    type_one_hot = [float(node_type == candidate) for candidate in ("robot", "object", "fixture", "site")]
    if feature_contract == "legacy_v1":
        feature_vector = [float(value) for value in node.get("feature_vector", [])]
        return feature_vector + type_one_hot

    # The holder-object contract removes absolute position and joint position.
    # Pair geometry already carries robot-relative position, while absolute arm
    # configuration can become a task-progress shortcut in the small task set.
    features = node.get("features", {})
    if not isinstance(features, Mapping):
        features = {}
    position_valid = float(features.get("position_valid") == 1)
    gripper_qpos = _finite_float_list(features.get("gripper_qpos"))
    gripper_valid = float(features.get("gripper_qpos_valid") == 1 and len(gripper_qpos) == 2)
    if not gripper_valid:
        gripper_qpos = [0.0, 0.0]
    gripper_aperture = max((abs(value) for value in gripper_qpos), default=0.0)
    joint_velocity = _finite_float_list(features.get("joint_vel"))
    joint_velocity_valid = float(features.get("joint_vel_valid") == 1 and len(joint_velocity) == 7)
    if not joint_velocity_valid:
        joint_velocity = [0.0] * 7
    return (
        type_one_hot
        + [position_valid]
        + gripper_qpos
        + [gripper_valid, gripper_aperture]
        + joint_velocity
        + [joint_velocity_valid]
    )


def _edge_geometry(
    edge: Mapping[str, Any],
    feature_contract: str = "geometry_v1",
    node_types: Mapping[str, str] | None = None,
) -> list[float]:
    if feature_contract not in EDGE_FEATURE_CONTRACTS:
        raise ValueError(f"unknown edge feature contract: {feature_contract}")
    features = edge.get("features", {})
    if not isinstance(features, Mapping):
        features = {}
    relative = _finite_float_list(features.get("relative_position"))
    relative = relative if len(relative) == 3 else [0.0, 0.0, 0.0]
    try:
        distance = float(features.get("distance", 0.0) or 0.0)
    except (TypeError, ValueError):
        distance = 0.0
    if not math.isfinite(distance):
        distance = 0.0
    distance_valid = float(features.get("distance_valid", 0) or 0)
    geometry = relative + [distance, distance_valid]
    if feature_contract == "geometry_v1":
        return geometry

    contact_value, contact_valid = _relation_value(edge, "contact")
    source = str(edge.get("source"))
    target = str(edge.get("target"))
    pair = edge.get("node_type_pair")
    if isinstance(pair, Sequence) and not isinstance(pair, (str, bytes)) and len(pair) == 2:
        source_type, target_type = str(pair[0]), str(pair[1])
    else:
        resolved_types = node_types or {}
        source_type, target_type = resolved_types.get(source, "unknown"), resolved_types.get(target, "unknown")
    direction = [
        float(source_type == "robot" and target_type == "object"),
        float(source_type == "object" and target_type == "robot"),
    ]
    # Current holding is intentionally excluded: it is a temporal weak label
    # derived from contact, closure, and past motion. Feeding it back would let
    # the probe copy the label instead of learning action-conditioned dynamics.
    holder_object = geometry + [contact_value, contact_valid] + direction
    if feature_contract == "holder_object_v2":
        return holder_object
    history = _finite_float_list(features.get("causal_pair_history_v1"))
    if not history:
        raise ValueError(
            "holder_object_causal_history_v1 requires edge.features."
            "causal_pair_history_v1"
        )
    return holder_object + history


def _record_from_sample(
    sample: Mapping[str, Any],
    relations: Sequence[str] = EDGE_RELATIONS,
    node_feature_contract: str = "legacy_v1",
    edge_feature_contract: str = "geometry_v1",
) -> dict[str, Any]:
    graph = sample["graph_t"]
    target_graph = sample["graph_target"]
    nodes = graph.get("nodes", [])
    node_ids = [str(node["node_id"]) for node in nodes]
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    node_types = {str(node["node_id"]): str(node.get("node_type", "unknown")) for node in nodes}
    current_edges = {_edge_key(edge): edge for edge in graph.get("edges", [])}
    target_edges = {_edge_key(edge): edge for edge in target_graph.get("edges", [])}
    raw_prediction_edge_keys = sample.get("prediction_edge_keys")
    prediction_edge_keys = (
        {
            (str(key[0]), str(key[1]))
            for key in raw_prediction_edge_keys
            if isinstance(key, (list, tuple)) and len(key) == 2
        }
        if isinstance(raw_prediction_edge_keys, (list, tuple))
        else None
    )

    edge_keys = [key for key in current_edges if key in target_edges]
    current_labels: list[list[float]] = []
    future_labels: list[list[float]] = []
    current_valid: list[list[float]] = []
    future_valid: list[list[float]] = []
    changed: list[list[float]] = []
    edge_src: list[int] = []
    edge_tgt: list[int] = []
    edge_geometry: list[list[float]] = []
    for key in edge_keys:
        current_edge = current_edges[key]
        future_edge = target_edges[key]
        is_prediction_edge = prediction_edge_keys is None or key in prediction_edge_keys
        edge_src.append(node_index[key[0]])
        edge_tgt.append(node_index[key[1]])
        edge_geometry.append(_edge_geometry(current_edge, edge_feature_contract, node_types))
        now_row: list[float] = []
        future_row: list[float] = []
        now_mask_row: list[float] = []
        future_mask_row: list[float] = []
        changed_row: list[float] = []
        for relation in relations:
            now_value, now_is_valid = _relation_value(current_edge, relation)
            future_value, future_is_valid = _relation_value(future_edge, relation)
            now_row.append(now_value)
            future_row.append(future_value)
            now_mask_row.append(float(now_is_valid and is_prediction_edge))
            future_mask_row.append(float(future_is_valid and is_prediction_edge))
            changed_row.append(
                float(
                    is_prediction_edge
                    and now_is_valid
                    and future_is_valid
                    and now_value != future_value
                )
            )
        current_labels.append(now_row)
        future_labels.append(future_row)
        current_valid.append(now_mask_row)
        future_valid.append(future_mask_row)
        changed.append(changed_row)

    actions = []
    for entry in sample.get("action_window", []):
        action_value = entry.get("action", []) if isinstance(entry, Mapping) else entry
        actions.append(_finite_float_list(action_value))
    action_dim = max((len(action) for action in actions), default=0)
    action_step_mask: list[float] = []
    action_flat: list[float] = []
    for action in actions:
        valid = bool(action_dim and len(action) == action_dim)
        action_step_mask.append(float(valid))
        action_flat.extend(action if valid else [0.0] * action_dim)
    return {
        "sample_id": _stable_sample_id(sample),
        "episode_id": str(sample.get("episode_id")),
        "suite": sample.get("suite"),
        "task_id": sample.get("task_id"),
        "start_step": sample.get("start_step"),
        "target_step": sample.get("target_step"),
        "tau": sample.get("tau"),
        "event_id": sample.get("event_id"),
        "split": str(sample.get("split", "train")),
        "edge_keys": [[str(source), str(target)] for source, target in edge_keys],
        "node_features": [_node_features(node, node_feature_contract) for node in nodes],
        "edge_src": edge_src,
        "edge_tgt": edge_tgt,
        "edge_geometry": edge_geometry,
        "current_labels": current_labels,
        "future_labels": future_labels,
        "current_valid": current_valid,
        "future_valid": future_valid,
        "changed": changed,
        "actions": action_flat,
        "action_steps": len(actions),
        "action_dim": action_dim,
        "action_step_mask": action_step_mask,
        "node_feature_contract": node_feature_contract,
        "edge_feature_contract": edge_feature_contract,
    }


def _sample_task_family(sample: Mapping[str, Any]) -> str:
    suite = sample.get("suite")
    task_id = sample.get("task_id")
    if suite is None or task_id is None:
        raise ValueError(f"sample is missing suite/task_id: {sample.get('episode_id')}")
    return f"{suite}:{task_id}"


def _apply_split_override(sample: Mapping[str, Any], split_config: Mapping[str, Any] | None) -> dict[str, Any]:
    output = dict(sample)
    if not isinstance(split_config, Mapping):
        return output
    validation = {str(value) for value in split_config.get("validation_task_families", [])}
    test = {str(value) for value in split_config.get("test_task_families", [])}
    family = _sample_task_family(sample)
    output["split"] = "test" if family in test else "validation" if family in validation else "train"
    return output


def load_probe_records(
    dataset: Mapping[str, Any],
    split_config: Mapping[str, Any] | None = None,
    relations: Sequence[str] = EDGE_RELATIONS,
    node_feature_contract: str = "legacy_v1",
    edge_feature_contract: str = "geometry_v1",
) -> list[dict[str, Any]]:
    samples = dataset.get("samples", [])
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise ValueError("dataset.samples must be a sequence")
    records = [
        _record_from_sample(
            _apply_split_override(sample, split_config),
            relations,
            node_feature_contract=node_feature_contract,
            edge_feature_contract=edge_feature_contract,
        )
        for sample in samples
    ]
    if not records:
        raise ValueError("dataset contains no usable samples")
    return records


def _normalization(
    records: Sequence[Mapping[str, Any]],
    action_contract: str = "flat_position_v1",
) -> dict[str, np.ndarray]:
    if action_contract not in ACTION_NORMALIZATION_CONTRACTS:
        raise ValueError(f"unknown action normalization contract: {action_contract}")
    node_rows = np.asarray(
        [row for record in records for row in record["node_features"]], dtype=np.float32
    )
    edge_rows = np.asarray(
        [row for record in records for row in record["edge_geometry"]], dtype=np.float32
    )
    # Action windows are flattened, but their lengths can differ when the
    # source trajectory does not contain the full requested horizon.  Keep
    # the vector representation used by the probe while padding only for
    # normalization; constructing an ndarray directly from ragged lists
    # raises a ValueError before the probe can run.
    action_dim = max((len(record["actions"]) for record in records), default=0)
    action_rows = np.zeros((len(records), action_dim), dtype=np.float32)
    action_present = np.zeros((len(records), action_dim), dtype=bool)
    for row_index, record in enumerate(records):
        values = np.asarray(record["actions"], dtype=np.float32)
        if values.size:
            action_rows[row_index, : len(values)] = values
            action_present[row_index, : len(values)] = True

    def stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = values.mean(axis=0) if values.size else np.zeros((values.shape[-1],), dtype=np.float32)
        std = values.std(axis=0) if values.size else np.ones((values.shape[-1],), dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)

    node_mean, node_std = stats(node_rows)
    edge_mean, edge_std = stats(edge_rows)
    if action_dim and action_contract == "channel_v2":
        step_dims = {int(record.get("action_dim", 0)) for record in records if record.get("action_dim", 0)}
        if len(step_dims) != 1:
            raise ValueError(f"channel_v2 requires one action step dimension, got {sorted(step_dims)}")
        step_dim = next(iter(step_dims))
        channel_rows: list[list[float]] = []
        for record in records:
            values = list(record["actions"])
            masks = list(record.get("action_step_mask", []))
            steps = int(record.get("action_steps", 0))
            for step in range(steps):
                start = step * step_dim
                row = values[start : start + step_dim]
                if len(row) == step_dim and (not masks or masks[step]):
                    channel_rows.append(row)
        channel_values = np.asarray(channel_rows, dtype=np.float32)
        channel_mean, channel_std = stats(channel_values)
        repeats = int(math.ceil(action_dim / step_dim))
        action_mean = np.tile(channel_mean, repeats)[:action_dim]
        action_std = np.tile(channel_std, repeats)[:action_dim]
    elif action_dim:
        action_mean = np.zeros(action_dim, dtype=np.float32)
        action_std = np.ones(action_dim, dtype=np.float32)
        for column in range(action_dim):
            observed = action_rows[action_present[:, column], column]
            if observed.size:
                action_mean[column] = float(observed.mean())
                action_std[column] = float(observed.std())
        action_std = np.where(action_std < 1e-6, 1.0, action_std)
    else:
        action_mean = np.zeros((0,), dtype=np.float32)
        action_std = np.ones((0,), dtype=np.float32)
    return {
        "node_mean": node_mean,
        "node_std": node_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }


@dataclass
class ProbeShape:
    max_nodes: int
    max_edges: int
    node_dim: int
    edge_dim: int
    action_dim: int
    relation_dim: int
    action_steps: int = 0
    action_step_dim: int = 0
    history_dim: int = 0


class ProbeCollator:
    def __init__(self, shape: ProbeShape, normalization: Mapping[str, np.ndarray]):
        self.shape = shape
        self.normalization = normalization

    def __call__(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        batch_size = len(records)
        shape = self.shape
        node_x = np.zeros((batch_size, shape.max_nodes, shape.node_dim), dtype=np.float32)
        node_mask = np.zeros((batch_size, shape.max_nodes), dtype=np.float32)
        edge_src = np.zeros((batch_size, shape.max_edges), dtype=np.int64)
        edge_tgt = np.zeros((batch_size, shape.max_edges), dtype=np.int64)
        edge_geometry = np.zeros((batch_size, shape.max_edges, shape.edge_dim), dtype=np.float32)
        edge_mask = np.zeros((batch_size, shape.max_edges), dtype=np.float32)
        current_labels = np.zeros((batch_size, shape.max_edges, shape.relation_dim), dtype=np.float32)
        future_labels = np.zeros_like(current_labels)
        current_valid = np.zeros_like(current_labels)
        future_valid = np.zeros_like(current_labels)
        changed = np.zeros_like(current_labels)
        actions = np.zeros((batch_size, shape.action_dim), dtype=np.float32)
        training_donor_actions = np.zeros(
            (batch_size, shape.action_dim), dtype=np.float32
        )
        zero_actions = np.zeros((batch_size, shape.action_dim), dtype=np.float32)
        if shape.action_dim:
            zero_actions[:] = (
                -self.normalization["action_mean"][: shape.action_dim]
                / self.normalization["action_std"][: shape.action_dim]
            )
        action_step_mask = np.zeros((batch_size, shape.action_steps), dtype=np.float32)
        training_donor_action_step_mask = np.zeros(
            (batch_size, shape.action_steps), dtype=np.float32
        )
        for batch_index, record in enumerate(records):
            node_count = min(shape.max_nodes, len(record["node_features"]))
            edge_count = min(shape.max_edges, len(record["edge_src"]))
            node_values = np.asarray(record["node_features"][:node_count], dtype=np.float32)
            node_x[batch_index, :node_count] = (
                node_values - self.normalization["node_mean"]
            ) / self.normalization["node_std"]
            node_mask[batch_index, :node_count] = 1.0
            edge_src[batch_index, :edge_count] = np.asarray(record["edge_src"][:edge_count], dtype=np.int64)
            edge_tgt[batch_index, :edge_count] = np.asarray(record["edge_tgt"][:edge_count], dtype=np.int64)
            edge_values = np.asarray(record["edge_geometry"][:edge_count], dtype=np.float32)
            edge_geometry[batch_index, :edge_count] = (
                edge_values - self.normalization["edge_mean"]
            ) / self.normalization["edge_std"]
            edge_mask[batch_index, :edge_count] = 1.0
            current_labels[batch_index, :edge_count] = record["current_labels"][:edge_count]
            future_labels[batch_index, :edge_count] = record["future_labels"][:edge_count]
            current_valid[batch_index, :edge_count] = record["current_valid"][:edge_count]
            future_valid[batch_index, :edge_count] = record["future_valid"][:edge_count]
            changed[batch_index, :edge_count] = record["changed"][:edge_count]
            action_values = np.asarray(record["actions"][: shape.action_dim], dtype=np.float32)
            actions[batch_index, : len(action_values)] = (
                action_values - self.normalization["action_mean"][: len(action_values)]
            ) / self.normalization["action_std"][: len(action_values)]
            donor_values = np.asarray(
                record.get("training_donor_actions", record["actions"])[
                    : shape.action_dim
                ],
                dtype=np.float32,
            )
            training_donor_actions[batch_index, : len(donor_values)] = (
                donor_values
                - self.normalization["action_mean"][: len(donor_values)]
            ) / self.normalization["action_std"][: len(donor_values)]
            step_mask_values = np.asarray(
                record.get("action_step_mask", [])[: shape.action_steps], dtype=np.float32
            )
            action_step_mask[batch_index, : len(step_mask_values)] = step_mask_values
            donor_step_mask_values = np.asarray(
                record.get(
                    "training_donor_action_step_mask",
                    record.get("action_step_mask", []),
                )[: shape.action_steps],
                dtype=np.float32,
            )
            training_donor_action_step_mask[
                batch_index, : len(donor_step_mask_values)
            ] = donor_step_mask_values
        return {
            "node_x": torch.from_numpy(node_x),
            "node_mask": torch.from_numpy(node_mask),
            "edge_src": torch.from_numpy(edge_src),
            "edge_tgt": torch.from_numpy(edge_tgt),
            "edge_geometry": torch.from_numpy(edge_geometry),
            "edge_mask": torch.from_numpy(edge_mask),
            "current_labels": torch.from_numpy(current_labels),
            "future_labels": torch.from_numpy(future_labels),
            "current_valid": torch.from_numpy(current_valid),
            "future_valid": torch.from_numpy(future_valid),
            "changed": torch.from_numpy(changed),
            "actions": torch.from_numpy(actions),
            "training_donor_actions": torch.from_numpy(training_donor_actions),
            "zero_actions": torch.from_numpy(zero_actions),
            "action_step_mask": torch.from_numpy(action_step_mask),
            "training_donor_action_step_mask": torch.from_numpy(
                training_donor_action_step_mask
            ),
            "sample_id": [str(record.get("sample_id")) for record in records],
            "episode_id": [str(record.get("episode_id")) for record in records],
            "suite": [record.get("suite") for record in records],
            "task_id": [record.get("task_id") for record in records],
            "start_step": [record.get("start_step") for record in records],
            "target_step": [record.get("target_step") for record in records],
            "tau": [record.get("tau") for record in records],
            "event_id": [record.get("event_id") for record in records],
            "edge_keys": [list(record.get("edge_keys", [])) for record in records],
        }


def _mlp(nn_module, input_dim: int, hidden_dim: int, output_dim: int) -> Any:
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class StructuredActionEncoder(nn.Module):
    """Encode a fixed LIBERO action window without destroying its time layout.

    The final action channel is treated as the gripper command and the earlier
    channels as arm motion. Learned step embeddings retain order, while raw
    first/last/mean/sum/delta summaries make persistent and changing commands
    directly accessible to a small smoke-test model.
    """

    def __init__(self, shape: ProbeShape, hidden_dim: int):
        super().__init__()
        if shape.action_steps <= 0 or shape.action_step_dim < 2:
            raise ValueError(
                "structured action encoding requires action_steps > 0 and "
                "action_step_dim >= 2"
            )
        if shape.action_steps * shape.action_step_dim != shape.action_dim:
            raise ValueError(
                "structured action layout does not match flattened action_dim: "
                f"{shape.action_steps}*{shape.action_step_dim}!={shape.action_dim}"
            )
        self.action_steps = shape.action_steps
        self.action_step_dim = shape.action_step_dim
        branch_dim = max(hidden_dim // 2, 8)
        self.arm_encoder = _mlp(nn, shape.action_step_dim - 1, branch_dim, branch_dim)
        self.gripper_encoder = _mlp(nn, 1, branch_dim, branch_dim)
        self.step_projection = nn.Linear(branch_dim * 2, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, shape.action_steps, hidden_dim))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.step_attention = nn.Linear(hidden_dim, 1)
        summary_dim = shape.action_step_dim * 5 + 3
        self.summary_encoder = _mlp(nn, summary_dim, hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, actions, action_step_mask=None):
        batch = actions.shape[0]
        steps = actions.reshape(batch, self.action_steps, self.action_step_dim)
        if action_step_mask is None or action_step_mask.shape[1] != self.action_steps:
            action_step_mask = torch.ones(
                batch, self.action_steps, dtype=actions.dtype, device=actions.device
            )
        mask = action_step_mask > 0
        mask_f = mask.to(actions.dtype)
        any_valid = mask.any(dim=1, keepdim=True)

        arm_h = self.arm_encoder(steps[..., :-1])
        gripper_h = self.gripper_encoder(steps[..., -1:])
        step_h = self.step_projection(torch.cat([arm_h, gripper_h], dim=-1))
        step_h = step_h + self.position_embedding[:, : self.action_steps]
        scores = self.step_attention(torch.tanh(step_h)).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1) * mask_f
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = (step_h * weights.unsqueeze(-1)).sum(dim=1)

        count = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        masked_steps = steps * mask_f.unsqueeze(-1)
        action_sum = masked_steps.sum(dim=1)
        action_mean = action_sum / count
        indices = torch.arange(self.action_steps, device=actions.device).view(1, -1)
        first_index = torch.where(mask, indices, self.action_steps).min(dim=1).values
        last_index = torch.where(mask, indices, -1).max(dim=1).values
        safe_first = first_index.clamp(min=0, max=self.action_steps - 1)
        safe_last = last_index.clamp(min=0, max=self.action_steps - 1)
        batch_index = torch.arange(batch, device=actions.device)
        first = steps[batch_index, safe_first] * any_valid.to(actions.dtype)
        last = steps[batch_index, safe_last] * any_valid.to(actions.dtype)
        delta = last - first
        gripper = steps[..., -1]
        gripper_min = gripper.masked_fill(~mask, float("inf")).min(dim=1).values
        gripper_max = gripper.masked_fill(~mask, float("-inf")).max(dim=1).values
        gripper_min = torch.where(any_valid.squeeze(1), gripper_min, torch.zeros_like(gripper_min))
        gripper_max = torch.where(any_valid.squeeze(1), gripper_max, torch.zeros_like(gripper_max))
        coverage = mask_f.mean(dim=1)
        summary = torch.cat(
            [first, last, action_mean, action_sum, delta, gripper_min[:, None], gripper_max[:, None], coverage[:, None]],
            dim=-1,
        )
        encoded = self.output_norm(pooled + self.summary_encoder(summary))
        return encoded * any_valid.to(actions.dtype)


class RelationalDynamicsProbe(nn.Module):
    def __init__(
        self,
        model_id: str,
        shape: ProbeShape,
        hidden_dim: int = 64,
        current_head_contract: str = "legacy",
    ):
        super().__init__()
        if current_head_contract not in {"legacy", "action_free_pair"}:
            raise ValueError(
                f"unknown current_head_contract={current_head_contract}"
            )
        self.model_id = model_id
        self.shape = shape
        self.hidden_dim = hidden_dim
        self.current_head_contract = current_head_contract
        self.uses_action = model_id not in NO_ACTION_MODEL_IDS
        self.uses_history = model_id in PAIR_LOCAL_HISTORY_MODEL_IDS
        self.structured_action = model_id in STRUCTURED_ACTION_MODEL_IDS
        if model_id != "p0_flat_mlp":
            self.node_encoder = _mlp(nn, shape.node_dim, hidden_dim, hidden_dim)
            if self.uses_action:
                self.action_encoder = (
                    StructuredActionEncoder(shape, hidden_dim)
                    if self.structured_action
                    else _mlp(nn, shape.action_dim, hidden_dim, hidden_dim)
                )
        self.relation_dim = shape.relation_dim
        if model_id == "p0_flat_mlp":
            flat_dim = shape.max_nodes * shape.node_dim + shape.max_nodes + shape.action_dim
            self.global_encoder = _mlp(nn, flat_dim, hidden_dim, hidden_dim)
            self.edge_encoder = _mlp(nn, hidden_dim + shape.edge_dim, hidden_dim, hidden_dim)
        elif model_id in {"p1_node_no_message", "b1_target_object_mlp"}:
            self.edge_encoder = _mlp(nn, hidden_dim * 3, hidden_dim, hidden_dim)
        elif model_id == "b1_pair_feature_mlp_v2":
            # B1 must receive exactly the same pair geometry/state features as
            # the sparse GNN. Otherwise B1-vs-G1 confounds message passing with
            # input availability.
            self.edge_encoder = _mlp(
                nn, hidden_dim * 3 + shape.edge_dim, hidden_dim, hidden_dim
            )
        elif model_id in PAIR_LOCAL_MODEL_IDS:
            if shape.history_dim <= 0 or shape.history_dim >= shape.edge_dim:
                raise ValueError(
                    "pair-local H0--H3 models require 0 < history_dim < edge_dim"
                )
            self.base_edge_dim = shape.edge_dim - shape.history_dim
            pair_input_dim = hidden_dim * 2 + self.base_edge_dim
            if self.uses_history:
                pair_input_dim += shape.history_dim
            self.pair_encoder = _mlp(
                nn, pair_input_dim, hidden_dim, hidden_dim
            )
            if model_id in PAIR_LOCAL_ACTION_MODEL_IDS:
                self.action_film = nn.Linear(hidden_dim * 3, hidden_dim * 2 + 1)
                nn.init.zeros_(self.action_film.weight)
                nn.init.zeros_(self.action_film.bias)
                nn.init.constant_(self.action_film.bias[-1:], 4.0)
            # All four cells retain the same post-pair transition depth.  The
            # action cells modulate the pair token before this block rather
            # than concatenating action only at the prediction head.
            self.edge_encoder = _mlp(nn, hidden_dim, hidden_dim, hidden_dim)
        elif model_id == "p2_gnn_empty_edge":
            self.node_update = _mlp(nn, hidden_dim * 2, hidden_dim, hidden_dim)
            self.edge_encoder = _mlp(nn, hidden_dim * 3, hidden_dim, hidden_dim)
        elif model_id in {
            "p3_gnn_geometry",
            "p4_gnn_soft_attention",
            "g1_sparse_holder_object_gnn",
            "g1_constant_action_holder_object_gnn_v2",
            "g3_action_conditioned_holder_object_gnn",
            "s0_g1_no_action_holder_object_gnn_v2",
        }:
            message_input_dim = hidden_dim * 2 + shape.edge_dim
            if model_id == "g3_action_conditioned_holder_object_gnn":
                message_input_dim += hidden_dim
            self.message_encoder = _mlp(nn, message_input_dim, hidden_dim, hidden_dim)
            self.node_update = _mlp(nn, hidden_dim * 2, hidden_dim, hidden_dim)
            if model_id == "g3_action_conditioned_holder_object_gnn":
                self.edge_gate = nn.Sequential(
                    nn.Linear(message_input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                self.node_norm = nn.LayerNorm(hidden_dim)
            if model_id == "p4_gnn_soft_attention":
                self.edge_attention = nn.Sequential(
                    nn.Linear(hidden_dim * 2 + shape.edge_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
            edge_input_dim = hidden_dim * 2 + shape.edge_dim
            if self.uses_action:
                edge_input_dim += hidden_dim
            self.edge_encoder = _mlp(
                nn, edge_input_dim, hidden_dim, hidden_dim
            )
        elif model_id in V2_MESSAGE_MODEL_IDS:
            pair_input_dim = hidden_dim * 2 + shape.edge_dim
            self.pair_encoder = _mlp(nn, pair_input_dim, hidden_dim, hidden_dim)
            self.node_update = _mlp(nn, hidden_dim * 2, hidden_dim, hidden_dim)
            self.node_norm = nn.LayerNorm(hidden_dim)
            # Pair/action/product terms expose object-specific action alignment
            # explicitly. FiLM can alter message content, unlike a scalar gate
            # that can only turn one shared message up/down.
            if model_id in ACTION_FILM_MODEL_IDS:
                self.action_film = nn.Linear(hidden_dim * 3, hidden_dim * 2 + 1)
                # Begin close to the unconditioned pair message. This keeps the
                # S-LS vs S-EF comparison from being dominated by an arbitrary
                # random modulation at initialization while leaving every FiLM
                # parameter trainable.
                nn.init.zeros_(self.action_film.weight)
                nn.init.zeros_(self.action_film.bias)
                nn.init.constant_(self.action_film.bias[-1:], 4.0)
            future_edge_input_dim = hidden_dim * 2 + shape.edge_dim
            if self.uses_action:
                future_edge_input_dim += hidden_dim
            self.edge_encoder = _mlp(
                nn, future_edge_input_dim, hidden_dim, hidden_dim
            )
        else:
            raise ValueError(f"unknown model_id={model_id}")
        if (
            model_id in EXPERIMENTAL_MODEL_IDS
            or current_head_contract == "action_free_pair"
        ):
            current_edge_dim = (
                shape.edge_dim - shape.history_dim
                if model_id in PAIR_LOCAL_MODEL_IDS
                else shape.edge_dim
            )
            self.current_pair_encoder = _mlp(
                nn, hidden_dim * 2 + current_edge_dim, hidden_dim, hidden_dim
            )
        self.future_head = nn.Linear(hidden_dim, shape.relation_dim)
        self.current_head = nn.Linear(hidden_dim, shape.relation_dim)

    def _gather(self, values, indices):
        batch = values.shape[0]
        batch_indices = torch.arange(batch, device=values.device).unsqueeze(1)
        return values[batch_indices, indices.clamp(min=0, max=values.shape[1] - 1)]

    def _aggregate_messages(self, node_h, edge_src, edge_tgt, edge_geometry, edge_mask, action_h=None):
        source_h = self._gather(node_h, edge_src)
        target_h = self._gather(node_h, edge_tgt)
        message_input = torch.cat([source_h, target_h, edge_geometry], dim=-1)
        if self.model_id == "g3_action_conditioned_holder_object_gnn":
            if action_h is None:
                raise ValueError("G3 requires an action embedding for edge messages")
            action_edge = action_h.unsqueeze(1).expand(-1, edge_src.shape[1], -1)
            message_input = torch.cat([message_input, action_edge], dim=-1)
        messages = self.message_encoder(message_input)
        if self.model_id == "g3_action_conditioned_holder_object_gnn":
            gate = torch.sigmoid(self.edge_gate(message_input))
            messages = messages * gate
        elif self.model_id == "p4_gnn_soft_attention":
            weights = torch.sigmoid(self.edge_attention(message_input))
            messages = messages * weights
        messages = messages * edge_mask.unsqueeze(-1)
        batch, node_count, hidden = node_h.shape
        offsets = torch.arange(batch, device=node_h.device).view(batch, 1) * node_count
        flat_target = (edge_tgt + offsets).reshape(-1)
        aggregate = torch.zeros(batch * node_count, hidden, device=node_h.device)
        aggregate.index_add_(0, flat_target, messages.reshape(-1, hidden))
        degree = torch.zeros(batch * node_count, 1, device=node_h.device)
        degree.index_add_(0, flat_target, edge_mask.reshape(-1, 1))
        aggregate = aggregate.reshape(batch, node_count, hidden) / degree.clamp_min(1.0).reshape(batch, node_count, 1)
        return aggregate

    def _aggregate_v2_messages(
        self, node_h, edge_src, edge_tgt, edge_geometry, edge_mask, action_h=None
    ):
        source_h = self._gather(node_h, edge_src)
        target_h = self._gather(node_h, edge_tgt)
        pair_h = self.pair_encoder(torch.cat([source_h, target_h, edge_geometry], dim=-1))
        if self.model_id in ACTION_FILM_MODEL_IDS:
            if action_h is None:
                raise ValueError(f"{self.model_id} requires an action embedding")
            action_edge = action_h.unsqueeze(1).expand_as(pair_h)
            film_input = torch.cat([pair_h, action_edge, pair_h * action_edge], dim=-1)
            gamma, beta, gate_logit = torch.split(
                self.action_film(film_input), [self.hidden_dim, self.hidden_dim, 1], dim=-1
            )
            messages = torch.sigmoid(gate_logit) * (
                pair_h * (1.0 + torch.tanh(gamma)) + beta
            )
        else:
            messages = pair_h
        messages = messages * edge_mask.unsqueeze(-1)
        batch, node_count, hidden = node_h.shape
        offsets = torch.arange(batch, device=node_h.device).view(batch, 1) * node_count
        flat_target = (edge_tgt + offsets).reshape(-1)
        aggregate = torch.zeros(batch * node_count, hidden, device=node_h.device)
        aggregate.index_add_(0, flat_target, messages.reshape(-1, hidden))
        degree = torch.zeros(batch * node_count, 1, device=node_h.device)
        degree.index_add_(0, flat_target, edge_mask.reshape(-1, 1))
        return aggregate.reshape(batch, node_count, hidden) / degree.clamp_min(1.0).reshape(
            batch, node_count, 1
        )

    def forward(
        self,
        node_x,
        node_mask,
        edge_src,
        edge_tgt,
        edge_geometry,
        edge_mask,
        actions,
        edge_shuffle=False,
        action_step_mask=None,
    ):
        import torch

        if edge_shuffle:
            edge_src = torch.roll(edge_src, shifts=1, dims=1)
            edge_tgt = torch.roll(edge_tgt, shifts=-1, dims=1)
        current_edge_h = None
        if self.model_id == "p0_flat_mlp":
            flat = torch.cat([node_x.reshape(node_x.shape[0], -1), node_mask, actions], dim=-1)
            graph_h = self.global_encoder(flat).unsqueeze(1).expand(-1, edge_src.shape[1], -1)
            edge_h = self.edge_encoder(torch.cat([graph_h, edge_geometry], dim=-1))
        else:
            action_h = None
            if self.uses_action:
                # Equal-parameter capacity control for G1.  Every example sees
                # the same fixed template, so the action encoder can learn a
                # constant branch but cannot receive sample-specific action
                # information.  Keeping the original encoder makes the
                # parameter count exactly equal to G1.
                action_input = (
                    torch.ones_like(actions)
                    if self.model_id in CONSTANT_ACTION_MODEL_IDS
                    else actions
                )
                action_h = (
                    self.action_encoder(action_input, action_step_mask)
                    if self.structured_action
                    else self.action_encoder(action_input)
                )
            node_h = self.node_encoder(node_x)
            node_h = node_h * node_mask.unsqueeze(-1)
            if hasattr(self, "current_pair_encoder"):
                current_source_h = self._gather(node_h, edge_src)
                current_target_h = self._gather(node_h, edge_tgt)
                current_geometry = (
                    edge_geometry[..., : self.base_edge_dim]
                    if self.model_id in PAIR_LOCAL_MODEL_IDS
                    else edge_geometry
                )
                current_edge_h = self.current_pair_encoder(
                    torch.cat(
                        [current_source_h, current_target_h, current_geometry], dim=-1
                    )
                )
            if self.model_id == "p2_gnn_empty_edge":
                for _ in range(2):
                    total = node_h.sum(dim=1, keepdim=True)
                    count = node_mask.sum(dim=1, keepdim=True).unsqueeze(-1)
                    aggregate = (total - node_h) / (count - 1.0).clamp_min(1.0)
                    node_h = self.node_update(torch.cat([node_h, aggregate], dim=-1))
                    node_h = node_h * node_mask.unsqueeze(-1)
                source_h = self._gather(node_h, edge_src)
                target_h = self._gather(node_h, edge_tgt)
                edge_h = self.edge_encoder(torch.cat([source_h, target_h, action_h.unsqueeze(1).expand_as(source_h)], dim=-1))
            elif self.model_id in {"p1_node_no_message", "b1_target_object_mlp"}:
                source_h = self._gather(node_h, edge_src)
                target_h = self._gather(node_h, edge_tgt)
                edge_h = self.edge_encoder(torch.cat([source_h, target_h, action_h.unsqueeze(1).expand_as(source_h)], dim=-1))
            elif self.model_id == "b1_pair_feature_mlp_v2":
                source_h = self._gather(node_h, edge_src)
                target_h = self._gather(node_h, edge_tgt)
                action_edge = action_h.unsqueeze(1).expand_as(source_h)
                edge_h = self.edge_encoder(
                    torch.cat([source_h, target_h, action_edge, edge_geometry], dim=-1)
                )
            elif self.model_id in PAIR_LOCAL_MODEL_IDS:
                source_h = self._gather(node_h, edge_src)
                target_h = self._gather(node_h, edge_tgt)
                pair_parts = [
                    source_h,
                    target_h,
                    edge_geometry[..., : self.base_edge_dim],
                ]
                if self.uses_history:
                    pair_parts.append(edge_geometry[..., self.base_edge_dim :])
                pair_h = self.pair_encoder(torch.cat(pair_parts, dim=-1))
                if self.model_id in PAIR_LOCAL_ACTION_MODEL_IDS:
                    if action_h is None:
                        raise ValueError(f"{self.model_id} requires action")
                    action_edge = action_h.unsqueeze(1).expand_as(pair_h)
                    film_input = torch.cat(
                        [pair_h, action_edge, pair_h * action_edge], dim=-1
                    )
                    gamma, beta, gate_logit = torch.split(
                        self.action_film(film_input),
                        [self.hidden_dim, self.hidden_dim, 1],
                        dim=-1,
                    )
                    pair_h = torch.sigmoid(gate_logit) * (
                        pair_h * (1.0 + torch.tanh(gamma)) + beta
                    )
                edge_h = self.edge_encoder(pair_h)
            elif self.model_id in V2_MESSAGE_MODEL_IDS:
                aggregate = self._aggregate_v2_messages(
                    node_h, edge_src, edge_tgt, edge_geometry, edge_mask, action_h
                )
                updated = self.node_update(torch.cat([node_h, aggregate], dim=-1))
                node_h = self.node_norm(node_h + updated) * node_mask.unsqueeze(-1)
                source_h = self._gather(node_h, edge_src)
                target_h = self._gather(node_h, edge_tgt)
                future_edge_parts = [source_h, target_h, edge_geometry]
                if action_h is not None:
                    future_edge_parts.append(
                        action_h.unsqueeze(1).expand(-1, edge_src.shape[1], -1)
                    )
                edge_h = self.edge_encoder(torch.cat(future_edge_parts, dim=-1))
            else:
                message_steps = 1 if self.model_id in {
                    "g1_sparse_holder_object_gnn",
                    "g1_constant_action_holder_object_gnn_v2",
                    "g3_action_conditioned_holder_object_gnn",
                    "s0_g1_no_action_holder_object_gnn_v2",
                } else 2
                for _ in range(message_steps):
                    aggregate = self._aggregate_messages(
                        node_h,
                        edge_src,
                        edge_tgt,
                        edge_geometry,
                        edge_mask,
                        action_h=action_h if self.model_id == "g3_action_conditioned_holder_object_gnn" else None,
                    )
                    updated = self.node_update(torch.cat([node_h, aggregate], dim=-1))
                    if self.model_id == "g3_action_conditioned_holder_object_gnn":
                        node_h = self.node_norm(node_h + updated)
                    else:
                        node_h = updated
                    node_h = node_h * node_mask.unsqueeze(-1)
                source_h = self._gather(node_h, edge_src)
                target_h = self._gather(node_h, edge_tgt)
                edge_parts = [source_h, target_h, edge_geometry]
                if action_h is not None:
                    edge_parts.append(
                        action_h.unsqueeze(1).expand(-1, edge_src.shape[1], -1)
                    )
                edge_h = self.edge_encoder(torch.cat(edge_parts, dim=-1))
        current_representation = (
            current_edge_h
            if current_edge_h is not None
            else edge_h
        )
        return self.future_head(edge_h), self.current_head(current_representation)


def _masked_bce(logits, labels, valid, pos_weight):
    import torch
    import torch.nn.functional as F

    loss = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
        pos_weight=pos_weight.view(1, 1, -1),
    )
    weighted = loss * valid
    return weighted.sum() / valid.sum().clamp_min(1.0)


def _f1_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    relations: Sequence[str] = EDGE_RELATIONS,
) -> dict[str, Any]:
    per_relation: dict[str, Any] = {}
    scores: list[float] = []
    for index, relation in enumerate(relations):
        selected = mask[..., index].astype(bool)
        truth = y_true[..., index][selected].astype(bool)
        prediction = y_pred[..., index][selected].astype(bool)
        support = int(selected.sum())
        if support == 0:
            per_relation[relation] = {"support": 0, "f1": None, "precision": None, "recall": None}
            continue
        tp = int(np.logical_and(truth, prediction).sum())
        fp = int(np.logical_and(~truth, prediction).sum())
        fn = int(np.logical_and(truth, ~prediction).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        per_relation[relation] = {
            "support": support,
            "positive": int(truth.sum()),
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
        scores.append(f1)
    return {
        "macro_f1": float(np.mean(scores)) if scores else None,
        "per_relation": per_relation,
    }


def _to_device(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _episode_disjoint_action_permutation(
    records: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Return a deterministic, marginal-preserving donor permutation.

    Records are first grouped by episode and then cyclically shifted by the
    largest episode-group size.  When no episode owns more than half of the
    split, this is a bijection in which every recipient receives an action
    from a different episode.  Unlike the legacy within-batch roll, the
    control cannot silently pair neighboring frames from the same demo.
    """

    count = len(records)
    if count < 2:
        raise ValueError("episode-disjoint action shuffle requires at least 2 records")
    episodes = np.asarray([str(record.get("episode_id")) for record in records])
    order = np.argsort(episodes, kind="stable")
    _, episode_counts = np.unique(episodes, return_counts=True)
    shift = int(episode_counts.max())
    if shift * 2 > count:
        raise ValueError(
            "episode-disjoint action shuffle is impossible because one episode "
            f"owns {shift}/{count} records"
        )
    donors_in_sorted_order = np.roll(order, shift=shift)
    donor_by_recipient = np.empty(count, dtype=np.int64)
    donor_by_recipient[order] = donors_in_sorted_order
    if len(np.unique(donor_by_recipient)) != count:
        raise AssertionError("global action donor mapping is not a permutation")
    if np.any(episodes == episodes[donor_by_recipient]):
        raise AssertionError("global action donor mapping retained a same-episode pair")
    return donor_by_recipient


def _action_matching_descriptor(
    record: Mapping[str, Any],
    relations: Sequence[str],
) -> np.ndarray:
    actions = np.asarray(record.get("actions", []), dtype=np.float64)
    action_steps = int(record.get("action_steps", 0))
    action_dim = int(record.get("action_dim", 0))
    if action_steps > 0 and action_dim > 0 and actions.size == action_steps * action_dim:
        steps = actions.reshape(action_steps, action_dim)
        gripper = steps[:, -1]
        arm = steps[:, :-1]
        arm_norm = float(np.linalg.norm(arm))
        gripper_mean = float(gripper.mean())
        gripper_delta = float(gripper[-1] - gripper[0])
    else:
        arm_norm = float(np.linalg.norm(actions))
        gripper_mean = 0.0
        gripper_delta = 0.0
    relation_index = {relation: index for index, relation in enumerate(relations)}
    labels = np.asarray(record.get("current_labels", []), dtype=np.float64)
    valid = np.asarray(record.get("current_valid", []), dtype=np.float64)

    def any_current(relation: str) -> float:
        index = relation_index.get(relation)
        if index is None or labels.size == 0 or valid.size == 0:
            return 0.0
        return float(np.any((valid[..., index] > 0) & (labels[..., index] >= 0.5)))

    return np.asarray(
        [
            float(np.linalg.norm(actions)),
            arm_norm,
            gripper_mean,
            gripper_delta,
            any_current("contact"),
            any_current("holding"),
        ],
        dtype=np.float64,
    )


def _matched_episode_disjoint_action_permutation(
    records: Sequence[Mapping[str, Any]],
    relations: Sequence[str] = EDGE_RELATIONS,
    optimization_passes: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Match shuffled training actions within task and across episodes.

    The mapping is a task-local bijection.  It starts from a guaranteed
    episode-disjoint cyclic assignment and then swaps donor assignments when
    doing so reduces action-magnitude/current-state descriptor distance while
    preserving task identity and episode disjointness.
    """

    count = len(records)
    if count < 2:
        raise ValueError("matched action shuffle requires at least 2 records")
    tasks = np.asarray(
        [f"{record.get('suite')}:{record.get('task_id')}" for record in records]
    )
    episodes = np.asarray([str(record.get("episode_id")) for record in records])
    descriptors = np.stack(
        [_action_matching_descriptor(record, relations) for record in records]
    )
    donor_by_recipient = np.full(count, -1, dtype=np.int64)
    task_reports: dict[str, Any] = {}

    for task in sorted(set(tasks.tolist())):
        task_indices = np.flatnonzero(tasks == task)
        task_episodes = episodes[task_indices]
        order_local = np.argsort(task_episodes, kind="stable")
        ordered_indices = task_indices[order_local]
        _, episode_counts = np.unique(task_episodes, return_counts=True)
        shift = int(episode_counts.max())
        if shift * 2 > len(task_indices):
            raise ValueError(
                f"task {task} cannot form an episode-disjoint bijection: "
                f"largest episode owns {shift}/{len(task_indices)} records"
            )
        initial_donors = np.roll(ordered_indices, shift=shift)
        donor_by_recipient[ordered_indices] = initial_donors

        task_descriptor = descriptors[task_indices].copy()
        mean = task_descriptor[:, :4].mean(axis=0)
        std = task_descriptor[:, :4].std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        normalized = descriptors.copy()
        normalized[task_indices, :4] = (
            task_descriptor[:, :4] - mean
        ) / std

        def cost(recipient: int, donor: int) -> float:
            continuous = float(
                np.mean(
                    (
                        normalized[recipient, :4]
                        - normalized[donor, :4]
                    )
                    ** 2
                )
            )
            state_penalty = float(
                np.abs(
                    descriptors[recipient, 4:]
                    - descriptors[donor, 4:]
                ).sum()
            )
            return continuous + 0.25 * state_penalty

        initial_mean_cost = float(
            np.mean(
                [cost(index, donor_by_recipient[index]) for index in task_indices]
            )
        )
        swaps = 0
        for _ in range(max(int(optimization_passes), 0)):
            changed = False
            for left_position, left in enumerate(task_indices):
                for right in task_indices[left_position + 1 :]:
                    left_donor = int(donor_by_recipient[left])
                    right_donor = int(donor_by_recipient[right])
                    if episodes[left] == episodes[right_donor]:
                        continue
                    if episodes[right] == episodes[left_donor]:
                        continue
                    before = cost(left, left_donor) + cost(right, right_donor)
                    after = cost(left, right_donor) + cost(right, left_donor)
                    if after + 1e-12 < before:
                        donor_by_recipient[left] = right_donor
                        donor_by_recipient[right] = left_donor
                        swaps += 1
                        changed = True
            if not changed:
                break
        final_mean_cost = float(
            np.mean(
                [cost(index, donor_by_recipient[index]) for index in task_indices]
            )
        )
        task_reports[task] = {
            "records": int(len(task_indices)),
            "episodes": int(len(set(task_episodes.tolist()))),
            "initial_mean_matching_cost": initial_mean_cost,
            "final_mean_matching_cost": final_mean_cost,
            "improving_swaps": swaps,
        }

    if np.any(donor_by_recipient < 0):
        raise AssertionError("matched action donor mapping is incomplete")
    if len(np.unique(donor_by_recipient)) != count:
        raise AssertionError("matched action donor mapping is not a bijection")
    if np.any(tasks != tasks[donor_by_recipient]):
        raise AssertionError("matched action donor crossed task boundaries")
    if np.any(episodes == episodes[donor_by_recipient]):
        raise AssertionError("matched action donor retained a same-episode pair")
    return donor_by_recipient, {
        "method": "task_local_episode_disjoint_bijection_with_descriptor_swap_optimization",
        "descriptor": [
            "action_l2_norm",
            "arm_l2_norm",
            "gripper_mean",
            "gripper_delta",
            "current_contact_any",
            "current_holding_any",
        ],
        "tasks": task_reports,
        "same_task_fraction": 1.0,
        "different_episode_fraction": 1.0,
        "marginal_preserved": True,
    }


def attach_training_action_donors(
    records: Sequence[Mapping[str, Any]],
    relations: Sequence[str] = EDGE_RELATIONS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    donors, qa = _matched_episode_disjoint_action_permutation(records, relations)
    decorated: list[dict[str, Any]] = []
    for recipient_index, record in enumerate(records):
        donor_index = int(donors[recipient_index])
        donor = records[donor_index]
        output = dict(record)
        output["training_donor_actions"] = list(donor.get("actions", []))
        output["training_donor_action_step_mask"] = list(
            donor.get("action_step_mask", [])
        )
        output["training_action_donor_sample_id"] = str(donor.get("sample_id"))
        output["training_action_donor_episode_id"] = str(donor.get("episode_id"))
        output["training_action_donor_task_id"] = donor.get("task_id")
        decorated.append(output)
    return decorated, qa


def _collect_predictions(model, loader, device: str, mode: str) -> dict[str, Any]:
    import torch
    from torch.utils.data import SequentialSampler

    y_true: list[np.ndarray] = []
    y_score: list[np.ndarray] = []
    current_score: list[np.ndarray] = []
    future_valid: list[np.ndarray] = []
    current_true: list[np.ndarray] = []
    current_valid: list[np.ndarray] = []
    changed: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    model.eval()
    global_actions = None
    global_action_step_mask = None
    global_donors = None
    global_edge_geometry = None
    batches = loader
    if mode in {"global_shuffled_action", "global_shuffled_history"}:
        if not isinstance(loader.sampler, SequentialSampler):
            raise ValueError(f"{mode} requires an unshuffled loader")
        batches = list(loader)
        if mode == "global_shuffled_action":
            global_actions = torch.cat([batch["actions"] for batch in batches], dim=0)
            step_masks = [batch.get("action_step_mask") for batch in batches]
            if all(mask is not None for mask in step_masks):
                global_action_step_mask = torch.cat(step_masks, dim=0)
        else:
            if int(getattr(model.shape, "history_dim", 0)) <= 0:
                raise ValueError("global_shuffled_history requires history_dim > 0")
            global_edge_geometry = torch.cat(
                [batch["edge_geometry"] for batch in batches], dim=0
            )
        global_donors = _episode_disjoint_action_permutation(list(loader.dataset))
        bank_size = (
            global_actions.shape[0]
            if global_actions is not None
            else global_edge_geometry.shape[0]
        )
        if len(global_donors) != bank_size:
            raise ValueError(f"global {mode} bank and dataset lengths differ")
    global_offset = 0
    with torch.no_grad():
        for batch in batches:
            batch = _to_device(batch, device)
            actions = batch["actions"]
            edge_geometry = batch["edge_geometry"]
            action_step_mask = batch.get("action_step_mask")
            if mode == "global_shuffled_action":
                batch_size = int(actions.shape[0])
                donor_indices = torch.as_tensor(
                    global_donors[global_offset : global_offset + batch_size],
                    dtype=torch.long,
                )
                actions = global_actions[donor_indices].to(device)
                if global_action_step_mask is not None:
                    action_step_mask = global_action_step_mask[donor_indices].to(device)
                global_offset += batch_size
            elif mode == "global_shuffled_history":
                batch_size = int(actions.shape[0])
                donor_indices = torch.as_tensor(
                    global_donors[global_offset : global_offset + batch_size],
                    dtype=torch.long,
                )
                edge_geometry = edge_geometry.clone()
                history_dim = int(model.shape.history_dim)
                edge_geometry[..., -history_dim:] = global_edge_geometry[
                    donor_indices, ..., -history_dim:
                ].to(device)
                global_offset += batch_size
            elif mode == "no_history":
                history_dim = int(getattr(model.shape, "history_dim", 0))
                if history_dim <= 0:
                    raise ValueError("no_history requires history_dim > 0")
                edge_geometry = edge_geometry.clone()
                # Edge inputs are already train-normalized, so zero is the
                # training-mean causal-history vector.
                edge_geometry[..., -history_dim:] = 0.0
            elif mode == "no_action":
                # Legacy control: zero in normalized space is the train-mean
                # action, not a physical zero command. Kept for report
                # compatibility; use physical_zero_action for the latter.
                actions = torch.zeros_like(actions)
            elif mode == "physical_zero_action":
                actions = batch["zero_actions"]
            elif mode == "shuffled_action" and actions.shape[0] > 1:
                actions = torch.roll(actions, shifts=1, dims=0)
                if action_step_mask is not None:
                    action_step_mask = torch.roll(action_step_mask, shifts=1, dims=0)
            elif mode in {"reversed_action", "shuffled_gripper", "shuffled_arm"}:
                shape = model.shape
                if shape.action_steps * shape.action_step_dim != actions.shape[1]:
                    raise ValueError(f"{mode} requires a fixed structured action layout")
                steps = actions.reshape(
                    actions.shape[0], shape.action_steps, shape.action_step_dim
                ).clone()
                if mode == "reversed_action":
                    steps = torch.flip(steps, dims=(1,))
                    if action_step_mask is not None:
                        action_step_mask = torch.flip(action_step_mask, dims=(1,))
                elif actions.shape[0] > 1 and mode == "shuffled_gripper":
                    steps[..., -1] = torch.roll(steps[..., -1], shifts=1, dims=0)
                elif actions.shape[0] > 1 and mode == "shuffled_arm":
                    steps[..., :-1] = torch.roll(steps[..., :-1], shifts=1, dims=0)
                actions = steps.reshape_as(actions)
            future_logits, current_logits = model(
                batch["node_x"], batch["node_mask"], batch["edge_src"], batch["edge_tgt"],
                edge_geometry, batch["edge_mask"], actions,
                edge_shuffle=mode == "shuffled_edge",
                action_step_mask=action_step_mask,
            )
            y_true.append(batch["future_labels"].cpu().numpy())
            y_score.append(torch.sigmoid(future_logits).cpu().numpy())
            current_score.append(torch.sigmoid(current_logits).cpu().numpy())
            future_valid.append(batch["future_valid"].cpu().numpy())
            current_true.append(batch["current_labels"].cpu().numpy())
            current_valid.append(batch["current_valid"].cpu().numpy())
            changed.append(batch["changed"].cpu().numpy())
            for batch_index, sample_id_value in enumerate(batch["sample_id"]):
                metadata.append(
                    {
                        "sample_id": str(sample_id_value),
                        "episode_id": str(batch["episode_id"][batch_index]),
                        "suite": batch["suite"][batch_index],
                        "task_id": batch["task_id"][batch_index],
                        "start_step": batch["start_step"][batch_index],
                        "target_step": batch["target_step"][batch_index],
                        "tau": batch["tau"][batch_index],
                        "event_id": batch["event_id"][batch_index],
                        "edge_keys": batch["edge_keys"][batch_index],
                    }
                )
    return {
        "future_true": np.concatenate(y_true),
        "future_score": np.concatenate(y_score),
        "current_score": np.concatenate(current_score),
        "future_valid": np.concatenate(future_valid),
        "current_true": np.concatenate(current_true),
        "current_valid": np.concatenate(current_valid),
        "changed": np.concatenate(changed),
        "metadata": metadata,
    }


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    truth = y_true.astype(bool).reshape(-1)
    score = y_score.astype(np.float64).reshape(-1)
    positive = int(truth.sum())
    if positive == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    ordered_truth = truth[order]
    true_positive = np.cumsum(ordered_truth)
    rank = np.arange(1, len(ordered_truth) + 1, dtype=np.float64)
    precision = true_positive / rank
    return float(precision[ordered_truth].sum() / positive)


def _brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    truth = y_true.astype(np.float64).reshape(-1)
    score = np.clip(y_score.astype(np.float64).reshape(-1), 0.0, 1.0)
    return float(np.mean((score - truth) ** 2)) if truth.size else None


def _expected_calibration_error(
    y_true: np.ndarray,
    y_score: np.ndarray,
    bins: int = 10,
) -> float | None:
    truth = y_true.astype(np.float64).reshape(-1)
    score = np.clip(y_score.astype(np.float64).reshape(-1), 0.0, 1.0)
    if not truth.size:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_index = np.minimum(np.digitize(score, edges[1:-1], right=False), bins - 1)
    error = 0.0
    for index in range(bins):
        selected = bin_index == index
        if not np.any(selected):
            continue
        error += float(selected.mean()) * abs(
            float(score[selected].mean()) - float(truth[selected].mean())
        )
    return float(error)


def _binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    selected = mask.astype(bool)
    truth = y_true[selected].astype(bool)
    prediction = y_pred[selected].astype(bool)
    score = y_score[selected].astype(np.float64)
    support = int(selected.sum())
    if support == 0:
        return {
            "support": 0,
            "positive": 0,
            "f1": None,
            "precision": None,
            "recall": None,
            "pr_auc": None,
            "brier_score": None,
            "ece": None,
            "ece_bins": 10,
        }
    true_positive = int(np.logical_and(truth, prediction).sum())
    false_positive = int(np.logical_and(~truth, prediction).sum())
    false_negative = int(np.logical_and(truth, ~prediction).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "support": support,
        "positive": int(truth.sum()),
        "predicted_positive": int(prediction.sum()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "pr_auc": _average_precision(truth, score),
        "pr_auc_definition": "average_precision",
        "brier_score": _brier_score(truth, score),
        "ece": _expected_calibration_error(truth, score, bins=10),
        "ece_bins": 10,
    }


def _holding_metrics(
    outputs: Mapping[str, Any],
    relations: Sequence[str],
    threshold: float,
    current_threshold: float = 0.5,
) -> dict[str, Any] | None:
    if "holding" not in relations:
        return None
    holding_index = list(relations).index("holding")
    current = outputs["current_true"][..., holding_index] >= 0.5
    future = outputs["future_true"][..., holding_index] >= 0.5
    score = outputs["future_score"][..., holding_index]
    prediction = score >= float(threshold)
    current_score = outputs.get("current_score", outputs["current_true"])[
        ..., holding_index
    ]
    current_prediction = current_score >= float(current_threshold)
    valid = np.logical_and(
        outputs["current_valid"][..., holding_index] > 0,
        outputs["future_valid"][..., holding_index] > 0,
    )

    change_truth = np.logical_xor(current, future)
    change_prediction = np.logical_xor(current, prediction)
    change_score = np.where(current, 1.0 - score, score)
    end_to_end_change_prediction = np.logical_xor(current_prediction, prediction)
    end_to_end_change_score = (
        current_score * (1.0 - score) + (1.0 - current_score) * score
    )
    onset_mask = np.logical_and(valid, ~current)
    release_mask = np.logical_and(valid, current)
    hard_negative_mask = np.zeros_like(valid, dtype=bool)
    if "contact" in relations:
        contact_index = list(relations).index("contact")
        current_contact = np.logical_and(
            outputs["current_valid"][..., contact_index] > 0,
            outputs["current_true"][..., contact_index] >= 0.5,
        )
        future_contact = np.logical_and(
            outputs["future_valid"][..., contact_index] > 0,
            outputs["future_true"][..., contact_index] >= 0.5,
        )
        hard_negative_mask = np.logical_and.reduce(
            [valid, ~current, ~future, np.logical_or(current_contact, future_contact)]
        )
    hard_negative_support = int(hard_negative_mask.sum())
    hard_negative_fp = int(np.logical_and(prediction, hard_negative_mask).sum())
    conditional_change_metrics = _binary_metrics(
        change_truth, change_prediction, change_score, valid
    )
    return {
        "threshold": float(threshold),
        "current_threshold": float(current_threshold),
        "current_state": _binary_metrics(
            current, current_prediction, current_score, valid
        ),
        "future_state": _binary_metrics(future, prediction, score, valid),
        # Compatibility alias retained for v1 result readers.  This metric is
        # conditional on an oracle symbolic current holding state.
        "change_event": conditional_change_metrics,
        "conditional_oracle_current_change_event": conditional_change_metrics,
        "end_to_end_change_event": _binary_metrics(
            change_truth,
            end_to_end_change_prediction,
            end_to_end_change_score,
            valid,
        ),
        "onset": _binary_metrics(future, prediction, score, onset_mask),
        "release": _binary_metrics(
            ~future, ~prediction, 1.0 - score, release_mask
        ),
        "future_value_on_true_changed": _binary_metrics(
            future, prediction, score, np.logical_and(valid, change_truth)
        ),
        "hard_negative": {
            "support": hard_negative_support,
            "false_positive": hard_negative_fp,
            "false_positive_rate": (
                hard_negative_fp / hard_negative_support
                if hard_negative_support
                else None
            ),
            "definition": (
                "current_or_future_contact_and_current_future_holding_false"
            ),
        },
        "metric_contract": {
            "change_event": "compatibility alias for conditional_oracle_current_change_event",
            "conditional_oracle_current_change_event": (
                "ground-truth current holding XOR predicted future holding"
            ),
            "end_to_end_change_event": (
                "predicted current holding XOR predicted future holding"
            ),
        },
    }


def _select_holding_threshold(
    outputs: Mapping[str, Any],
    relations: Sequence[str],
    current_threshold: float = 0.5,
) -> tuple[float, dict[str, Any] | None]:
    if "holding" not in relations:
        return 0.5, None
    best_threshold = 0.5
    best_metrics = _holding_metrics(
        outputs, relations, best_threshold, current_threshold=current_threshold
    )
    best_key = (-1.0, -1.0, -1.0)
    for threshold in np.linspace(0.05, 0.95, 91):
        metrics = _holding_metrics(
            outputs,
            relations,
            float(threshold),
            current_threshold=current_threshold,
        )
        if metrics is None:
            continue
        change = metrics["change_event"]
        future = metrics["future_state"]
        if not change["positive"]:
            primary = future["f1"] or 0.0
        else:
            primary = change["f1"] or 0.0
        key = (
            primary,
            future["f1"] or 0.0,
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def _select_current_holding_threshold(
    outputs: Mapping[str, Any],
    relations: Sequence[str],
) -> float:
    if "holding" not in relations:
        return 0.5
    holding_index = list(relations).index("holding")
    current = outputs["current_true"][..., holding_index] >= 0.5
    score = outputs.get("current_score", outputs["current_true"])[
        ..., holding_index
    ]
    valid = outputs["current_valid"][..., holding_index] > 0
    best_threshold = 0.5
    best_key = (-1.0, -1.0)
    for threshold in np.linspace(0.05, 0.95, 91):
        metrics = _binary_metrics(current, score >= threshold, score, valid)
        key = (metrics["f1"] or 0.0, -abs(float(threshold) - 0.5))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _holding_prediction_rows(
    outputs: Mapping[str, Any],
    relations: Sequence[str],
    holding_threshold: float,
    current_holding_threshold: float,
    *,
    mode: str,
    view: str | None,
) -> list[dict[str, Any]]:
    if "holding" not in relations:
        return []
    holding_index = list(relations).index("holding")
    contact_index = list(relations).index("contact") if "contact" in relations else None
    rows: list[dict[str, Any]] = []
    for sample_index, metadata in enumerate(outputs.get("metadata", [])):
        edge_keys = list(metadata.get("edge_keys", []))
        for edge_index, key in enumerate(edge_keys):
            current_valid = bool(
                outputs["current_valid"][sample_index, edge_index, holding_index] > 0
            )
            future_valid = bool(
                outputs["future_valid"][sample_index, edge_index, holding_index] > 0
            )
            if not (current_valid and future_valid):
                continue
            current_target = bool(
                outputs["current_true"][sample_index, edge_index, holding_index] >= 0.5
            )
            future_target = bool(
                outputs["future_true"][sample_index, edge_index, holding_index] >= 0.5
            )
            current_probability = float(
                outputs["current_score"][sample_index, edge_index, holding_index]
            )
            future_probability = float(
                outputs["future_score"][sample_index, edge_index, holding_index]
            )
            current_prediction = current_probability >= current_holding_threshold
            future_prediction = future_probability >= holding_threshold
            current_contact = False
            future_contact = False
            if contact_index is not None:
                current_contact = bool(
                    outputs["current_valid"][sample_index, edge_index, contact_index] > 0
                    and outputs["current_true"][sample_index, edge_index, contact_index] >= 0.5
                )
                future_contact = bool(
                    outputs["future_valid"][sample_index, edge_index, contact_index] > 0
                    and outputs["future_true"][sample_index, edge_index, contact_index] >= 0.5
                )
            event_id = metadata.get("event_id")
            rows.append(
                {
                    "sample_id": str(metadata["sample_id"]),
                    "episode_id": str(metadata["episode_id"]),
                    "task_id": metadata.get("task_id"),
                    "suite": metadata.get("suite"),
                    "start_step": metadata.get("start_step"),
                    "target_step": metadata.get("target_step"),
                    "tau": metadata.get("tau"),
                    "event_cluster_id": (
                        str(event_id) if event_id is not None else str(metadata["sample_id"])
                    ),
                    "event_cluster_source": (
                        "explicit_event_id" if event_id is not None else "sample_id_proxy"
                    ),
                    "edge_source": str(key[0]),
                    "edge_target": str(key[1]),
                    "view": view,
                    "mode": mode,
                    "current_target": int(current_target),
                    "future_target": int(future_target),
                    "current_probability": current_probability,
                    "future_probability": future_probability,
                    "current_threshold": float(current_holding_threshold),
                    "future_threshold": float(holding_threshold),
                    "current_prediction": int(current_prediction),
                    "future_prediction": int(future_prediction),
                    "change_target": int(current_target != future_target),
                    "conditional_oracle_current_change_probability": float(
                        1.0 - future_probability
                        if current_target
                        else future_probability
                    ),
                    "conditional_oracle_current_change_prediction": int(
                        current_target != future_prediction
                    ),
                    "end_to_end_change_probability": float(
                        current_probability * (1.0 - future_probability)
                        + (1.0 - current_probability) * future_probability
                    ),
                    "end_to_end_change_prediction": int(
                        current_prediction != future_prediction
                    ),
                    "onset_target": int((not current_target) and future_target),
                    "release_target": int(current_target and (not future_target)),
                    "hard_negative": int(
                        (not current_target)
                        and (not future_target)
                        and (current_contact or future_contact)
                    ),
                }
            )
    return rows


def evaluate_model(
    model,
    loader,
    device: str,
    mode: str = "correct",
    relations: Sequence[str] = EDGE_RELATIONS,
    holding_threshold: float = 0.5,
    current_holding_threshold: float = 0.5,
    calibrate_holding_threshold: bool = False,
    include_predictions: bool = False,
    prediction_view: str | None = None,
) -> dict[str, Any]:
    outputs = _collect_predictions(model, loader, device, mode)
    calibrated_on_this_split = bool(calibrate_holding_threshold and mode == "correct")
    if calibrated_on_this_split:
        current_holding_threshold = _select_current_holding_threshold(
            outputs, relations
        )
        holding_threshold, holding = _select_holding_threshold(
            outputs,
            relations,
            current_threshold=current_holding_threshold,
        )
    else:
        holding = _holding_metrics(
            outputs,
            relations,
            holding_threshold,
            current_threshold=current_holding_threshold,
        )
    thresholds = np.full((len(relations),), 0.5, dtype=np.float32)
    if "holding" in relations:
        thresholds[list(relations).index("holding")] = float(holding_threshold)
    prediction = outputs["future_score"] >= thresholds.reshape(1, 1, -1)
    overall = _f1_metrics(
        outputs["future_true"], prediction, outputs["future_valid"], relations
    )
    changed_metrics = _f1_metrics(
        outputs["future_true"],
        prediction,
        outputs["future_valid"] * outputs["changed"],
        relations,
    )
    result = {
        "mode": mode,
        "future_relation": overall,
        "changed_relation": changed_metrics,
        "holding": holding,
        "holding_threshold": float(holding_threshold),
        "current_holding_threshold": float(current_holding_threshold),
        "threshold_calibrated_on_this_split": calibrated_on_this_split,
    }
    if include_predictions:
        result["prediction_rows"] = _holding_prediction_rows(
            outputs,
            relations,
            holding_threshold,
            current_holding_threshold,
            mode=mode,
            view=prediction_view,
        )
    return result


def _positive_weight(
    records: Sequence[Mapping[str, Any]],
    relation_dim: int,
    label_key: str = "future_labels",
    valid_key: str = "future_valid",
):
    import torch

    positive = np.zeros(relation_dim, dtype=np.float64)
    negative = np.zeros(relation_dim, dtype=np.float64)
    for record in records:
        labels = np.asarray(record[label_key], dtype=np.float32)
        valid = np.asarray(record[valid_key], dtype=np.float32)
        positive += (labels * valid).sum(axis=0)
        negative += ((1.0 - labels) * valid).sum(axis=0)
    return torch.tensor(np.clip(negative / np.maximum(positive, 1.0), 1.0, 20.0), dtype=torch.float32)


def _checkpoint_selection(
    metrics: Mapping[str, Any],
    criterion: str = "legacy_holding_event_f1",
) -> tuple[float, str]:
    holding = metrics.get("holding")
    if criterion == "holding_event_pr_auc":
        if isinstance(holding, Mapping):
            change = holding.get("conditional_oracle_current_change_event")
            if not isinstance(change, Mapping):
                change = holding.get("change_event")
            if isinstance(change, Mapping) and change.get("pr_auc") is not None:
                return float(change["pr_auc"]), "holding_event_pr_auc"
        return -1.0, "holding_event_pr_auc_unsupported"
    if criterion != "legacy_holding_event_f1":
        raise ValueError(f"unknown checkpoint criterion: {criterion}")
    if isinstance(holding, Mapping):
        change = holding.get("change_event")
        if isinstance(change, Mapping) and int(change.get("positive", 0)) > 0:
            return float(change.get("f1") or 0.0), "holding_change_event_f1"
        future = holding.get("future_state")
        if isinstance(future, Mapping) and int(future.get("positive", 0)) > 0:
            return float(future.get("f1") or 0.0), "holding_future_state_f1_fallback"
    future_relation = metrics.get("future_relation", {})
    return float(future_relation.get("macro_f1") or 0.0), "future_relation_macro_f1_fallback"


def _training_action_inputs(batch: Mapping[str, Any], mode: str):
    """Return action tensors for one training step under a controlled mode.

    ``shuffled_batch`` is the legacy within-batch cyclic shift.
    ``episode_disjoint_matched`` consumes donor tensors prepared by
    :func:`attach_training_action_donors`; these preserve the task-local action
    marginal while forcing a different-episode donor and matching coarse
    action magnitude/current state.
    """

    if mode not in {"correct", "shuffled_batch", "episode_disjoint_matched"}:
        raise ValueError(f"unknown training_action_mode={mode}")
    actions = batch["actions"]
    action_step_mask = batch.get("action_step_mask")
    shuffled = False
    if mode == "episode_disjoint_matched":
        actions = batch["training_donor_actions"]
        action_step_mask = batch.get("training_donor_action_step_mask")
        shuffled = True
    elif mode == "shuffled_batch" and actions.shape[0] > 1:
        actions = torch.roll(actions, shifts=1, dims=0)
        if action_step_mask is not None:
            action_step_mask = torch.roll(action_step_mask, shifts=1, dims=0)
        shuffled = True
    return actions, action_step_mask, shuffled


def train_one(
    model_id: str,
    shape: ProbeShape,
    train_records: Sequence[Mapping[str, Any]],
    val_loader,
    train_loader,
    device: str,
    hidden_dim: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    current_loss_weight: float,
    relations: Sequence[str] = EDGE_RELATIONS,
    training_action_mode: str = "correct",
    current_head_contract: str = "legacy",
    checkpoint_criterion: str = "legacy_holding_event_f1",
) -> tuple[Any, dict[str, Any]]:
    import torch

    model = RelationalDynamicsProbe(
        model_id,
        shape,
        hidden_dim=hidden_dim,
        current_head_contract=current_head_contract,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    future_pos_weight = _positive_weight(train_records, shape.relation_dim).to(device)
    current_pos_weight = (
        _positive_weight(
            train_records,
            shape.relation_dim,
            label_key="current_labels",
            valid_key="current_valid",
        ).to(device)
        if (
            model_id in EXPERIMENTAL_MODEL_IDS
            or current_head_contract == "action_free_pair"
        )
        else future_pos_weight
    )
    best_score = -1.0
    best_state = None
    best_holding_threshold = 0.5
    best_current_holding_threshold = 0.5
    best_val_future_macro_f1 = None
    best_validation_metrics = None
    checkpoint_metric = "uninitialized"
    stale = 0
    history: list[dict[str, Any]] = []
    shuffled_action_batches = 0
    shuffled_action_examples = 0
    unshuffled_singleton_examples = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = _to_device(batch, device)
            train_actions, train_action_step_mask, action_was_shuffled = (
                _training_action_inputs(batch, training_action_mode)
            )
            if action_was_shuffled:
                shuffled_action_batches += 1
                shuffled_action_examples += int(train_actions.shape[0])
            elif training_action_mode == "shuffled_batch":
                unshuffled_singleton_examples += int(train_actions.shape[0])
            optimizer.zero_grad(set_to_none=True)
            future_logits, current_logits = model(
                batch["node_x"], batch["node_mask"], batch["edge_src"], batch["edge_tgt"],
                batch["edge_geometry"], batch["edge_mask"], train_actions,
                action_step_mask=train_action_step_mask,
            )
            future_loss = _masked_bce(
                future_logits,
                batch["future_labels"],
                batch["future_valid"],
                future_pos_weight,
            )
            current_loss = _masked_bce(
                current_logits,
                batch["current_labels"],
                batch["current_valid"],
                current_pos_weight,
            )
            loss = future_loss + current_loss_weight * current_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            mode="correct",
            relations=relations,
            calibrate_holding_threshold=True,
        )
        score, score_name = _checkpoint_selection(
            val_metrics, criterion=checkpoint_criterion
        )
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "checkpoint_score": score,
            "checkpoint_metric": score_name,
            "holding_threshold": val_metrics["holding_threshold"],
            "current_holding_threshold": val_metrics[
                "current_holding_threshold"
            ],
            "val_future_macro_f1": val_metrics["future_relation"]["macro_f1"],
            "val_holding_change_event_f1": (
                val_metrics["holding"]["change_event"]["f1"]
                if val_metrics.get("holding")
                else None
            ),
        })
        if score > best_score + 1e-6:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_holding_threshold = float(val_metrics["holding_threshold"])
            best_current_holding_threshold = float(
                val_metrics["current_holding_threshold"]
            )
            best_val_future_macro_f1 = val_metrics["future_relation"]["macro_f1"]
            best_validation_metrics = copy.deepcopy(val_metrics)
            checkpoint_metric = score_name
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "hidden_dim": hidden_dim,
        "best_checkpoint_score": best_score,
        "checkpoint_metric": checkpoint_metric,
        "best_val_future_macro_f1": best_val_future_macro_f1,
        "best_val_macro_f1": best_val_future_macro_f1,
        "holding_threshold": best_holding_threshold,
        "current_holding_threshold": best_current_holding_threshold,
        "threshold_fit": "validation_correct_action_only",
        "threshold_grid": {"minimum": 0.05, "maximum": 0.95, "step": 0.01},
        "best_validation_metrics": best_validation_metrics,
        "epochs_ran": len(history),
        "history": history,
        "parameter_count": _parameter_count(model),
        "current_head_contract": current_head_contract,
        "current_loss_weight": float(current_loss_weight),
        "checkpoint_criterion_requested": checkpoint_criterion,
        "training_action_mode": training_action_mode,
        "training_action_shuffled_batches": shuffled_action_batches,
        "training_action_shuffled_examples": shuffled_action_examples,
        "training_action_unshuffled_singleton_examples": unshuffled_singleton_examples,
    }


def _loader(records, shape, normalization, batch_size, shuffle):
    import torch
    from torch.utils.data import DataLoader

    collator = ProbeCollator(shape, normalization)
    return DataLoader(records, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


def _parameter_count(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def _select_hidden_dims(
    model_ids: Sequence[str],
    shape: ProbeShape,
    config: Mapping[str, Any],
) -> dict[str, int]:
    """Choose per-model widths closest to one declared parameter budget."""

    requested = int(config.get("hidden_dim", 64))
    if not bool(config.get("parameter_match", False)):
        return {model_id: requested for model_id in model_ids}
    target = int(config.get("target_parameter_count", 70000))
    current_head_contract = str(config.get("current_head_contract", "legacy"))
    candidates = [int(value) for value in config.get(
        "candidate_hidden_dims", [40, 48, 56, 64, 72, 80, 88, 96]
    )]
    selected: dict[str, int] = {}
    for model_id in model_ids:
        choices: list[tuple[int, int]] = []
        for hidden_dim in candidates:
            model = RelationalDynamicsProbe(
                model_id,
                shape,
                hidden_dim=hidden_dim,
                current_head_contract=current_head_contract,
            )
            parameter_count = _parameter_count(model)
            choices.append((abs(parameter_count - target), hidden_dim))
            del model
        selected[model_id] = min(choices)[1]
    return selected


def run_probe(dataset: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    relations = tuple(str(value) for value in config.get("target_relations", EDGE_RELATIONS))
    unknown_relations = sorted(set(relations) - set(EDGE_RELATIONS))
    if not relations or unknown_relations:
        raise ValueError(
            f"target_relations must be a non-empty subset of {EDGE_RELATIONS}; "
            f"unknown={unknown_relations}"
        )
    records = load_probe_records(
        dataset,
        split_config=config.get("split"),
        relations=relations,
        node_feature_contract=str(config.get("node_feature_contract", "legacy_v1")),
        edge_feature_contract=str(config.get("edge_feature_contract", "geometry_v1")),
    )
    train_records = [record for record in records if record["split"] == "train"]
    val_records = [record for record in records if record["split"] == "validation"]
    test_records = [record for record in records if record["split"] == "test"]
    if not train_records or not val_records or not test_records:
        raise ValueError("dataset must contain train, validation, and test episode splits")
    shape = ProbeShape(
        max_nodes=max(len(record["node_features"]) for record in records),
        max_edges=max(len(record["edge_src"]) for record in records),
        node_dim=len(records[0]["node_features"][0]),
        edge_dim=len(records[0]["edge_geometry"][0]),
        action_dim=max(len(record["actions"]) for record in records),
        relation_dim=len(relations),
        action_steps=max(int(record.get("action_steps", 0)) for record in records),
        action_step_dim=max(int(record.get("action_dim", 0)) for record in records),
    )
    action_normalization_contract = str(
        config.get("action_normalization_contract", "flat_position_v1")
    )
    normalization = _normalization(train_records, action_normalization_contract)
    requested_device = str(config.get("device", "auto"))
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    batch_size = int(config.get("batch_size", 64))
    train_loader = _loader(train_records, shape, normalization, batch_size, shuffle=True)
    val_loader = _loader(val_records, shape, normalization, batch_size, shuffle=False)
    test_loader = _loader(test_records, shape, normalization, batch_size, shuffle=False)
    model_ids = list(config.get("models", MODEL_IDS))
    seeds = [int(seed) for seed in config.get("seeds", [0, 1, 2])]
    hidden_dims = _select_hidden_dims(model_ids, shape, config)
    current_head_contract = str(config.get("current_head_contract", "legacy"))
    selected_parameter_counts = {
        model_id: _parameter_count(
            RelationalDynamicsProbe(
                model_id,
                shape,
                hidden_dim=hidden_dims[model_id],
                current_head_contract=current_head_contract,
            )
        )
        for model_id in model_ids
    }
    results: list[dict[str, Any]] = []
    for model_id in model_ids:
        for seed in seeds:
            _set_seed(seed)
            model, training = train_one(
                model_id=model_id,
                shape=shape,
                train_records=train_records,
                val_loader=val_loader,
                train_loader=train_loader,
                device=device,
                hidden_dim=hidden_dims[model_id],
                epochs=int(config.get("epochs", 40)),
                patience=int(config.get("patience", 8)),
                learning_rate=float(config.get("learning_rate", 1e-3)),
                current_loss_weight=float(config.get("current_loss_weight", 0.25)),
                relations=relations,
                current_head_contract=current_head_contract,
                checkpoint_criterion=str(
                    config.get(
                        "checkpoint_criterion", "legacy_holding_event_f1"
                    )
                ),
            )
            modes = ["correct", "no_action", "shuffled_action", "shuffled_edge"]
            if action_normalization_contract == "channel_v2":
                modes.extend([
                    "physical_zero_action",
                    "reversed_action",
                    "shuffled_gripper",
                    "shuffled_arm",
                ])
            evaluations = {
                mode: evaluate_model(
                    model,
                    test_loader,
                    device,
                    mode=mode,
                    relations=relations,
                    holding_threshold=training["holding_threshold"],
                    current_holding_threshold=training[
                        "current_holding_threshold"
                    ],
                )
                for mode in modes
            }
            results.append({
                "model_id": model_id,
                "seed": seed,
                "hidden_dim": hidden_dims[model_id],
                "device": device,
                "training": training,
                "evaluations": evaluations,
            })
            print(json.dumps({
                "model_id": model_id,
                "seed": seed,
                "best_val_macro_f1": training["best_val_macro_f1"],
                "checkpoint_metric": training["checkpoint_metric"],
                "best_checkpoint_score": training["best_checkpoint_score"],
                "holding_threshold": training["holding_threshold"],
                "test_macro_f1": evaluations["correct"]["future_relation"]["macro_f1"],
                "changed_f1": evaluations["correct"]["changed_relation"]["macro_f1"],
                "holding_change_event_f1": (
                    evaluations["correct"]["holding"]["change_event"]["f1"]
                    if evaluations["correct"].get("holding")
                    else None
                ),
            }, ensure_ascii=False))
    return {
        "probe_version": "phase3-offline.v2" if (
            config.get("node_feature_contract") == "holder_object_v2"
            or config.get("edge_feature_contract") == "holder_object_v2"
            or action_normalization_contract == "channel_v2"
        ) else "phase3-offline.v1",
        "dataset_version": dataset.get("dataset_version"),
        "config": dict(config),
        "device": device,
        "relations": list(relations),
        "shape": shape.__dict__,
        "split_counts": {
            "train_samples": len(train_records),
            "validation_samples": len(val_records),
            "test_samples": len(test_records),
            "train_episode_ids": sorted({record["episode_id"] for record in train_records}),
            "validation_episode_ids": sorted({record["episode_id"] for record in val_records}),
            "test_episode_ids": sorted({record["episode_id"] for record in test_records}),
        },
        "normalization_fit": "train_samples_only",
        "feature_contract": {
            "node": str(config.get("node_feature_contract", "legacy_v1")),
            "edge": str(config.get("edge_feature_contract", "geometry_v1")),
            "action_normalization": action_normalization_contract,
            "current_holding_is_input": False,
            "future_or_target_fields_are_input": False,
            "node_feature_names": (
                list(HOLDER_OBJECT_V2_NODE_FEATURES)
                if config.get("node_feature_contract") == "holder_object_v2"
                else None
            ),
            "edge_feature_names": (
                list(HOLDER_OBJECT_V2_EDGE_FEATURES)
                if config.get("edge_feature_contract") == "holder_object_v2"
                else None
            ),
        },
        "parameter_matching": {
            "enabled": bool(config.get("parameter_match", False)),
            "target_parameter_count": config.get("target_parameter_count"),
            "hidden_dims": hidden_dims,
            "parameter_counts": selected_parameter_counts,
            "absolute_target_differences": {
                model_id: (
                    abs(count - int(config["target_parameter_count"]))
                    if config.get("target_parameter_count") is not None
                    else None
                )
                for model_id, count in selected_parameter_counts.items()
            },
        },
        "models": results,
        "controls": {
            "no_action": "legacy zero in normalized space (train-mean action)",
            "physical_zero_action": "raw zero command normalized with train statistics",
            "shuffled_action": "legacy within-batch roll; can retain same-episode neighboring actions and is not a strong semantic control",
            "global_shuffled_action": "deterministic split-wide action permutation with a different-episode donor for every row",
            "reversed_action": "reverse the six steps within each action window",
            "shuffled_gripper": "batch-roll only the gripper command sequence",
            "shuffled_arm": "batch-roll only the six arm command channels",
            "shuffled_edge": "sender and receiver indices independently rolled while labels stay fixed",
        },
        "limitations": [
            "This is an oracle graph offline probe, not RGB graph perception.",
            "P5 recurrent/history model is deferred because the current Phase 2R sample stores graph_t and graph_target, not a full past graph sequence.",
            "The existing scale-up test split contains the final two episode IDs; report task coverage with the metrics.",
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run_probe(dataset, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "probe_version": report["probe_version"],
        "device": report["device"],
        "models": len(report["models"]),
        "split_counts": report["split_counts"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
