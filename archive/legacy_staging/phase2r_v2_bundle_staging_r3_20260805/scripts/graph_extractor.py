"""Build a deterministic Phase 2 spatial graph from a LIBERO snapshot.

This is deliberately dependency-free.  It freezes the graph boundary before
adding a GNN: logical names identify nodes, positions define a complete
directed spatial topology, and unavailable predicates remain unknown.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


NODE_TYPES = ("robot", "object", "fixture", "site")
FEATURE_DIM = 24


def _finite_vector(value: Any, dim: int) -> list[float] | None:
    if value is None or isinstance(value, Mapping) or isinstance(value, (str, bytes)):
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or len(value) != dim:
        return None
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return vector if all(math.isfinite(item) for item in vector) else None


def _zero_or_value(value: list[float] | None, dim: int) -> tuple[list[float], float]:
    return (value, 1.0) if value is not None else ([0.0] * dim, 0.0)


def _position(entry: Mapping[str, Any]) -> list[float] | None:
    pose = entry.get("pose")
    if not isinstance(pose, Mapping):
        return None
    return _finite_vector(pose.get("position"), 3)


def _predicate_record(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"value": value, "valid": 1, "status": "available"}
    if isinstance(value, Mapping) and "error" in value:
        return {
            "value": None,
            "valid": 0,
            "status": "error",
            "error": str(value["error"]),
        }
    if value is None:
        return {"value": None, "valid": 0, "status": "missing"}
    return {"value": None, "valid": 0, "status": "unsupported"}


def _node_type_one_hot(node_type: str) -> list[int]:
    return [int(node_type == candidate) for candidate in NODE_TYPES]


def _make_feature_vector(
    position: list[float] | None,
    is_object_of_interest: bool,
    gripper_qpos: list[float] | None,
    joint_pos: list[float] | None,
    joint_vel: list[float] | None,
) -> tuple[list[float], dict[str, Any]]:
    position_values, position_valid = _zero_or_value(position, 3)
    gripper_values, gripper_valid = _zero_or_value(gripper_qpos, 2)
    joint_pos_values, joint_pos_valid = _zero_or_value(joint_pos, 7)
    joint_vel_values, joint_vel_valid = _zero_or_value(joint_vel, 7)
    vector = (
        position_values
        + [position_valid, float(bool(is_object_of_interest))]
        + gripper_values
        + [gripper_valid]
        + joint_pos_values
        + [joint_pos_valid]
        + joint_vel_values
        + [joint_vel_valid]
    )
    if len(vector) != FEATURE_DIM:
        raise AssertionError(f"unexpected feature dimension: {len(vector)}")
    groups = {
        "position": position_values,
        "position_valid": int(position_valid),
        "is_object_of_interest": int(bool(is_object_of_interest)),
        "gripper_qpos": gripper_values,
        "gripper_qpos_valid": int(gripper_valid),
        "joint_pos": joint_pos_values,
        "joint_pos_valid": int(joint_pos_valid),
        "joint_vel": joint_vel_values,
        "joint_vel_valid": int(joint_vel_valid),
    }
    return vector, groups


def _make_object_node(entry: Mapping[str, Any]) -> dict[str, Any]:
    node_id = str(entry["logical_id"])
    node_type = str(entry.get("node_type", "unknown"))
    if node_type not in NODE_TYPES:
        raise ValueError(f"unsupported node type for {node_id!r}: {node_type!r}")
    position = _position(entry)
    vector, groups = _make_feature_vector(
        position=position,
        is_object_of_interest=bool(entry.get("is_object_of_interest", False)),
        gripper_qpos=None,
        joint_pos=None,
        joint_vel=None,
    )
    return {
        "node_id": node_id,
        "node_type": node_type,
        "node_type_one_hot": _node_type_one_hot(node_type),
        "runtime_body_id": entry.get("body_id"),
        "raw_pose": entry.get("pose"),
        "features": groups,
        "feature_vector": vector,
        "predicate_state": {
            "joint_state": _predicate_record(entry.get("joint_state")),
            "is_open": _predicate_record(entry.get("is_open")),
            "is_close": _predicate_record(entry.get("is_close")),
        },
    }


def _make_robot_node(snapshot: Mapping[str, Any], robot_id: str = "robot0") -> dict[str, Any]:
    robot = snapshot.get("robot_state", {})
    if not isinstance(robot, Mapping):
        robot = {}
    position = _finite_vector(robot.get("eef_pos"), 3)
    vector, groups = _make_feature_vector(
        position=position,
        is_object_of_interest=False,
        gripper_qpos=_finite_vector(robot.get("gripper_qpos"), 2),
        joint_pos=_finite_vector(robot.get("joint_pos"), 7),
        joint_vel=_finite_vector(robot.get("joint_vel"), 7),
    )
    return {
        "node_id": robot_id,
        "node_type": "robot",
        "node_type_one_hot": _node_type_one_hot("robot"),
        "runtime_body_id": None,
        "raw_pose": {"position": position, "quaternion": robot.get("eef_quat")},
        "features": groups,
        "feature_vector": vector,
        "predicate_state": {},
        "available_observation_keys": list(robot.get("available_keys", [])),
    }


def _distance(left: list[float], right: list[float]) -> tuple[list[float], float]:
    delta = [right[index] - left[index] for index in range(3)]
    return delta, math.sqrt(sum(value * value for value in delta))


def _make_edges(nodes: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for source in nodes:
        source_position = source["features"].get("position")
        if source["features"].get("position_valid") != 1:
            continue
        for target in nodes:
            if source["node_id"] == target["node_id"]:
                continue
            target_position = target["features"].get("position")
            if target["features"].get("position_valid") != 1:
                continue
            delta, distance = _distance(source_position, target_position)
            edges.append(
                {
                    "source": source["node_id"],
                    "target": target["node_id"],
                    "edge_type": "spatial_directed",
                    "node_type_pair": [source["node_type"], target["node_type"]],
                    "features": {
                        "relative_position": delta,
                        "distance": distance,
                        "distance_valid": 1,
                    },
                    "semantic_relation": {
                        "value": None,
                        "valid": 0,
                        "status": "not_in_phase2_v1",
                    },
                }
            )
    return edges


def extract_graph_snapshot(snapshot: Mapping[str, Any], step: int | None = None) -> dict[str, Any]:
    """Convert one raw LIBERO snapshot into a deterministic graph snapshot."""

    object_entries = snapshot.get("object_states", [])
    if not isinstance(object_entries, Sequence):
        raise ValueError("snapshot.object_states must be a sequence")
    nodes = [_make_robot_node(snapshot)]
    nodes.extend(_make_object_node(entry) for entry in object_entries)
    node_ids = [node["node_id"] for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("duplicate logical_id in one snapshot")
    edges = _make_edges(nodes)
    predicate_errors = sum(
        1
        for node in nodes
        for record in node.get("predicate_state", {}).values()
        if record.get("status") == "error"
    )
    return {
        "step": int(snapshot.get("step", 0) if step is None else step),
        "node_feature_dim": FEATURE_DIM,
        "nodes": nodes,
        "edges": edges,
        "audit": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "predicate_error_count": predicate_errors,
            "missing_position_node_ids": [
                node["node_id"]
                for node in nodes
                if node["features"].get("position_valid") != 1
            ],
            "body_id_used_as_identity": False,
        },
    }


def build_graph_sequence(payload: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, Sequence) or isinstance(snapshots, (str, bytes)):
        raise ValueError(
            "input does not contain full 'snapshots'; use the full Colab capture, "
            "not the compact Phase 1 manifest"
        )
    graphs = [extract_graph_snapshot(snapshot) for snapshot in snapshots]
    return {
        "graph_spec": spec["spec_version"],
        "source": {
            "suite": payload.get("suite"),
            "task_id": payload.get("task_id"),
            "task_name": payload.get("task_name"),
            "seed": payload.get("seed"),
            "init_state_id": payload.get("init_state_id"),
        },
        "graphs": graphs,
        "temporal_audit": {
            "ordered_by": "step",
            "steps": [graph["step"] for graph in graphs],
            "future_fields_included": False,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=Path("configs/phase2_graph_spec.json"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    output = build_graph_sequence(payload, spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "graph_spec": output["graph_spec"],
        "graphs": len(output["graphs"]),
        "nodes_per_graph": [graph["audit"]["node_count"] for graph in output["graphs"]],
        "edges_per_graph": [graph["audit"]["edge_count"] for graph in output["graphs"]],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
