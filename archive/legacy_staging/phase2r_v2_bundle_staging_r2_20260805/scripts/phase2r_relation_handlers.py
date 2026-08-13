"""Phase 2R frame transform and capability-aware relation handlers.

The handlers deliberately distinguish ``false`` from ``unknown``.  LIBERO's
object-state wrappers are not uniform across object/site types, so a wrapper
exception is never converted into a negative training label.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any


RELATIONS = (
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
    "open",
    "close",
)
GEOMETRIC_RELATIONS = {"left", "right", "front", "behind", "above", "below"}
SEMANTIC_RELATIONS = {"on", "inside", "holding", "open", "close"}
MARGIN = 0.02
GRIPPER_CLOSED_THRESHOLD = 0.01


def relation_record(value: bool | None, definition: str, error: str | None = None) -> dict[str, Any]:
    """Return the common value/valid record used by the graph schema."""

    if isinstance(value, bool):
        return {
            "value": value,
            "valid": 1,
            "definition_version": definition,
            "status": "available",
        }
    record: dict[str, Any] = {
        "value": None,
        "valid": 0,
        "definition_version": definition,
        "status": "unknown",
    }
    if error:
        record["status"] = "error"
        record["error"] = str(error)[:240]
    return record


def pair_key(source_id: str, target_id: str) -> str:
    """Encode a directed logical-id pair for JSON storage."""

    return f"{source_id}->{target_id}"


def _finite_vector(value: Any, dimension: int) -> list[float] | None:
    if value is None or isinstance(value, Mapping) or isinstance(value, (str, bytes)):
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or len(value) != dimension:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _normalise_quaternion(quaternion: Any) -> list[float] | None:
    value = _finite_vector(quaternion, 4)
    if value is None:
        return None
    norm = math.sqrt(sum(item * item for item in value))
    if norm <= 1e-12:
        return None
    return [item / norm for item in value]


def transform_position_to_robot_base(
    position: Any,
    robot_base_pose: Mapping[str, Any],
) -> list[float] | None:
    """Transform a world position into the robot-base frame.

    LIBERO/MuJoCo body quaternions are stored here as ``wxyz``.  The inverse
    base rotation is applied after subtracting the base translation.
    """

    point = _finite_vector(position, 3)
    base_position = _finite_vector(robot_base_pose.get("position"), 3)
    quaternion = _normalise_quaternion(robot_base_pose.get("quaternion"))
    if point is None or base_position is None or quaternion is None:
        return None
    translated = [point[index] - base_position[index] for index in range(3)]
    w, x, y, z = quaternion
    # q_conjugate * [0, translated] * q, expanded for a small dependency-free
    # transform.  This is the inverse rotation for a unit wxyz quaternion.
    q_conjugate = (-x, -y, -z)
    uv = [
        q_conjugate[1] * translated[2] - q_conjugate[2] * translated[1],
        q_conjugate[2] * translated[0] - q_conjugate[0] * translated[2],
        q_conjugate[0] * translated[1] - q_conjugate[1] * translated[0],
    ]
    uuv = [
        q_conjugate[1] * uv[2] - q_conjugate[2] * uv[1],
        q_conjugate[2] * uv[0] - q_conjugate[0] * uv[2],
        q_conjugate[0] * uv[1] - q_conjugate[1] * uv[0],
    ]
    return [
        translated[index] + 2.0 * (w * uv[index] + uuv[index])
        for index in range(3)
    ]


def transform_snapshot_to_robot_base(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Copy one raw snapshot and transform all model-facing positions."""

    transformed = copy.deepcopy(dict(snapshot))
    robot_base_pose = transformed.get("robot_base_pose")
    if not isinstance(robot_base_pose, Mapping):
        frame = transformed.get("frame", {})
        robot_base_pose = frame.get("robot_base_pose") if isinstance(frame, Mapping) else None
    if not isinstance(robot_base_pose, Mapping):
        raise ValueError(
            "robot_base_pose is missing; recollect trajectories with the Phase 2R collector"
        )
    if transform_position_to_robot_base([0.0, 0.0, 0.0], robot_base_pose) is None:
        raise ValueError("robot_base_pose has invalid position or wxyz quaternion")

    robot_state = transformed.get("robot_state")
    if isinstance(robot_state, Mapping):
        robot_state = dict(robot_state)
        eef_position = transform_position_to_robot_base(robot_state.get("eef_pos"), robot_base_pose)
        if robot_state.get("eef_pos") is not None and eef_position is None:
            raise ValueError("robot_state.eef_pos is invalid and cannot be transformed")
        if eef_position is not None:
            robot_state["eef_pos"] = eef_position
        transformed["robot_state"] = robot_state

    object_states = transformed.get("object_states", [])
    if isinstance(object_states, Sequence) and not isinstance(object_states, (str, bytes)):
        for entry in object_states:
            if not isinstance(entry, Mapping):
                continue
            pose = entry.get("pose")
            if not isinstance(pose, Mapping):
                continue
            position = pose.get("position")
            transformed_position = transform_position_to_robot_base(position, robot_base_pose)
            if position is not None and transformed_position is None:
                raise ValueError(f"invalid pose position for logical_id={entry.get('logical_id')!r}")
            if transformed_position is not None:
                pose["position"] = transformed_position
                pose["frame"] = "robot_base"

    geometry = transformed.get("geometry")
    if isinstance(geometry, Mapping):
        transformed_geometry: dict[str, Any] = {}
        for node_id, box in geometry.items():
            if not isinstance(box, Mapping):
                continue
            lower = _finite_vector(box.get("aabb_min"), 3)
            upper = _finite_vector(box.get("aabb_max"), 3)
            center = _finite_vector(box.get("center"), 3)
            if lower is None or upper is None or center is None:
                transformed_geometry[str(node_id)] = copy.deepcopy(dict(box))
                continue
            corners = [
                [x, y, z]
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
                for z in (lower[2], upper[2])
            ]
            transformed_corners = [
                transform_position_to_robot_base(corner, robot_base_pose)
                for corner in corners
            ]
            transformed_center = transform_position_to_robot_base(center, robot_base_pose)
            if transformed_center is None or any(corner is None for corner in transformed_corners):
                transformed_geometry[str(node_id)] = copy.deepcopy(dict(box))
                continue
            valid_corners = [corner for corner in transformed_corners if corner is not None]
            transformed_geometry[str(node_id)] = {
                **copy.deepcopy(dict(box)),
                "center": transformed_center,
                "aabb_min": [min(corner[i] for corner in valid_corners) for i in range(3)],
                "aabb_max": [max(corner[i] for corner in valid_corners) for i in range(3)],
                "frame": "robot_base",
            }
        transformed["geometry"] = transformed_geometry

    transformed["coordinate_frame"] = "robot_base"
    transformed["frame_transform"] = {
        "source": "raw_world",
        "target": "robot_base",
        "robot_base_pose": copy.deepcopy(dict(robot_base_pose)),
        "quaternion_convention": "wxyz",
        "position_transform": "subtract_base_then_inverse_rotate",
    }
    return transformed


def _position(node: Mapping[str, Any]) -> list[float] | None:
    features = node.get("features", {})
    if not isinstance(features, Mapping) or features.get("position_valid") != 1:
        return None
    return _finite_vector(features.get("position"), 3)


def _pair_record_from_snapshot(
    snapshot: Mapping[str, Any],
    relation: str,
    source_id: str,
    target_id: str,
) -> dict[str, Any] | None:
    records = snapshot.get("semantic_relation_records", {})
    if not isinstance(records, Mapping):
        return None
    relation_records = records.get(relation, {})
    if not isinstance(relation_records, Mapping):
        return None
    raw = relation_records.get(pair_key(source_id, target_id))
    if not isinstance(raw, Mapping):
        return None
    return dict(raw)


def _node_record_from_snapshot(
    snapshot: Mapping[str, Any],
    relation: str,
    node_id: str,
) -> dict[str, Any] | None:
    node_semantics = snapshot.get("node_semantics", {})
    if not isinstance(node_semantics, Mapping):
        return None
    node_records = node_semantics.get(node_id, {})
    if not isinstance(node_records, Mapping):
        return None
    raw = node_records.get(relation)
    return dict(raw) if isinstance(raw, Mapping) else None


def _contact_record(
    snapshot: Mapping[str, Any],
    source_id: str,
    target_id: str,
) -> dict[str, Any]:
    contact_pairs = snapshot.get("contact_pairs")
    if contact_pairs is None:
        return relation_record(None, "mujoco_contact")
    pair = tuple(sorted((source_id, target_id)))
    available = {
        tuple(sorted((str(item[0]), str(item[1]))))
        for item in contact_pairs
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2
    }
    return relation_record(pair in available, "mujoco_contact")


def _holding_record(
    snapshot: Mapping[str, Any],
    source_id: str,
    target_id: str,
    gripper_threshold: float,
) -> dict[str, Any]:
    authoritative = _pair_record_from_snapshot(snapshot, "holding", source_id, target_id)
    if authoritative is not None:
        return authoritative
    if source_id != "robot0":
        return relation_record(None, "gripper_contact_and_closed_qpos")
    robot_contacts = snapshot.get("robot_contact_pairs")
    robot_contact = False
    if isinstance(robot_contacts, Sequence) and not isinstance(robot_contacts, (str, bytes)):
        robot_contact = any(
            isinstance(pair, Sequence)
            and len(pair) == 2
            and tuple(sorted((str(pair[0]), str(pair[1])))) == tuple(sorted((source_id, target_id)))
            for pair in robot_contacts
        )
    robot_state = snapshot.get("robot_state", {})
    gripper_qpos = robot_state.get("gripper_qpos") if isinstance(robot_state, Mapping) else None
    values = _finite_vector(gripper_qpos, 2)
    if values is None:
        return relation_record(None, "gripper_contact_and_closed_qpos")
    closed = max(abs(value) for value in values) <= float(gripper_threshold)
    return relation_record(
        bool(robot_contact and closed),
        f"gripper_contact_and_abs_qpos_le_{gripper_threshold:g}",
    )


def _geometry_record(
    snapshot: Mapping[str, Any],
    relation: str,
    source_id: str,
    target_id: str,
) -> dict[str, Any]:
    """Use an explicit captured AABB only; never infer size from point poses."""

    if relation not in {"on", "inside"}:
        return relation_record(None, f"{relation}_handler_unavailable")
    geometry = snapshot.get("geometry", {})
    if not isinstance(geometry, Mapping):
        return relation_record(None, f"geometry_{relation}_aabb_missing")
    source_box = geometry.get(source_id)
    target_box = geometry.get(target_id)
    if not isinstance(source_box, Mapping) or not isinstance(target_box, Mapping):
        return relation_record(None, f"geometry_{relation}_aabb_missing")
    source_position = _finite_vector(source_box.get("center"), 3)
    target_min = _finite_vector(target_box.get("aabb_min"), 3)
    target_max = _finite_vector(target_box.get("aabb_max"), 3)
    if source_position is None or target_min is None or target_max is None:
        return relation_record(None, f"geometry_{relation}_aabb_invalid")
    inside = all(target_min[i] - MARGIN <= source_position[i] <= target_max[i] + MARGIN for i in range(3))
    if relation == "inside":
        return relation_record(inside, "geometry_source_center_inside_target_aabb_margin_0.02")
    source_min = _finite_vector(source_box.get("aabb_min"), 3)
    source_max = _finite_vector(source_box.get("aabb_max"), 3)
    if source_min is None or source_max is None:
        return relation_record(None, "geometry_on_source_aabb_missing")
    horizontal_inside = (
        target_min[0] - MARGIN <= source_position[0] <= target_max[0] + MARGIN
        and target_min[1] - MARGIN <= source_position[1] <= target_max[1] + MARGIN
    )
    vertical_gap = abs(source_min[2] - target_max[2])
    return relation_record(
        bool(horizontal_inside and vertical_gap <= MARGIN),
        "geometry_source_bottom_on_target_top_margin_0.02",
    )


def relation_labels(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    margin: float = MARGIN,
    gripper_threshold: float = GRIPPER_CLOSED_THRESHOLD,
) -> dict[str, dict[str, Any]]:
    """Build all edge labels with explicit capability-aware fallbacks."""

    source_id = str(source["node_id"])
    target_id = str(target["node_id"])
    source_pos = _position(source)
    target_pos = _position(target)
    labels: dict[str, dict[str, Any]] = {}
    if source_pos is None or target_pos is None:
        for relation in RELATIONS:
            labels[relation] = relation_record(None, "position_unavailable")
        return labels
    dx = target_pos[0] - source_pos[0]
    dy = target_pos[1] - source_pos[1]
    dz = target_pos[2] - source_pos[2]
    frame_name = str(snapshot.get("coordinate_frame", "unspecified_frame"))
    labels.update(
        {
            "left": relation_record(dx < -margin, f"{frame_name}_axis_x_margin_{margin:g}"),
            "right": relation_record(dx > margin, f"{frame_name}_axis_x_margin_{margin:g}"),
            "front": relation_record(dy > margin, f"{frame_name}_axis_y_margin_{margin:g}"),
            "behind": relation_record(dy < -margin, f"{frame_name}_axis_y_margin_{margin:g}"),
            "above": relation_record(dz > margin, f"{frame_name}_axis_z_margin_{margin:g}"),
            "below": relation_record(dz < -margin, f"{frame_name}_axis_z_margin_{margin:g}"),
            "contact": _contact_record(snapshot, source_id, target_id),
        }
    )
    for relation in ("on", "inside"):
        wrapper_record = _pair_record_from_snapshot(snapshot, relation, source_id, target_id)
        geometry_record = _geometry_record(snapshot, relation, source_id, target_id)
        if isinstance(wrapper_record, Mapping) and wrapper_record.get("valid"):
            labels[relation] = dict(wrapper_record)
        elif isinstance(geometry_record, Mapping) and geometry_record.get("valid"):
            labels[relation] = geometry_record
        else:
            labels[relation] = dict(wrapper_record or geometry_record)
    labels["holding"] = _holding_record(snapshot, source_id, target_id, gripper_threshold)
    # open/close are unary node states, not pairwise relations.  Keep them
    # unknown on edges and expose the node-level records separately.
    for relation in ("open", "close"):
        labels[relation] = relation_record(None, "node_state_only_not_pairwise")
    return labels


def node_semantic_records(snapshot: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """Normalize captured node predicates for open/close and future targets."""

    result: dict[str, dict[str, dict[str, Any]]] = {}
    for entry in snapshot.get("object_states", []):
        if not isinstance(entry, Mapping):
            continue
        node_id = str(entry.get("logical_id"))
        predicate_state = entry.get("predicate_state", {})
        if not isinstance(predicate_state, Mapping):
            predicate_state = {}
        node_records: dict[str, dict[str, Any]] = {}
        for relation, key in (("open", "is_open"), ("close", "is_close")):
            raw = entry.get(key)
            if raw is None:
                raw = predicate_state.get(key)
            if isinstance(raw, bool):
                node_records[relation] = relation_record(raw, "libero_state_wrapper")
            elif isinstance(raw, Mapping) and isinstance(raw.get("value"), bool) and raw.get("valid"):
                node_records[relation] = dict(raw)
            else:
                error = raw.get("error") if isinstance(raw, Mapping) else None
                node_records[relation] = relation_record(None, "libero_state_wrapper", error=error)
        result[node_id] = node_records
    return result
