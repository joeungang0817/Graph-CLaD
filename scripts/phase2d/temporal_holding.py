"""Temporal holding labels recovered from the Phase 2D Colab prototype.

The label is evidence-based: contact and a closed gripper are necessary but
not sufficient.  A positive label also requires stable object/EFF relative
pose and object motion that follows the EFF over a short history window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence


@dataclass(frozen=True)
class HoldingPolicy:
    history_frames: int = 3
    gripper_closed_threshold: float = 0.025
    relative_pose_threshold: float = 0.03
    minimum_object_motion: float = 0.002
    minimum_eef_motion: float = 0.005
    follow_error_threshold: float = 0.05


def _vector(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return None
    try:
        result = tuple(float(value[i]) for i in range(3))
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(component) for component in result) else None


def _position(entry: Mapping[str, Any] | None) -> tuple[float, float, float] | None:
    if not isinstance(entry, Mapping):
        return None
    for key in ("position", "pos", "eef_pos"):
        result = _vector(entry.get(key))
        if result is not None:
            return result
    pose = entry.get("pose")
    if isinstance(pose, Mapping):
        return _position(pose)
    return _vector(pose)


def _object_positions(snapshot: Mapping[str, Any]) -> dict[str, tuple[float, float, float]]:
    entries = snapshot.get("object_states", [])
    if isinstance(entries, Mapping):
        entries = list(entries.values())
    result: dict[str, tuple[float, float, float]] = {}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, Mapping):
            continue
        object_id = entry.get("logical_id", entry.get("node_id", entry.get("name")))
        position = _position(entry)
        if object_id is not None and position is not None:
            result[str(object_id)] = position
    return result


def _subtract(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(left[i]) - float(right[i]) for i in range(3))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _pair_key(source: str, target: str) -> str:
    # Match the directed key contract in Phase 2R relation handlers.
    return f"{source}->{target}"


def _holding_evidence(
    snapshots: Sequence[Mapping[str, Any]], index: int, object_id: str, policy: HoldingPolicy
) -> dict[str, Any]:
    snapshot = snapshots[index]
    current_objects = _object_positions(snapshot)
    eef_position = _position(snapshot.get("robot_state"))
    relative_pose = (
        _subtract(current_objects[object_id], eef_position)
        if eef_position is not None and object_id in current_objects
        else None
    )

    raw_pairs = snapshot.get("robot_contact_pairs")
    contact_pairs: set[tuple[str, str]] = set()
    if isinstance(raw_pairs, list):
        for pair in raw_pairs:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                contact_pairs.add(tuple(sorted((str(pair[0]), str(pair[1])))))
    finger_contact = tuple(sorted(("robot0", object_id))) in contact_pairs

    robot = snapshot.get("robot_state", {})
    qpos = robot.get("gripper_qpos") if isinstance(robot, Mapping) else None
    try:
        qvalues = [float(value) for value in qpos] if isinstance(qpos, (list, tuple)) else []
    except (TypeError, ValueError):
        qvalues = []
    closed_gripper = bool(qvalues) and max(abs(value) for value in qvalues) <= policy.gripper_closed_threshold

    stable_checks: list[bool] = []
    follow_checks: list[bool] = []
    start = max(0, index - policy.history_frames + 1)
    for step in range(start + 1, index + 1):
        previous_objects = _object_positions(snapshots[step - 1])
        next_objects = _object_positions(snapshots[step])
        previous_eef = _position(snapshots[step - 1].get("robot_state"))
        next_eef = _position(snapshots[step].get("robot_state"))
        if (
            previous_eef is None or next_eef is None
            or object_id not in previous_objects or object_id not in next_objects
        ):
            continue
        previous_relative = _subtract(previous_objects[object_id], previous_eef)
        next_relative = _subtract(next_objects[object_id], next_eef)
        stable_checks.append(
            _norm(_subtract(next_relative, previous_relative)) <= policy.relative_pose_threshold
        )
        object_delta = _subtract(next_objects[object_id], previous_objects[object_id])
        eef_delta = _subtract(next_eef, previous_eef)
        follow_checks.append(
            _norm(eef_delta) >= policy.minimum_eef_motion
            and _norm(object_delta) >= policy.minimum_object_motion
            and _norm(_subtract(object_delta, eef_delta)) <= policy.follow_error_threshold
        )

    required_checks = max(1, policy.history_frames - 1)
    relative_pose_stable = bool(len(stable_checks) >= required_checks and all(stable_checks))
    object_followed_eef = bool(any(follow_checks))
    return {
        "finger_contact": finger_contact,
        "closed_gripper": closed_gripper,
        "relative_pose_stable": relative_pose_stable,
        "object_followed_eef": object_followed_eef,
        "history_frames": policy.history_frames,
        "relative_pose_threshold": policy.relative_pose_threshold,
        "valid": bool(relative_pose is not None and qvalues and isinstance(raw_pairs, list)),
    }


def annotate_temporal_holding(
    snapshots: Sequence[MutableMapping[str, Any]], policy: HoldingPolicy | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Annotate snapshots and return transition events, intervals, and traces."""

    policy = policy or HoldingPolicy()
    object_ids = sorted({object_id for snapshot in snapshots for object_id in _object_positions(snapshot)})
    previous_state = {object_id: "free" for object_id in object_ids}
    traces: dict[str, list[dict[str, Any]]] = {object_id: [] for object_id in object_ids}
    events: list[dict[str, Any]] = []

    for index, snapshot in enumerate(snapshots):
        holding_records = snapshot.setdefault("semantic_relation_records", {}).setdefault("holding", {})
        current_states: dict[str, str] = {}
        frame_events: list[dict[str, Any]] = []
        for object_id in object_ids:
            evidence = _holding_evidence(snapshots, index, object_id, policy)
            candidate = evidence["finger_contact"] and evidence["closed_gripper"]
            qualified = candidate and evidence["relative_pose_stable"] and evidence["object_followed_eef"]
            if qualified or (
                previous_state[object_id] == "holding"
                and candidate and evidence["relative_pose_stable"]
            ):
                state = "holding"
            elif candidate:
                state = "contact_candidate"
            elif previous_state[object_id] in {"holding", "contact_candidate"}:
                state = "release"
            else:
                state = "free"

            value = state == "holding"
            valid = bool(evidence.pop("valid"))
            label = {
                "value": value,
                "valid": valid,
                "confidence": "high" if value and valid else ("low" if valid else "unknown"),
                "status": "temporal_holding_v1",
                "state": state,
                "evidence": evidence,
            }
            holding_records[_pair_key("robot0", object_id)] = label
            traces[object_id].append(label)
            current_states[object_id] = state
            if state != previous_state[object_id]:
                event = {
                    "step": index,
                    "event": "holding_state_change",
                    "object_id": object_id,
                    "from_state": previous_state[object_id],
                    "to_state": state,
                    "valid": valid,
                }
                events.append(event)
                frame_events.append(event)
            previous_state[object_id] = state
        snapshot["holding_state"] = current_states
        snapshot["holding_state_events"] = frame_events

    intervals: list[dict[str, Any]] = []
    for object_id, records in traces.items():
        start: int | None = None
        for index, record in enumerate(records):
            if record["state"] == "holding" and start is None:
                start = index
            if start is not None and (record["state"] != "holding" or index == len(records) - 1):
                end = index if record["state"] == "holding" else index - 1
                intervals.append(
                    {"object_id": object_id, "start": start, "end": end, "length": end - start + 1}
                )
                start = None
    return events, intervals, traces
