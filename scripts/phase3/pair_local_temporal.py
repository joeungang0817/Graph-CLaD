"""Causal pair-history features for the Phase 3 H0--H3 architecture gate.

The Phase 2D transition samples store only ``graph_t`` and ``graph_target``.
This module reconstructs a short history by indexing graph frames from other
windows in the same episode.  Every requested frame is at or before the
sample's current step; target graphs are used only as another serialization
of an already-past frame.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HISTORY_FEATURE_KEY = "causal_pair_history_v1"
HISTORY_FEATURE_NAMES = (
    "relative_position_delta_x",
    "relative_position_delta_y",
    "relative_position_delta_z",
    "relative_position_delta_valid",
    "relative_velocity_x",
    "relative_velocity_y",
    "relative_velocity_z",
    "relative_velocity_valid",
    "contact_fraction",
    "contact_trailing_fraction",
    "contact_persistence_valid",
    "gripper_closure_velocity",
    "gripper_closure_velocity_valid",
    "object_following_residual_mean",
    "object_following_residual_max",
    "object_following_stability_valid",
)
HISTORY_DIM = len(HISTORY_FEATURE_NAMES)


def _task_path(root: Path, task_id: int) -> Path:
    return root / f"task{task_id}" / f"phase2d_task{task_id}_graph_dataset.jsonl.gz"


def _canonical_digest(graph: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        graph, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _node_map(graph: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(node.get("node_id")): node
        for node in graph.get("nodes", [])
        if isinstance(node, Mapping)
    }


def _edge_map(graph: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(edge.get("source")), str(edge.get("target"))): edge
        for edge in graph.get("edges", [])
        if isinstance(edge, Mapping)
    }


def _finite_vector(value: Any, length: int) -> list[float] | None:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if len(result) != length or not all(math.isfinite(item) for item in result):
        return None
    return result


def _relation(edge: Mapping[str, Any], name: str) -> tuple[float, bool]:
    record = (edge.get("relations", {}) or {}).get(name, {})
    if not isinstance(record, Mapping) or not record.get("valid"):
        return 0.0, False
    return float(bool(record.get("value"))), True


def requested_history_keys(
    samples: Iterable[Mapping[str, Any]], lookback_steps: int
) -> set[tuple[int, str, int]]:
    if lookback_steps < 1:
        raise ValueError("lookback_steps must be at least one")
    requested: set[tuple[int, str, int]] = set()
    for sample in samples:
        task_id = int(sample["task_id"])
        episode_id = str(sample["episode_id"])
        current_step = int(sample["start_step"])
        for step in range(max(0, current_step - lookback_steps), current_step + 1):
            requested.add((task_id, episode_id, step))
    return requested


def load_requested_graph_history(
    natural_root: Path,
    samples: Sequence[Mapping[str, Any]],
    lookback_steps: int = 3,
) -> tuple[dict[tuple[int, str, int], Mapping[str, Any]], dict[str, Any]]:
    """Load only graph frames needed by ``samples`` and report exact QA."""

    requested = requested_history_keys(samples, lookback_steps)
    by_task: dict[int, set[tuple[int, str, int]]] = defaultdict(set)
    for key in requested:
        by_task[key[0]].add(key)
    graphs: dict[tuple[int, str, int], Mapping[str, Any]] = {}
    digests: dict[tuple[int, str, int], str] = {}
    conflicts: list[dict[str, Any]] = []
    for task_id in sorted(by_task):
        path = _task_path(natural_root, task_id)
        if not path.exists():
            raise FileNotFoundError(path)
        unresolved = by_task[task_id]
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                episode_default = f"task{task_id}_{payload.get('demo_id', 'unknown')}"
                for raw in payload.get("samples", []):
                    if not isinstance(raw, Mapping):
                        continue
                    episode_id = str(raw.get("episode_id") or episode_default)
                    candidates = (
                        (int(raw.get("start_step", -1)), raw.get("graph_t")),
                        (int(raw.get("target_step", -1)), raw.get("graph_target")),
                    )
                    for step, graph in candidates:
                        key = (task_id, episode_id, step)
                        if key not in unresolved or not isinstance(graph, Mapping):
                            continue
                        digest = _canonical_digest(graph)
                        if key in digests and digests[key] != digest:
                            conflicts.append(
                                {
                                    "task_id": task_id,
                                    "episode_id": episode_id,
                                    "step": step,
                                    "first_sha256": digests[key],
                                    "new_sha256": digest,
                                }
                            )
                            continue
                        graphs[key] = graph
                        digests[key] = digest
        # Do not stop at the first copy: duplicate frame payloads must agree.

    missing = sorted(requested - set(graphs))
    qa = {
        "contract": "causal_pair_history_v1",
        "lookback_steps": int(lookback_steps),
        "time_range": "[max(0,t-lookback_steps), t]",
        "future_frame_reads": 0,
        "requested_frames": len(requested),
        "available_frames": len(graphs),
        "missing_frames": [
            {"task_id": task, "episode_id": episode, "step": step}
            for task, episode, step in missing
        ],
        "conflicting_frames": conflicts,
        "status": "pass" if not missing and not conflicts else "fail",
    }
    return graphs, qa


def _point(
    graph: Mapping[str, Any], edge_key: tuple[str, str]
) -> dict[str, Any]:
    nodes = _node_map(graph)
    edges = _edge_map(graph)
    source = nodes.get(edge_key[0], {})
    target = nodes.get(edge_key[1], {})
    source_type = str(source.get("node_type", "unknown"))
    target_type = str(target.get("node_type", "unknown"))
    source_features = source.get("features", {}) or {}
    target_features = target.get("features", {}) or {}
    source_position = _finite_vector(source_features.get("position"), 3)
    target_position = _finite_vector(target_features.get("position"), 3)
    positions_valid = bool(
        source_position is not None
        and target_position is not None
        and source_features.get("position_valid") == 1
        and target_features.get("position_valid") == 1
    )
    relative = (
        [target_position[i] - source_position[i] for i in range(3)]
        if positions_valid
        else None
    )
    if source_type == "robot":
        robot_features, robot_position = source_features, source_position
        object_position = target_position
    elif target_type == "robot":
        robot_features, robot_position = target_features, target_position
        object_position = source_position
    else:
        robot_features, robot_position, object_position = {}, None, None
    qpos = _finite_vector(robot_features.get("gripper_qpos"), 2)
    qpos_valid = bool(
        qpos is not None and robot_features.get("gripper_qpos_valid") == 1
    )
    aperture = max(abs(value) for value in qpos) if qpos_valid else None
    edge = edges.get(edge_key, {})
    contact, contact_valid = _relation(edge, "contact")
    if not contact_valid:
        contact, contact_valid = _relation(edges.get((edge_key[1], edge_key[0]), {}), "contact")
    return {
        "relative": relative,
        "robot_position": robot_position if positions_valid else None,
        "object_position": object_position if positions_valid else None,
        "aperture": aperture,
        "contact": contact,
        "contact_valid": contact_valid,
    }


def causal_pair_history_features(
    graphs: Sequence[tuple[int, Mapping[str, Any]]],
    edge_key: tuple[str, str],
) -> tuple[list[float], dict[str, Any]]:
    """Compute the fixed 16-D feature vector from frames ordered through t."""

    ordered = sorted(graphs, key=lambda item: item[0])
    points = [(step, _point(graph, edge_key)) for step, graph in ordered]

    position_points = [
        (step, point)
        for step, point in points
        if point["relative"] is not None
    ]
    if len(position_points) >= 2:
        first_step, first = position_points[0]
        last_step, last = position_points[-1]
        gap = max(last_step - first_step, 1)
        relative_delta = [
            last["relative"][i] - first["relative"][i] for i in range(3)
        ]
        relative_velocity = [value / gap for value in relative_delta]
        relative_valid = velocity_valid = 1.0
    else:
        relative_delta = [0.0, 0.0, 0.0]
        relative_velocity = [0.0, 0.0, 0.0]
        relative_valid = velocity_valid = 0.0

    contact_values = [point["contact"] for _, point in points if point["contact_valid"]]
    if contact_values:
        contact_fraction = sum(contact_values) / len(contact_values)
        trailing = 0
        for value in reversed(contact_values):
            if not value:
                break
            trailing += 1
        contact_trailing_fraction = trailing / len(contact_values)
        contact_valid = 1.0
    else:
        contact_fraction = contact_trailing_fraction = contact_valid = 0.0

    aperture_points = [
        (step, point["aperture"])
        for step, point in points
        if point["aperture"] is not None
    ]
    if len(aperture_points) >= 2:
        first_step, first_aperture = aperture_points[0]
        last_step, last_aperture = aperture_points[-1]
        # Positive means the aperture is closing.
        closure_velocity = (first_aperture - last_aperture) / max(
            last_step - first_step, 1
        )
        closure_valid = 1.0
    else:
        closure_velocity = closure_valid = 0.0

    residuals: list[float] = []
    for (_, previous), (_, current) in zip(points, points[1:]):
        if any(
            value is None
            for value in (
                previous["robot_position"], previous["object_position"],
                current["robot_position"], current["object_position"],
            )
        ):
            continue
        robot_delta = [
            current["robot_position"][i] - previous["robot_position"][i]
            for i in range(3)
        ]
        object_delta = [
            current["object_position"][i] - previous["object_position"][i]
            for i in range(3)
        ]
        residuals.append(
            math.sqrt(sum((object_delta[i] - robot_delta[i]) ** 2 for i in range(3)))
        )
    if residuals:
        residual_mean = sum(residuals) / len(residuals)
        residual_max = max(residuals)
        following_valid = 1.0
    else:
        residual_mean = residual_max = following_valid = 0.0

    features = (
        relative_delta
        + [relative_valid]
        + relative_velocity
        + [velocity_valid]
        + [contact_fraction, contact_trailing_fraction, contact_valid]
        + [closure_velocity, closure_valid]
        + [residual_mean, residual_max, following_valid]
    )
    if len(features) != HISTORY_DIM or not all(math.isfinite(value) for value in features):
        raise AssertionError("invalid causal pair-history feature vector")
    qa = {
        "frames": len(points),
        "first_step": points[0][0] if points else None,
        "last_step": points[-1][0] if points else None,
        "valid_masks": {
            "relative_position_delta": int(relative_valid),
            "relative_velocity": int(velocity_valid),
            "contact_persistence": int(contact_valid),
            "gripper_closure_velocity": int(closure_valid),
            "object_following_stability": int(following_valid),
        },
    }
    return [float(value) for value in features], qa


def attach_causal_pair_history(
    sample: Mapping[str, Any],
    graph_index: Mapping[tuple[int, str, int], Mapping[str, Any]],
    lookback_steps: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attach causal features to every current edge without touching labels."""

    output = copy.deepcopy(dict(sample))
    task_id = int(output["task_id"])
    episode_id = str(output["episode_id"])
    current_step = int(output["start_step"])
    steps = list(range(max(0, current_step - lookback_steps), current_step + 1))
    frames = [
        (step, graph_index[(task_id, episode_id, step)])
        for step in steps
        if (task_id, episode_id, step) in graph_index
    ]
    edge_qas: list[dict[str, Any]] = []
    for edge in output.get("graph_t", {}).get("edges", []):
        edge_key = (str(edge.get("source")), str(edge.get("target")))
        features, qa = causal_pair_history_features(frames, edge_key)
        edge.setdefault("features", {})[HISTORY_FEATURE_KEY] = features
        edge_qas.append({"edge": list(edge_key), **qa})
    return output, {
        "sample_id": "|".join(
            str(output.get(key))
            for key in ("suite", "task_id", "episode_id", "start_step", "target_step", "tau")
        ),
        "current_step": current_step,
        "requested_steps": steps,
        "available_steps": [step for step, _ in frames],
        "future_frame_reads": 0,
        "edges": edge_qas,
    }
