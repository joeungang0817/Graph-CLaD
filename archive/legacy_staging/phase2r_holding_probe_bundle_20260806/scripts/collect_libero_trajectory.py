"""Collect controlled, action-bearing LIBERO oracle trajectories.

This is the Phase 1b pilot collector. It deliberately records privileged
simulator state and a small bounded nonzero action probe; it is not an RGB
perception result or a demonstration-quality policy dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from .inspect_libero_state import (
        _infer_action_dim,
        _jsonable,
        _sim_metadata,
        extract_object_states,
        extract_robot_state,
        observation_schema,
    )
except ImportError:  # pragma: no cover - CLI execution from scripts/
    from inspect_libero_state import (
        _infer_action_dim,
        _jsonable,
        _sim_metadata,
        extract_object_states,
        extract_robot_state,
        observation_schema,
    )


def _contact_pairs(env: Any, object_states: list[dict[str, Any]]) -> list[list[str]]:
    """Map current MuJoCo contacts to logical IDs, including gripper contacts.

    MuJoCo names the Panda arm bodies ``robot0_*`` but names the gripper
    bodies ``gripper0_*``.  Both represent the robot side of a holding
    relation and must map to the stable logical node ``robot0``.
    """

    sim = getattr(env, "sim", None)
    data = getattr(sim, "data", None)
    model = getattr(sim, "model", None)
    if data is None or model is None:
        return []
    body_names = {
        int(entry["body_id"]): str(entry["logical_id"])
        for entry in object_states
        if entry.get("body_id") is not None
    }
    pairs: set[tuple[str, str]] = set()
    try:
        ncon = int(data.ncon)
        geom_bodyid = model.geom_bodyid
        for index in range(ncon):
            contact = data.contact[index]
            body_a = int(geom_bodyid[int(contact.geom1)])
            body_b = int(geom_bodyid[int(contact.geom2)])
            for body_id, logical_id in ((body_a, "robot0"), (body_b, "robot0")):
                if body_id not in body_names:
                    try:
                        body_name = str(model.body_id2name(body_id))
                    except Exception:
                        body_name = ""
                    if body_name.startswith(("robot0", "gripper0")):
                        body_names[body_id] = logical_id
            if body_a not in body_names or body_b not in body_names:
                continue
            left, right = sorted((body_names[body_a], body_names[body_b]))
            if left != right:
                pairs.add((left, right))
    except Exception:
        return []
    return [[left, right] for left, right in sorted(pairs)]


def _robot_base_pose(env: Any) -> dict[str, Any] | None:
    """Capture the per-episode world pose used for the robot-base transform."""

    sim = getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        return None
    try:
        body_id = int(model.body_name2id("robot0_base"))
        return {
            "body_name": "robot0_base",
            "body_id": body_id,
            "position": _jsonable(data.body_xpos[body_id]),
            "quaternion": _jsonable(data.body_xquat[body_id]),
            "quaternion_convention": "MuJoCo body_xquat raw wxyz ordering",
        }
    except Exception:
        return None


def _task_semantic_metadata(bddl_file: Path) -> dict[str, Any]:
    """Read explicit relation targets from the task BDDL when available.

    LIBERO's Python state wrappers do not expose a uniform ``inside`` API for
    sites and fixtures.  The BDDL goal is the authoritative task-level source
    for which targets are intended to behave as containers, so this metadata
    is used only to grant geometry fallback capability; it does not turn an
    arbitrary overlap into an ``inside`` label.
    """

    try:
        text = bddl_file.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "source": "bddl_goal_predicates",
            "available": False,
            "error": f"{type(exc).__name__}: {exc}"[:240],
            "inside_pairs": [],
            "containment_capable_ids": [],
            "on_pairs": [],
        }

    def binary_predicates(predicate: str) -> list[list[str]]:
        pattern = rf"\(\s*{predicate}\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*\)"
        return [
            [match.group(1), match.group(2)]
            for match in re.finditer(pattern, text, flags=re.IGNORECASE)
        ]

    inside_pairs = binary_predicates("In")
    return {
        "source": "bddl_goal_predicates",
        "available": True,
        "inside_pairs": inside_pairs,
        "containment_capable_ids": sorted({pair[1] for pair in inside_pairs}),
        "on_pairs": binary_predicates("On"),
    }


def _geometry_boxes(
    env: Any,
    object_states: list[dict[str, Any]],
    task_semantics: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Capture conservative world AABBs for bodies and MuJoCo sites.

    The box is audit/semantic metadata, not a learned feature.  It is used by
    the optional ``on``/``inside`` geometry handlers only when the simulator
    exposes enough size information.  Site boxes are particularly important
    for LIBERO's ``In object region`` goals, where the wrapper often lacks a
    usable containment method.
    """

    sim = getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        return {}
    geom_bodyid = getattr(model, "geom_bodyid", None)
    geom_size = getattr(model, "geom_size", None)
    geom_xpos = getattr(data, "geom_xpos", None)
    geom_xmat = getattr(data, "geom_xmat", None)
    if geom_bodyid is None or geom_size is None or geom_xpos is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    task_semantics = task_semantics if isinstance(task_semantics, Mapping) else {}
    containment_capable_ids = {
        str(value) for value in task_semantics.get("containment_capable_ids", [])
    }
    for entry in object_states:
        body_id = entry.get("body_id")
        if body_id is None:
            continue
        try:
            body_id = int(body_id)
            geom_ids = [
                index for index, owner in enumerate(geom_bodyid)
                if int(owner) == body_id
            ]
            corners_min: list[np.ndarray] = []
            corners_max: list[np.ndarray] = []
            for geom_id in geom_ids:
                center = np.asarray(geom_xpos[geom_id], dtype=float)
                half_size = np.abs(np.asarray(geom_size[geom_id], dtype=float))
                if center.shape != (3,) or half_size.shape != (3,) or not np.isfinite(center).all():
                    continue
                if geom_xmat is not None:
                    rotation = np.asarray(geom_xmat[geom_id], dtype=float).reshape(3, 3)
                    if np.isfinite(rotation).all():
                        half_size = np.abs(rotation) @ half_size
                corners_min.append(center - half_size)
                corners_max.append(center + half_size)
            if not corners_min:
                continue
            lower = np.min(np.stack(corners_min), axis=0)
            upper = np.max(np.stack(corners_max), axis=0)
            result[str(entry["logical_id"])] = {
                "center": ((lower + upper) / 2.0).tolist(),
                "aabb_min": lower.tolist(),
                "aabb_max": upper.tolist(),
                "valid": 1,
                "source": "mujoco_body_geom_union_aabb",
                "support_surface_candidate": entry.get("node_type") in {"fixture", "site"},
                "containment_capable": str(entry["logical_id"]) in containment_capable_ids,
            }
        except Exception:
            continue

    # LIBERO relation targets are frequently MuJoCo sites rather than bodies.
    # Use the site's oriented box and store its conservative world AABB.  A
    # site is a support candidate by default, while containment capability is
    # granted only by an explicit BDDL ``In`` predicate above.
    raw_env = getattr(env, "env", env)
    object_sites = getattr(raw_env, "object_sites_dict", {}) or {}
    site_xpos = getattr(data, "site_xpos", None)
    site_xmat = getattr(data, "site_xmat", None)
    site_size = getattr(model, "site_size", None)
    if site_xpos is not None and site_size is not None:
        for logical_id, site_object in object_sites.items():
            logical_id = str(logical_id)
            if logical_id in result:
                continue
            site_name = str(getattr(site_object, "name", logical_id))
            try:
                site_id = int(model.site_name2id(site_name))
                center = np.asarray(site_xpos[site_id], dtype=float)
                size = np.asarray(site_size[site_id], dtype=float)
                if center.shape != (3,) or size.shape != (3,) or not np.isfinite(center).all():
                    continue
                rotation = np.eye(3)
                if site_xmat is not None:
                    candidate = np.asarray(site_xmat[site_id], dtype=float).reshape(3, 3)
                    if np.isfinite(candidate).all():
                        rotation = candidate
                half_size = np.abs(rotation) @ np.abs(size)
                lower = center - half_size
                upper = center + half_size
                result[logical_id] = {
                    "center": center.tolist(),
                    "aabb_min": lower.tolist(),
                    "aabb_max": upper.tolist(),
                    "valid": 1,
                    "source": "mujoco_site_oriented_aabb",
                    "support_surface_candidate": True,
                    "containment_capable": logical_id in containment_capable_ids,
                }
            except Exception:
                continue
    return result


def _safe_bool_call(obj: Any, method: str, other: Any) -> dict[str, Any]:
    """Call a capability-dependent pair predicate without making errors false."""

    try:
        value = getattr(obj, method)(other)
    except Exception as exc:
        return {
            "value": None,
            "valid": 0,
            "status": "error",
            "definition_version": f"libero_wrapper_{method}",
            "error": f"{type(exc).__name__}: {exc}"[:240],
        }
    if isinstance(value, bool):
        return {
            "value": value,
            "valid": 1,
            "status": "available",
            "definition_version": f"libero_wrapper_{method}",
        }
    return {
        "value": None,
        "valid": 0,
        "status": "unsupported_return_type",
        "definition_version": f"libero_wrapper_{method}",
    }


def _semantic_pair_records(env: Any, object_states: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Capture directed ``on`` and ``inside`` records when wrappers support them."""

    raw_env = getattr(env, "env", env)
    wrappers = getattr(raw_env, "object_states_dict", {}) or {}
    logical_ids = [
        str(entry["logical_id"])
        for entry in object_states
        if str(entry.get("logical_id")) in wrappers
    ]
    records: dict[str, dict[str, dict[str, Any]]] = {"on": {}, "inside": {}}
    for source_id in logical_ids:
        for target_id in logical_ids:
            if source_id == target_id:
                continue
            source = wrappers[source_id]
            target = wrappers[target_id]
            key = f"{source_id}->{target_id}"
            records["on"][key] = _safe_bool_call(source, "check_ontop", target)
            # ``target.check_contain(source)`` follows the graph direction:
            # source is inside target.  The reverse call is intentionally not
            # used because it would silently swap the relation semantics.
            records["inside"][key] = _safe_bool_call(target, "check_contain", source)
    return records


def bounded_probe_action(step: int, action_dim: int, amplitude: float) -> list[float]:
    """Create a deterministic, bounded nonzero probe without random actions."""

    action = np.zeros(action_dim, dtype=np.float32)
    if action_dim > 0:
        action[0] = amplitude * math.sin(0.31 * step)
    if action_dim > 1:
        action[1] = amplitude * math.cos(0.23 * step)
    if action_dim > 2:
        action[2] = amplitude * math.sin(0.17 * step + 0.4)
    return action.tolist()


def holding_probe_action(
    step: int,
    action_dim: int,
    eef_position: Sequence[float],
    target_position: Sequence[float],
    approach_steps: int = 32,
    close_steps: int = 16,
    height_offset: float = -0.02,
    lift_height: float = 0.10,
    position_scale: float = 0.06,
) -> list[float]:
    """Create a deterministic OSC-Pose approach, close, and lift action.

    The target is the first task-relevant object position.  The last action
    component follows the LIBERO/robosuite convention used by this runtime:
    ``-1`` opens and ``+1`` closes the gripper.  The helper is intentionally a
    controlled probe, not a policy or a claim of successful manipulation.
    """

    if action_dim < 7:
        raise ValueError("holding_probe_action requires a 7D OSC-Pose action")
    eef = [float(value) for value in eef_position]
    target = [float(value) for value in target_position]
    target[2] += float(height_offset)
    if step >= approach_steps + close_steps:
        target[2] += float(lift_height)
    delta = [
        max(-1.0, min(1.0, (target[index] - eef[index]) / max(position_scale, 1e-6)))
        for index in range(3)
    ]
    action = [0.0] * action_dim
    action[:3] = delta
    action[-1] = 1.0 if step >= approach_steps else -1.0
    return action


def make_trajectory_snapshot(
    env: Any,
    observation: Any,
    step: int,
    task_semantics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    object_states = extract_object_states(env)
    contact_pairs = _contact_pairs(env, object_states)
    return {
        "step": int(step),
        "observation_schema": observation_schema(observation),
        "robot_state": extract_robot_state(observation),
        "object_states": object_states,
        "contact_pairs": contact_pairs,
        "robot_contact_pairs": [
            pair for pair in contact_pairs if "robot0" in pair
        ],
        "geometry": _geometry_boxes(env, object_states, task_semantics),
        "robot_base_pose": _robot_base_pose(env),
        "semantic_relation_records": _semantic_pair_records(env, object_states),
        "task_semantics": dict(task_semantics) if isinstance(task_semantics, Mapping) else {},
    }


def collect_episode(
    suite_name: str,
    task_id: int,
    init_state_id: int,
    seed: int,
    steps: int,
    controller: str,
    amplitude: float,
    camera_height: int,
    camera_width: int,
    action_mode: str = "bounded_probe",
    holding_approach_steps: int = 32,
    holding_close_steps: int = 16,
    holding_height_offset: float = -0.02,
    holding_lift_height: float = 0.10,
) -> dict[str, Any]:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils import get_libero_path

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    task_semantics = _task_semantic_metadata(bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        controller=controller,
        camera_heights=camera_height,
        camera_widths=camera_width,
    )
    try:
        env.seed(seed)
        env.reset()
        init_states = task_suite.get_task_init_states(task_id)
        if init_state_id >= len(init_states):
            raise ValueError(
                f"init_state_id {init_state_id} is out of range; available={len(init_states)}"
            )
        observation = env.set_init_state(init_states[init_state_id])
        action_dim = _infer_action_dim(env)
        snapshots = [make_trajectory_snapshot(env, observation, 0, task_semantics)]
        holding_target = None
        if action_mode == "holding_probe":
            if controller != "OSC_POSE":
                raise ValueError("holding_probe requires controller=OSC_POSE")
            holding_target = next(
                (
                    entry.get("pose", {}).get("position")
                    for entry in snapshots[0]["object_states"]
                    if entry.get("is_object_of_interest")
                    and isinstance(entry.get("pose"), Mapping)
                    and isinstance(entry.get("pose", {}).get("position"), Sequence)
                ),
                None,
            )
            if holding_target is None:
                raise ValueError("holding_probe found no task-relevant object position")
        actions: list[dict[str, Any]] = []
        for step in range(steps):
            current_snapshot = snapshots[-1]
            if action_mode == "holding_probe":
                action = holding_probe_action(
                    step=step,
                    action_dim=action_dim,
                    eef_position=current_snapshot["robot_state"].get("eef_pos", [0.0, 0.0, 0.0]),
                    target_position=holding_target,
                    approach_steps=holding_approach_steps,
                    close_steps=holding_close_steps,
                    height_offset=holding_height_offset,
                    lift_height=holding_lift_height,
                )
            elif action_mode == "bounded_probe":
                action = bounded_probe_action(step + 1, action_dim, amplitude)
            else:
                raise ValueError(f"unsupported action_mode={action_mode!r}")
            observation, reward, done, info = env.step(np.asarray(action, dtype=np.float32))
            actions.append(
                {
                    "from_step": step,
                    "to_step": step + 1,
                    "action": action,
                    "reward": _jsonable(reward),
                    "done": bool(done),
                    "info": _jsonable(info),
                }
            )
            snapshots.append(make_trajectory_snapshot(env, observation, step + 1, task_semantics))
    finally:
        env.close()

    return {
        "episode_id": f"{suite_name}:task{task_id}:init{init_state_id}:seed{seed}",
        "suite": suite_name,
        "task_id": task_id,
        "task_name": task.name,
        "language": task.language,
        "bddl_file": str(bddl_file),
        "task_semantics": task_semantics,
        "seed": seed,
        "init_state_id": init_state_id,
        "controller": controller,
        "simulator": {"action_dim": action_dim},
        "action_probe": {
            "type": "holding_approach_close_lift"
            if action_mode == "holding_probe"
            else "bounded_deterministic_nonzero_probe",
            "mode": action_mode,
            "amplitude": amplitude,
            "action_dim": action_dim,
            "controller": controller,
            "holding_approach_steps": holding_approach_steps,
            "holding_close_steps": holding_close_steps,
            "holding_height_offset": holding_height_offset,
            "holding_lift_height": holding_lift_height,
        },
        "snapshots": snapshots,
        "actions": actions,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--controller", default="JOINT_POSITION")
    parser.add_argument("--amplitude", type=float, default=0.025)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--action-mode", choices=["bounded_probe", "holding_probe"], default="bounded_probe")
    parser.add_argument("--holding-approach-steps", type=int, default=32)
    parser.add_argument("--holding-close-steps", type=int, default=16)
    parser.add_argument("--holding-height-offset", type=float, default=-0.02)
    parser.add_argument("--holding-lift-height", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=Path("data/phase1b_trajectory_pilot.json"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    episodes = [
        collect_episode(
            suite_name=args.suite,
            task_id=args.task_id,
            init_state_id=init_state_id,
            seed=args.seed,
            steps=args.steps,
            controller=args.controller,
            amplitude=args.amplitude,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            action_mode=args.action_mode,
            holding_approach_steps=args.holding_approach_steps,
            holding_close_steps=args.holding_close_steps,
            holding_height_offset=args.holding_height_offset,
            holding_lift_height=args.holding_lift_height,
        )
        for init_state_id in args.init_state_ids
    ]
    payload = {
        "capture_status": "live_oracle_pilot",
        "source": "LIBERO simulator; controlled nonzero action probe",
        "privileged_state": True,
        "episodes": episodes,
        "notes": [
            "This pilot verifies action-bearing temporal alignment, not manipulation success.",
            "Raw world pose is retained for audit; model-facing Phase 2R graphs transform it to robot_base.",
            "logical_id is the cross-snapshot alignment key; body_id is audit metadata only.",
            "on/inside wrapper errors remain unknown; open/close are captured as node predicates.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"episodes": len(episodes), "steps_per_episode": args.steps}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
