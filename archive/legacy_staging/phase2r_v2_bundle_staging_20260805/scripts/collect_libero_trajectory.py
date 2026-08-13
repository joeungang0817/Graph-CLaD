"""Collect controlled, action-bearing LIBERO oracle trajectories.

This is the Phase 1b pilot collector. It deliberately records privileged
simulator state and a small bounded nonzero action probe; it is not an RGB
perception result or a demonstration-quality policy dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

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
    """Map current MuJoCo contacts to logical IDs, including robot contacts."""

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
                    if body_name.startswith("robot0"):
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


def make_trajectory_snapshot(env: Any, observation: Any, step: int) -> dict[str, Any]:
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
        "robot_base_pose": _robot_base_pose(env),
        "semantic_relation_records": _semantic_pair_records(env, object_states),
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
) -> dict[str, Any]:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils import get_libero_path

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
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
        snapshots = [make_trajectory_snapshot(env, observation, 0)]
        actions: list[dict[str, Any]] = []
        for step in range(steps):
            action = bounded_probe_action(step + 1, action_dim, amplitude)
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
            snapshots.append(make_trajectory_snapshot(env, observation, step + 1))
    finally:
        env.close()

    return {
        "episode_id": f"{suite_name}:task{task_id}:init{init_state_id}:seed{seed}",
        "suite": suite_name,
        "task_id": task_id,
        "task_name": task.name,
        "language": task.language,
        "bddl_file": str(bddl_file),
        "seed": seed,
        "init_state_id": init_state_id,
        "controller": controller,
        "simulator": {"action_dim": action_dim},
        "action_probe": {
            "type": "bounded_deterministic_nonzero_probe",
            "amplitude": amplitude,
            "action_dim": action_dim,
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
