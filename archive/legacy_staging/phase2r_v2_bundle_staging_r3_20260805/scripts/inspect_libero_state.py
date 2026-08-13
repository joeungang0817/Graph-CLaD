"""Inspect one LIBERO task's raw observation and privileged simulator state.

This script is intentionally an inspection tool, not a graph builder.  It
records the information needed to freeze the Phase 1 state contract before
Phase 2 chooses the final GraphSpec.

The script imports LIBERO only inside ``main`` so that the pure schema helpers
remain testable in an environment where LIBERO is not installed.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    """Convert numpy-like values and nested containers to JSON-safe values."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _describe(value: Any) -> dict[str, Any]:
    """Describe an observation value without serializing the full image."""

    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "keys": [str(key) for key in value.keys()],
        }

    shape = getattr(value, "shape", None)
    return {
        "kind": "array" if shape is not None else type(value).__name__,
        "shape": list(shape) if shape is not None else None,
        "dtype": str(getattr(value, "dtype", type(value).__name__)),
    }


def observation_schema(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact schema for a raw LIBERO observation dictionary."""

    return {
        str(key): _describe(value)
        for key, value in sorted(observation.items(), key=lambda item: str(item[0]))
    }


def _first_present(observation: Mapping[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in observation:
            return observation[name]
    return None


def extract_robot_state(
    observation: Mapping[str, Any], robot_prefix: str = "robot0"
) -> dict[str, Any]:
    """Extract raw robot fields when the standard robosuite keys are present.

    Missing keys are kept as ``None``.  This is deliberate: the inspection
    result should reveal the environment's actual contract rather than silently
    substitute a guessed state vector.
    """

    prefix = robot_prefix.rstrip("_")
    aliases = {
        "eef_pos": [f"{prefix}_eef_pos", f"{prefix}_eef_position"],
        "eef_quat": [f"{prefix}_eef_quat", f"{prefix}_eef_quaternion"],
        "gripper_qpos": [
            f"{prefix}_gripper_qpos",
            f"{prefix}_gripper_pos",
        ],
        "joint_pos": [f"{prefix}_joint_pos", f"{prefix}_joint_qpos"],
        "joint_vel": [f"{prefix}_joint_vel", f"{prefix}_joint_qvel"],
        "eef_vel_lin": [f"{prefix}_eef_vel_lin", f"{prefix}_eef_vel"],
        "eef_vel_ang": [f"{prefix}_eef_vel_ang"],
    }

    fields = {
        name: _jsonable(_first_present(observation, candidates))
        for name, candidates in aliases.items()
    }
    fields["available_keys"] = sorted(
        str(key)
        for key in observation
        if str(key).startswith(prefix)
    )
    fields["representation_status"] = "raw_observation_fields; not normalized"
    return fields


def _unwrap_env(env: Any) -> Any:
    """Return the underlying LIBERO task environment from its wrapper."""

    return getattr(env, "env", env)


def _safe_call(obj: Any, method_name: str) -> Any:
    method = getattr(obj, method_name, None)
    if method is None:
        return None
    try:
        return _jsonable(method())
    except Exception as exc:  # state predicates can be object-specific
        return {"error": f"{type(exc).__name__}: {exc}"}


def _safe_index(array: Any, index: int) -> Any:
    try:
        return _jsonable(array[index])
    except Exception:
        return None


def _object_names(raw_env: Any) -> list[str]:
    names: set[str] = set()
    for attr in ("objects_dict", "fixtures_dict", "object_sites_dict", "object_states_dict"):
        values = getattr(raw_env, attr, None)
        if values:
            names.update(str(name) for name in values.keys())
    return sorted(names)


def extract_object_states(env: Any) -> list[dict[str, Any]]:
    """Extract object / fixture / site states from a live LIBERO wrapper.

    ``logical_id`` is the stable identity used by the future graph.  The
    ``body_id`` is only a runtime MuJoCo index and must not be used as the
    identity across episodes or tasks.
    """

    raw_env = _unwrap_env(env)
    sim = getattr(env, "sim", getattr(raw_env, "sim", None))
    data = getattr(sim, "data", None)
    obj_body_id = getattr(raw_env, "obj_body_id", {}) or {}
    object_sites = getattr(raw_env, "object_sites_dict", {}) or {}
    state_wrappers = getattr(raw_env, "object_states_dict", {}) or {}
    objects = getattr(raw_env, "objects_dict", {}) or {}
    fixtures = getattr(raw_env, "fixtures_dict", {}) or {}
    object_of_interest = set(getattr(raw_env, "obj_of_interest", []) or [])

    result: list[dict[str, Any]] = []
    for name in _object_names(raw_env):
        if name in objects:
            node_type = "object"
        elif name in fixtures:
            node_type = "fixture"
        elif name in object_sites:
            node_type = "site"
        else:
            node_type = "unknown"

        state_wrapper = state_wrappers.get(name)
        pose = None
        quaternion_convention = None
        if state_wrapper is not None:
            geom_state = _safe_call(state_wrapper, "get_geom_state")
            if isinstance(geom_state, Mapping):
                pose = {
                    "position": geom_state.get("pos"),
                    "quaternion": geom_state.get("quat"),
                }
                quaternion_convention = (
                    "LIBERO state-wrapper output; verify ordering before normalization"
                )

        body_id = obj_body_id.get(name)
        if body_id is not None and data is not None:
            body_id = int(body_id)
            pose = pose or {}
            pose.setdefault("position", _safe_index(getattr(data, "body_xpos", []), body_id))
            pose.setdefault("quaternion", _safe_index(getattr(data, "body_xquat", []), body_id))
            quaternion_convention = quaternion_convention or "MuJoCo body_xquat raw ordering"
        elif name in object_sites and data is not None:
            site_data = object_sites[name]
            site_name = getattr(site_data, "name", name)
            try:
                site_pos = _jsonable(data.get_site_xpos(site_name))
                site_mat = _jsonable(data.get_site_xmat(site_name))
                pose = pose or {}
                pose.setdefault("position", site_pos)
                pose.setdefault("rotation_matrix", site_mat)
            except Exception:
                pass

        entry: dict[str, Any] = {
            "logical_id": name,
            "node_type": node_type,
            "is_object_of_interest": name in object_of_interest,
            "body_id": body_id,
            "pose": pose,
            "quaternion_convention": quaternion_convention,
            "joint_state": _safe_call(state_wrapper, "get_joint_state")
            if state_wrapper is not None
            else None,
            "is_open": _safe_call(state_wrapper, "is_open")
            if state_wrapper is not None
            else None,
            "is_close": _safe_call(state_wrapper, "is_close")
            if state_wrapper is not None
            else None,
        }
        result.append(entry)

    return result


def _sim_metadata(env: Any) -> dict[str, Any]:
    raw_env = _unwrap_env(env)
    sim = getattr(env, "sim", getattr(raw_env, "sim", None))
    model = getattr(sim, "model", None)
    sim_state = None
    get_state = getattr(sim, "get_state", None)
    if get_state is not None:
        try:
            sim_state = _jsonable(get_state().flatten())
        except Exception:
            sim_state = None

    metadata: dict[str, Any] = {
        "flat_state_dim": len(sim_state) if isinstance(sim_state, list) else None,
        "model_nbody": getattr(model, "nbody", None),
        "model_nsite": getattr(model, "nsite", None),
        "model_ngeom": getattr(model, "ngeom", None),
        "model_njnt": getattr(model, "njnt", None),
    }
    action_dim = getattr(raw_env, "action_dim", getattr(env, "action_dim", None))
    if action_dim is None:
        action_spec = getattr(raw_env, "action_spec", getattr(env, "action_spec", None))
        if action_spec is not None:
            try:
                action_dim = len(action_spec[0])
            except Exception:
                action_dim = None
    metadata["action_dim"] = int(action_dim) if action_dim is not None else None
    return metadata


def make_snapshot(env: Any, observation: Mapping[str, Any], step: int, robot_prefix: str) -> dict[str, Any]:
    return {
        "step": step,
        "observation_schema": observation_schema(observation),
        "robot_state": extract_robot_state(observation, robot_prefix=robot_prefix),
        "object_states": extract_object_states(env),
    }


def _infer_action_dim(env: Any) -> int:
    metadata_dim = _sim_metadata(env).get("action_dim")
    if metadata_dim is not None:
        return int(metadata_dim)
    # Standard LIBERO single-arm examples use a 7-dimensional action.
    return 7


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--robot-prefix", default="robot0")
    parser.add_argument(
        "--controller",
        default="JOINT_POSITION",
        help="Diagnostic controller; state inspection does not require IK or OSC control.",
    )
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/phase1_libero_state_capture.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    try:
        import numpy as np
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero.utils import get_libero_path
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "LIBERO is not installed in this runtime. Install the pinned LIBERO "
            "environment first, then rerun this inspection script."
        ) from exc

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.suite not in benchmark_dict:
        raise SystemExit(f"Unknown LIBERO suite {args.suite!r}; available: {sorted(benchmark_dict)}")
    task_suite = benchmark_dict[args.suite]()
    task = task_suite.get_task(args.task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        controller=args.controller,
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
    )
    try:
        env.seed(args.seed)
        observation = env.reset()
        init_states = task_suite.get_task_init_states(args.task_id)
        if args.init_state_id >= len(init_states):
            raise SystemExit(
                f"init-state-id {args.init_state_id} is out of range for "
                f"{len(init_states)} initial states"
            )
        observation = env.set_init_state(init_states[args.init_state_id])

        snapshots = [
            make_snapshot(env, observation, step=0, robot_prefix=args.robot_prefix)
        ]
        action_dim = _infer_action_dim(env)
        for step in range(1, args.steps + 1):
            action = np.zeros(action_dim, dtype=np.float32)
            observation, reward, done, info = env.step(action)
            snapshots.append(
                make_snapshot(env, observation, step=step, robot_prefix=args.robot_prefix)
            )

        payload = {
            "capture_status": "live",
            "source": "LIBERO simulator inspection",
            "suite": args.suite,
            "task_id": args.task_id,
            "task_name": task.name,
            "language": task.language,
            "bddl_file": str(bddl_file),
            "seed": args.seed,
            "init_state_id": args.init_state_id,
            "simulator": _sim_metadata(env),
            "snapshots": snapshots,
            "notes": [
                "logical_id is the graph identity; body_id is a runtime MuJoCo index",
                "raw quaternion ordering must be verified before normalization",
                "this file records privileged simulator state and must not be used as an RGB-perception claim",
            ],
            "last_step": {"reward": _jsonable(reward), "done": bool(done), "info": _jsonable(info)},
        }
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(json.dumps({"task": payload["task_name"], "snapshots": len(snapshots)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
