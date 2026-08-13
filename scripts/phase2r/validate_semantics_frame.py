"""Audit LIBERO semantic predicate APIs and robot-base frame stability."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PAIRS = [
    ("akita_black_bowl_1", "plate_1"),
    ("akita_black_bowl_1", "main_table_plate_region"),
    ("plate_1", "main_table_plate_region"),
    ("akita_black_bowl_1", "main_table_between_plate_ramekin_region"),
    ("flat_stove_1", "flat_stove_1_cook_region"),
]


def _signature(obj: Any, method: str) -> str | None:
    try:
        return str(inspect.signature(getattr(obj, method)))
    except Exception:
        return None


def _call(obj: Any, method: str, other: Any | None = None) -> dict[str, Any]:
    try:
        value = getattr(obj, method)() if other is None else getattr(obj, method)(other)
        return {"status": "available", "value": value, "type": type(value).__name__}
    except Exception as exc:
        return {
            "status": "error",
            "value": None,
            "type": type(exc).__name__,
            "error": str(exc)[:240],
        }


def _env_for_task(task_suite: Any, task_id: int, controller: str, height: int, width: int) -> tuple[Any, Any]:
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils import get_libero_path

    task = task_suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    return (
        OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            controller=controller,
            camera_heights=height,
            camera_widths=width,
        ),
        task,
    )


def audit(
    suite_name: str,
    task_id: int,
    init_state_ids: list[int],
    seed: int,
    controller: str,
    camera_height: int,
    camera_width: int,
) -> dict[str, Any]:
    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    first_env, _ = _env_for_task(task_suite, task_id, controller, camera_height, camera_width)
    try:
        first_env.seed(seed)
        first_env.reset()
        first_env.set_init_state(task_suite.get_task_init_states(task_id)[init_state_ids[0]])
        raw_env = getattr(first_env, "env", first_env)
        wrappers = getattr(raw_env, "object_states_dict", {})
        signatures = {}
        for name in sorted(set(sum(([a, b] for a, b in DEFAULT_PAIRS), [])) & set(wrappers)):
            obj = wrappers[name]
            signatures[name] = {
                "class": type(obj).__name__,
                "object_state_type": getattr(obj, "object_state_type", None),
                "methods": {
                    method: _signature(obj, method)
                    for method in [
                        "check_contact",
                        "check_contain",
                        "check_ontop",
                        "is_open",
                        "is_close",
                        "get_joint_state",
                    ]
                    if hasattr(obj, method)
                },
            }
        pair_results = []
        for left, right in DEFAULT_PAIRS:
            if left not in wrappers or right not in wrappers:
                pair_results.append({"left": left, "right": right, "status": "missing_wrapper"})
                continue
            result = {"left": left, "right": right}
            for method in ["check_contact", "check_contain", "check_ontop"]:
                result[method] = _call(wrappers[left], method, wrappers[right])
            pair_results.append(result)
    finally:
        first_env.close()

    base_records = []
    for init_state_id in init_state_ids:
        env, _ = _env_for_task(task_suite, task_id, controller, camera_height, camera_width)
        try:
            env.seed(seed)
            env.reset()
            env.set_init_state(task_suite.get_task_init_states(task_id)[init_state_id])
            body_id = env.sim.model.body_name2id("robot0_base")
            base_records.append(
                {
                    "init_state_id": init_state_id,
                    "position": np.asarray(env.sim.data.body_xpos[body_id]).round(10).tolist(),
                    "quaternion": np.asarray(env.sim.data.body_xquat[body_id]).round(10).tolist(),
                }
            )
        finally:
            env.close()
    positions = np.asarray([record["position"] for record in base_records], dtype=float)
    quaternions = np.asarray([record["quaternion"] for record in base_records], dtype=float)
    position_range = np.ptp(positions, axis=0).round(10).tolist()
    quaternion_range = np.ptp(quaternions, axis=0).round(10).tolist()

    def available_boolean(method: str) -> int:
        return sum(
            int(result.get(method, {}).get("status") == "available" and result[method].get("type") == "bool")
            for result in pair_results
        )

    return {
        "validation_version": "phase2r-semantics-frame.v1",
        "task": {
            "suite": suite_name,
            "task_id": task_id,
            "task_name": task.name,
            "seed": seed,
        },
        "frame": {
            "candidate": "robot_base",
            "body_name": "robot0_base",
            "records": base_records,
            "position_max_range": position_range,
            "quaternion_max_range": quaternion_range,
            "stable_across_init_states": all(value == 0.0 for value in position_range + quaternion_range),
        },
        "semantic_api": {
            "signatures": signatures,
            "pair_results": pair_results,
            "contact_boolean_count": available_boolean("check_contact"),
            "ontop_boolean_count": available_boolean("check_ontop"),
            "contain_boolean_count": available_boolean("check_contain"),
        },
        "gate_results": {
            "robot_base_stable": all(value == 0.0 for value in position_range + quaternion_range),
            "contact_handler_candidate": available_boolean("check_contact") == len(pair_results),
            "generic_ontop_handler_for_all_pairs": available_boolean("check_ontop") == len(pair_results),
            "generic_contain_handler_for_all_pairs": available_boolean("check_contain") == len(pair_results),
            "generic_open_close_handler": False,
        },
        "decision": {
            "primary_frame": "robot_base",
            "use_contact_wrapper": True,
            "use_generic_contain_or_ontop": False,
            "implement_geometry_or_task_specific_semantics": ["on", "inside", "holding", "open", "close"],
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--controller", default="JOINT_POSITION")
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = audit(
        suite_name=args.suite,
        task_id=args.task_id,
        init_state_ids=args.init_state_ids,
        seed=args.seed,
        controller=args.controller,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["gate_results"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
