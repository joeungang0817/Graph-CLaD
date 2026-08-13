"""Collect a controlled multi-task Phase 2R oracle scale-up capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .collect_libero_trajectory import collect_episode
except ImportError:  # pragma: no cover - CLI execution from scripts/
    from collect_libero_trajectory import collect_episode


def collect_scaleup(
    suite_name: str,
    task_ids: list[int],
    init_state_ids: list[int],
    seed: int,
    steps: int,
    controller: str,
    amplitude: float,
    camera_height: int,
    camera_width: int,
) -> dict[str, Any]:
    """Collect the same bounded probe for each task/init-state pair."""

    episodes = []
    for task_id in task_ids:
        for init_state_id in init_state_ids:
            episodes.append(
                collect_episode(
                    suite_name=suite_name,
                    task_id=task_id,
                    init_state_id=init_state_id,
                    seed=seed,
                    steps=steps,
                    controller=controller,
                    amplitude=amplitude,
                    camera_height=camera_height,
                    camera_width=camera_width,
                )
            )
    return {
        "capture_status": "live_oracle_scaleup",
        "source": "LIBERO simulator; controlled bounded nonzero action probe",
        "privileged_state": True,
        "scaleup_protocol": {
            "suite": suite_name,
            "task_ids": task_ids,
            "init_state_ids": init_state_ids,
            "seed": seed,
            "steps": steps,
            "controller": controller,
            "amplitude": amplitude,
        },
        "episodes": episodes,
        "notes": [
            "This scale-up checks frame stability and relation coverage, not policy success.",
            "Raw world pose is retained for audit; the builder creates robot_base graphs.",
            "Unsupported predicates remain unknown and are not converted to false.",
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--init-state-ids", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--controller", default="JOINT_POSITION")
    parser.add_argument("--amplitude", type=float, default=0.025)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = collect_scaleup(
        suite_name=args.suite,
        task_ids=args.task_ids,
        init_state_ids=args.init_state_ids,
        seed=args.seed,
        steps=args.steps,
        controller=args.controller,
        amplitude=args.amplitude,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "episode_count": len(payload["episodes"]),
        "task_ids": args.task_ids,
        "init_state_ids": args.init_state_ids,
        "steps_per_episode": args.steps,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

