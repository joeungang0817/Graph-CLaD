"""Validate that stored HDF5 action rows align with the next stored state.

This is a runtime QA tool, not a label builder.  It restores ``state[t]``,
steps exactly ``action[t]`` once, and compares the simulator state with the
stored ``state[t+1]``.  The tolerance is selected from train/validation smoke
only and then frozen before test rows are interpreted.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .io import atomic_json, load_json_config


def max_abs_state_error(actual: Any, expected: Any) -> float:
    """Return a finite max-absolute error for two flattened state vectors."""

    try:
        import numpy as np

        actual_array = np.asarray(actual, dtype=float).reshape(-1)
        expected_array = np.asarray(expected, dtype=float).reshape(-1)
    except Exception as exc:  # pragma: no cover - depends on optional numpy
        raise ValueError(f"state values cannot be converted to numeric vectors: {exc}") from exc
    if actual_array.shape != expected_array.shape:
        raise ValueError(
            f"state shape mismatch: actual={actual_array.shape}, expected={expected_array.shape}"
        )
    if not np.isfinite(actual_array).all() or not np.isfinite(expected_array).all():
        raise ValueError("state vectors contain NaN or Inf")
    return float(np.max(np.abs(actual_array - expected_array), initial=0.0))


def _demo_keys(handle: Any) -> list[str]:
    keys = [str(key) for key in handle["data"].keys() if str(key).startswith("demo_")]
    return sorted(keys, key=lambda value: int(value.rsplit("_", 1)[-1]))


def _action_row(value: Any, dimension: int = 7) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != dimension:
        raise ValueError(f"action must be a {dimension}-vector")
    row = [float(item) for item in value]
    if not all(math.isfinite(item) for item in row):
        raise ValueError("action contains NaN or Inf")
    return row


def _probe_steps(frame_count: int, max_steps: int) -> list[int]:
    if frame_count < 2 or max_steps <= 0:
        return []
    return list(range(min(frame_count - 1, max_steps)))


def validate_hdf5_action_timing(
    hdf5_path: Path,
    bddl_roots: Sequence[Path],
    *,
    task_id: int | None = None,
    max_steps_per_demo: int = 3,
    tolerance: float | None = None,
    render: bool = False,
) -> dict[str, Any]:
    """Run state/action timing checks for one official HDF5 source."""

    if max_steps_per_demo <= 0:
        raise ValueError("max_steps_per_demo must be positive")
    import h5py

    from scripts.phase2d.state_replay import make_environment

    # Import private helpers only here, so importing this module remains
    # possible on a laptop without LIBERO installed.
    from scripts.phase2d.state_replay import _restore_observation

    environment, bddl_path = make_environment(hdf5_path, bddl_roots, render=render)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        with h5py.File(hdf5_path, "r") as handle:
            for demo_key in _demo_keys(handle):
                group = handle["data"][demo_key]
                states = group["states"][:]
                actions = group["actions"][:]
                if len(states) < 2 or len(actions) < 1:
                    errors.append(f"{demo_key}: requires at least two states and one action")
                    continue
                for step in _probe_steps(min(len(states), len(actions) + 1), max_steps_per_demo):
                    try:
                        _restore_observation(environment, states[step])
                        environment.step(_action_row(actions[step]))
                        actual = environment.sim.get_state().flatten()
                        error = max_abs_state_error(actual, states[step + 1])
                        row = {
                            "demo_key": demo_key,
                            "step": int(step),
                            "next_step": int(step + 1),
                            "max_abs_state_error": error,
                            "within_tolerance": None if tolerance is None else bool(error <= tolerance),
                        }
                        rows.append(row)
                    except Exception as exc:
                        errors.append(f"{demo_key}@{step}: {type(exc).__name__}: {exc}")
    finally:
        environment.close()

    measured = [row["max_abs_state_error"] for row in rows]
    report = {
        "contract": "phase3c-action-timing.v1",
        "source_hdf5": str(hdf5_path),
        "source_sha256": _sha256_file(hdf5_path),
        "task_id": task_id,
        "bddl_path": str(bddl_path),
        "render": bool(render),
        "max_steps_per_demo": max_steps_per_demo,
        "frozen_tolerance": tolerance,
        "rows": rows,
        "summary": {
            "checked": len(rows),
            "mean_max_abs_state_error": sum(measured) / len(measured) if measured else None,
            "max_max_abs_state_error": max(measured) if measured else None,
            "within_tolerance": (
                sum(int(row["within_tolerance"]) for row in rows if row["within_tolerance"] is not None)
                if tolerance is not None
                else None
            ),
        },
        "errors": errors,
        "status": "pass" if rows and not errors else "fail",
    }
    return report


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--hdf5", type=Path, action="append")
    parser.add_argument("--bddl-root", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--max-steps-per-demo", type=int, default=3)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config: dict[str, Any] = load_json_config(args.config) if args.config else {}
    hdf5_paths = tuple(args.hdf5 or tuple(Path(value) for value in config.get("hdf5_inputs", [])))
    bddl_roots = tuple(args.bddl_root or tuple(Path(value) for value in config.get("bddl_roots", [])))
    if not hdf5_paths or not bddl_roots:
        raise SystemExit("--hdf5 and --bddl-root are required (directly or in --config)")
    reports = [
        validate_hdf5_action_timing(
            path,
            bddl_roots,
            task_id=args.task_id,
            max_steps_per_demo=int(config.get("max_steps_per_demo", args.max_steps_per_demo)),
            tolerance=args.tolerance if args.tolerance is not None else config.get("tolerance"),
            render=bool(args.render or config.get("render", False)),
        )
        for path in hdf5_paths
    ]
    result = {
        "contract": "phase3c-action-timing-batch.v1",
        "reports": reports,
        "status": "pass" if all(report["status"] == "pass" for report in reports) else "fail",
    }
    with atomic_json(args.output) as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
