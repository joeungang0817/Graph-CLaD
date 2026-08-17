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
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .io import atomic_json, load_json_config


def _path(value: Any) -> Path:
    raw = os.path.expandvars(str(value))
    if "$" in raw or "%" in raw:
        raise ValueError(f"unresolved environment variable in path: {value}")
    return Path(raw).expanduser()


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


def _probe_steps(frame_count: int, max_steps: int, policy: str = "uniform") -> list[int]:
    if frame_count < 2 or max_steps <= 0:
        return []
    available = frame_count - 1
    count = min(available, max_steps)
    if policy == "head":
        return list(range(count))
    if policy != "uniform":
        raise ValueError(f"unsupported probe policy: {policy}")
    if count == 1:
        return [0]
    return [round(index * (available - 1) / (count - 1)) for index in range(count)]


def action_timing_status(
    rows: Sequence[Mapping[str, Any]], errors: Sequence[str], tolerance: float | None
) -> str:
    """Fail when any checked transition exceeds a configured tolerance."""

    if not rows or errors:
        return "fail"
    if tolerance is None:
        return "pass"
    return "pass" if all(row.get("within_tolerance") is True for row in rows) else "fail"


def validate_hdf5_action_timing(
    hdf5_path: Path,
    bddl_roots: Sequence[Path],
    *,
    task_id: int | None = None,
    max_steps_per_demo: int = 3,
    probe_policy: str = "uniform",
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
                for step in _probe_steps(
                    min(len(states), len(actions) + 1), max_steps_per_demo, probe_policy
                ):
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
    within_count = (
        sum(int(row["within_tolerance"] is True) for row in rows)
        if tolerance is not None
        else None
    )
    report = {
        "contract": "phase3c-action-timing.v2",
        "source_hdf5": str(hdf5_path),
        "source_sha256": _sha256_file(hdf5_path),
        "task_id": task_id,
        "bddl_path": str(bddl_path),
        "render": bool(render),
        "max_steps_per_demo": max_steps_per_demo,
        "probe_policy": probe_policy,
        "frozen_tolerance": tolerance,
        "rows": rows,
        "summary": {
            "checked": len(rows),
            "mean_max_abs_state_error": sum(measured) / len(measured) if measured else None,
            "max_max_abs_state_error": max(measured) if measured else None,
            "within_tolerance": within_count,
            "outside_tolerance": len(rows) - within_count if within_count is not None else None,
            "initial_step_outside_tolerance": (
                sum(int(row["step"] == 0 and row["within_tolerance"] is False) for row in rows)
                if tolerance is not None else None
            ),
            "non_initial_step_outside_tolerance": (
                sum(int(row["step"] != 0 and row["within_tolerance"] is False) for row in rows)
                if tolerance is not None else None
            ),
        },
        "errors": errors,
        "status": action_timing_status(rows, errors, tolerance),
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
    parser.add_argument("--probe-policy", choices=("uniform", "head"), default="uniform")
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config: dict[str, Any] = load_json_config(args.config) if args.config else {}
    hdf5_paths = tuple(_path(value) for value in (args.hdf5 or config.get("hdf5_inputs", [])))
    bddl_roots = tuple(_path(value) for value in (args.bddl_root or config.get("bddl_roots", [])))
    if not hdf5_paths or not bddl_roots:
        raise SystemExit("--hdf5 and --bddl-root are required (directly or in --config)")
    reports = [
        validate_hdf5_action_timing(
            path,
            bddl_roots,
            task_id=args.task_id,
            max_steps_per_demo=int(config.get("max_steps_per_demo", args.max_steps_per_demo)),
            probe_policy=str(config.get("probe_policy", args.probe_policy)),
            tolerance=args.tolerance if args.tolerance is not None else config.get("tolerance"),
            render=bool(args.render or config.get("render", False)),
        )
        for path in hdf5_paths
    ]
    result = {
        "contract": "phase3c-action-timing-batch.v2",
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
