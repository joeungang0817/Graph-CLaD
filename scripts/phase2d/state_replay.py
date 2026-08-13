"""Exact LIBERO HDF5 state replay used to create Phase 2D oracle labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hdf5_environment_metadata(path: Path) -> dict[str, Any]:
    """Read environment metadata without replaying actions."""

    import h5py

    with h5py.File(path, "r") as handle:
        data = handle["data"]
        raw_args = data.attrs["env_args"]
        raw_bddl = data.attrs.get("bddl_file_name")
    if isinstance(raw_args, bytes):
        raw_args = raw_args.decode("utf-8")
    if isinstance(raw_bddl, bytes):
        raw_bddl = raw_bddl.decode("utf-8")
    env_args = json.loads(raw_args)
    return {"env_args": env_args, "canonical_bddl": str(raw_bddl) if raw_bddl else None}


def resolve_bddl_path(name_or_path: str, roots: Iterable[Path]) -> Path:
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate
    matches: list[Path] = []
    for root in roots:
        if root.exists():
            matches.extend(root.rglob(candidate.name))
    if not matches:
        raise FileNotFoundError(f"BDDL file not found: {candidate.name}")
    return sorted(matches)[0]


def make_environment(
    hdf5_path: Path,
    bddl_roots: Iterable[Path],
    camera_size: int = 128,
    render: bool = False,
) -> tuple[Any, Path]:
    """Restore the environment configuration stored with an official demo."""

    from libero.libero.envs import OffScreenRenderEnv

    metadata = hdf5_environment_metadata(hdf5_path)
    kwargs = dict(metadata["env_args"]["env_kwargs"])
    if metadata["canonical_bddl"]:
        kwargs["bddl_file_name"] = metadata["canonical_bddl"]
    controller = dict(kwargs.pop("controller_configs", {}) or {})
    bddl_path = resolve_bddl_path(str(kwargs.pop("bddl_file_name")), bddl_roots)
    kwargs.update(
        camera_heights=camera_size,
        camera_widths=camera_size,
        has_offscreen_renderer=render,
        use_camera_obs=render,
    )
    environment = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path), controller=controller.get("type", "OSC_POSE"), **kwargs
    )
    environment.reset()
    return environment, bddl_path


def _restore_observation(environment: Any, state: Any) -> Any:
    if hasattr(environment, "regenerate_obs_from_state"):
        return environment.regenerate_obs_from_state(state)
    observation = environment.set_init_state(state)
    simulation = getattr(environment, "sim", None)
    if simulation is not None and hasattr(simulation, "forward"):
        simulation.forward()
    return observation


def _state_error(environment: Any, state: Any) -> float | None:
    try:
        import numpy as np

        restored = np.asarray(environment.sim.get_state().flatten(), dtype=float)
        expected = np.asarray(state, dtype=float)
        return float(np.max(np.abs(restored - expected)))
    except Exception:
        return None


def replay_demo_group(
    environment: Any,
    demo_group: Any,
    task_semantics: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[list[float]], float]:
    """Restore every saved simulator state and extract oracle snapshots.

    Stored actions are returned only as model-conditioning inputs.  They are
    never stepped through the simulator to create labels.
    """

    import numpy as np

    from scripts.phase2r.collect_libero_trajectory import make_trajectory_snapshot

    states = demo_group["states"][:]
    actions = demo_group["actions"][:]
    snapshots: list[dict[str, Any]] = []
    maximum_error = 0.0
    for step, state in enumerate(states):
        observation = _restore_observation(environment, state)
        snapshot = make_trajectory_snapshot(
            environment, observation, step, task_semantics=dict(task_semantics or {})
        )
        snapshots.append(snapshot)
        error = _state_error(environment, state)
        if error is not None:
            maximum_error = max(maximum_error, error)
    action_rows = [np.asarray(row, dtype=float).tolist() for row in actions]
    return snapshots, action_rows, maximum_error

