"""Render a small, auditable Phase 3C camera-orientation contact sheet.

This command uses the same joined manifest, state replay, camera keys, and
normalization flags as the semantic feature-store builder.  It never guesses
which orientation is correct: the output places the configured frame beside
its vertically flipped alternative and records repeated-render digests.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .build_semantic_feature_store import (
    _expand_path,
    _hdf5_demo_group,
    _iter_joined,
    _resolve_hdf5,
    configured_camera_frames,
    frame_digest,
    required_frame_keys,
)
from .io import load_json_config, write_json


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_orientation_samples(
    required: Mapping[tuple[int, str], set[int]],
    *,
    max_tasks: int = 3,
    frames_per_task: int = 3,
) -> list[tuple[int, str, int]]:
    """Select deterministic early/middle/late frames from one demo per task."""

    if int(max_tasks) <= 0 or int(frames_per_task) <= 0:
        raise ValueError("max_tasks and frames_per_task must be positive")
    by_task: dict[int, list[tuple[str, list[int]]]] = defaultdict(list)
    for (task_id, demo_key), steps in required.items():
        ordered = sorted(int(step) for step in steps)
        if ordered:
            by_task[int(task_id)].append((str(demo_key), ordered))
    selected: list[tuple[int, str, int]] = []
    for task_id in sorted(by_task)[: int(max_tasks)]:
        demo_key, steps = sorted(by_task[task_id], key=lambda item: item[0])[0]
        if frames_per_task == 1:
            indices = [len(steps) // 2]
        else:
            indices = np.linspace(0, len(steps) - 1, num=min(frames_per_task, len(steps)))
            indices = [int(round(float(index))) for index in indices]
        for index in dict.fromkeys(indices):
            selected.append((task_id, demo_key, steps[index]))
    if not selected:
        raise ValueError("joined manifest has no frames for orientation QA")
    return selected


def _compose_contact_sheet(
    rows: Sequence[tuple[str, Sequence[np.ndarray]]],
    camera_names: Sequence[str],
) -> Any:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - SSH dependency
        raise RuntimeError("camera orientation QA requires Pillow") from exc
    if not rows:
        raise ValueError("contact sheet requires at least one row")
    first = np.asarray(rows[0][1][0])
    if first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError("contact-sheet frames must be HWC RGB")
    height, width = int(first.shape[0]), int(first.shape[1])
    label_width = 190
    header_height = 42
    row_height = height + 24
    columns = 2 * len(camera_names)
    canvas = Image.new(
        "RGB",
        (label_width + columns * width, header_height + len(rows) * row_height),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "configured vs vertical-flipped", fill=(0, 0, 0))
    for camera_index, name in enumerate(camera_names):
        x = label_width + camera_index * 2 * width
        draw.text((x + 6, 8), f"{name}: configured", fill=(0, 0, 0))
        draw.text((x + width + 6, 8), f"{name}: flipped", fill=(0, 0, 0))
    for row_index, (label, frames) in enumerate(rows):
        if len(frames) != len(camera_names):
            raise ValueError("contact-sheet row does not match camera count")
        y = header_height + row_index * row_height
        draw.text((8, y + 6), label, fill=(0, 0, 0))
        for camera_index, frame in enumerate(frames):
            array = np.asarray(frame)
            if array.shape != first.shape:
                raise ValueError("contact-sheet frames changed shape")
            configured = Image.fromarray(np.ascontiguousarray(array), mode="RGB")
            flipped = Image.fromarray(np.ascontiguousarray(array[::-1]), mode="RGB")
            x = label_width + camera_index * 2 * width
            canvas.paste(configured, (x, y))
            canvas.paste(flipped, (x + width, y))
    return canvas


def run_orientation_qa(
    config: Mapping[str, Any],
    *,
    max_tasks: int = 3,
    frames_per_task: int = 3,
) -> dict[str, Any]:
    store = dict(config.get("semantic_feature_store", config))
    joined_path = _expand_path(str(store["joined_manifest"]))
    output_root = _expand_path(str(store["output_root"]))
    camera_specs = store.get("cameras")
    if not isinstance(camera_specs, Sequence) or isinstance(camera_specs, (str, bytes)):
        raise ValueError("semantic_feature_store.cameras must be a sequence")
    if len(camera_specs) != 2:
        raise ValueError("Phase 3C orientation QA requires exactly two cameras")
    required = required_frame_keys(_iter_joined(joined_path))
    selected = select_orientation_samples(
        required, max_tasks=max_tasks, frames_per_task=frames_per_task
    )
    bddl_roots = [_expand_path(str(value)) for value in store.get("bddl_roots", [])]
    if not bddl_roots:
        raise ValueError("semantic_feature_store.bddl_roots must not be empty")

    try:
        import h5py
        import robosuite.macros as macros
        from scripts.phase2d.state_replay import _restore_observation, make_environment
    except ImportError as exc:  # pragma: no cover - SSH-only path
        raise RuntimeError("orientation QA requires h5py, LIBERO, and robosuite") from exc

    rendered_rows: list[tuple[str, Sequence[np.ndarray]]] = []
    records: list[dict[str, Any]] = []
    recorded_conventions: dict[str, str | None] = {}
    by_demo: dict[tuple[int, str], list[int]] = defaultdict(list)
    for task_id, demo_key, step in selected:
        by_demo[(task_id, demo_key)].append(step)
    for (task_id, demo_key), steps in sorted(by_demo.items()):
        hdf5_path = _resolve_hdf5(store, task_id)
        environment, _ = make_environment(
            hdf5_path,
            bddl_roots,
            camera_size=int(store.get("camera_size", 224)),
            render=True,
        )
        try:
            with h5py.File(hdf5_path, "r") as handle:
                raw_convention = handle["data"].attrs.get("macros_image_convention")
                if isinstance(raw_convention, bytes):
                    raw_convention = raw_convention.decode("utf-8")
                recorded_conventions[str(task_id)] = (
                    str(raw_convention) if raw_convention is not None else None
                )
                states = _hdf5_demo_group(handle, demo_key)["states"]
                for step in sorted(steps):
                    if step < 0 or step >= int(states.shape[0]):
                        raise IndexError(f"orientation QA step out of range: {task_id}/{demo_key}/{step}")
                    first_observation = _restore_observation(environment, states[step])
                    first_frames, _ = configured_camera_frames(first_observation, camera_specs)
                    second_observation = _restore_observation(environment, states[step])
                    second_frames, _ = configured_camera_frames(second_observation, camera_specs)
                    first_digests = [frame_digest(frame) for frame in first_frames]
                    second_digests = [frame_digest(frame) for frame in second_frames]
                    rendered_rows.append(
                        (f"task{task_id} {demo_key} step{step}", first_frames)
                    )
                    records.append(
                        {
                            "task_id": task_id,
                            "demo_key": demo_key,
                            "step": step,
                            "configured_sha256": first_digests,
                            "repeat_sha256": second_digests,
                            "repeat_match": first_digests == second_digests,
                        }
                    )
        finally:
            close = getattr(environment, "close", None)
            if callable(close):
                close()

    qa_root = output_root / "qa"
    qa_root.mkdir(parents=True, exist_ok=True)
    contact_sheet_path = qa_root / "orientation_contact_sheet.png"
    sheet = _compose_contact_sheet(
        rendered_rows,
        [str(spec.get("name", spec.get("key", "camera"))) for spec in camera_specs],
    )
    sheet.save(contact_sheet_path)
    determinism = {
        "schema": "phase3c-camera-orientation-qa.v1",
        "status": "pass" if all(row["repeat_match"] for row in records) else "fail",
        "review_status": "pending_human_orientation_choice",
        "joined_manifest": str(joined_path),
        "runtime_image_convention": str(macros.IMAGE_CONVENTION),
        "recorded_image_convention_by_task": recorded_conventions,
        "camera_specs": [dict(spec) for spec in camera_specs],
        "rows": records,
        "contact_sheet": str(contact_sheet_path),
        "contact_sheet_sha256": _sha256_file(contact_sheet_path),
    }
    report_path = qa_root / "determinism.json"
    write_json(report_path, determinism)
    return determinism


def main() -> None:  # pragma: no cover - SSH CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=3)
    parser.add_argument("--frames-per-task", type=int, default=3)
    args = parser.parse_args()
    result = run_orientation_qa(
        load_json_config(args.config),
        max_tasks=args.max_tasks,
        frames_per_task=args.frames_per_task,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
