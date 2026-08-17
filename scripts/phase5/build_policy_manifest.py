"""Build a causal Stage 2 policy manifest with future actions as labels only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from scripts.phase3c.contracts import canonical_sha256, parse_action_window
from scripts.phase3c.io import (
    atomic_jsonl,
    iter_json_objects,
    iter_phase2d_samples,
    load_json_config,
    write_json,
    write_json_line,
)


SCHEMA = "phase5-policy-sample.v1"


def _path(value: Any) -> Path:
    raw = os.path.expandvars(str(value))
    if "$" in raw:
        raise ValueError(f"unresolved environment variable in path: {value}")
    return Path(raw).expanduser()


def _iter_joined(path: Path):
    for value in iter_json_objects(path):
        if not isinstance(value, Mapping):
            raise ValueError(f"joined manifest row is not an object: {path}")
        yield value


def _sample_key(task_id: int, episode_id: Any, start_step: Any) -> tuple[int, str, int]:
    return int(task_id), str(episode_id), int(start_step)


def _source_action_index(
    paths_by_task: Mapping[str, Any],
    needed: set[tuple[int, str, int]],
    *,
    tau: int,
) -> dict[tuple[int, str, int], tuple[str, list[list[float]]]]:
    found: dict[tuple[int, str, int], tuple[str, list[list[float]]]] = {}
    for raw_task, raw_path in paths_by_task.items():
        task_id = int(raw_task)
        path = _path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        for sample in iter_phase2d_samples([path]):
            if int(sample.get("tau", -1)) != int(tau):
                continue
            episode_id = sample.get("episode_id")
            start_step = sample.get("start_step")
            if episode_id is None or start_step is None:
                continue
            key = _sample_key(task_id, episode_id, start_step)
            if key not in needed:
                continue
            action = parse_action_window(sample.get("action_window"), tau=tau)
            split = str(sample.get("split"))
            previous = found.get(key)
            if previous is not None and previous != (split, action):
                raise ValueError(f"conflicting action labels for source key {key}")
            found[key] = (split, action)
    missing = sorted(needed - set(found))
    if missing:
        raise ValueError(f"future action labels are missing for {len(missing)} joined rows: {missing[:3]}")
    return found


def build_policy_manifest(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("stage2_policy_manifest", config)
    if not isinstance(section, Mapping):
        raise ValueError("stage2_policy_manifest config must be an object")
    joined_path = _path(section["joined_manifest"])
    output_path = _path(section["output"])
    qa_path = _path(section.get("qa_output", str(output_path) + ".qa.json"))
    if output_path.exists() or qa_path.exists():
        raise FileExistsError("Stage 2 policy artifacts are immutable; use a new versioned path")
    tau = int(section.get("tau", 6))
    task_id = section.get("task_id")
    task_filter = None if task_id is None else int(task_id)
    split_filter = section.get("splits", ["train", "validation"])
    splits = {str(value) for value in split_filter}
    invalid_splits = sorted(splits - {"train", "validation", "test"})
    if invalid_splits:
        raise ValueError(f"splits must be canonical train/validation/test values: {invalid_splits}")
    max_records = section.get("max_records")
    max_records = None if max_records is None else int(max_records)
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive")
    paths_by_task = section.get("phase2d_by_task")
    if not isinstance(paths_by_task, Mapping) or not paths_by_task:
        raise ValueError("phase2d_by_task must map task IDs to Phase 2D files")

    selected: list[dict[str, Any]] = []
    needed: set[tuple[int, str, int]] = set()
    for row in _iter_joined(joined_path):
        if int(row.get("tau", -1)) != tau:
            continue
        if task_filter is not None and int(row.get("task_id", -1)) != task_filter:
            continue
        row_split = str(row.get("split"))
        if row_split not in splits:
            continue
        key = _sample_key(row["task_id"], row["episode_id"], row["current_step"])
        if key in needed:
            raise ValueError(f"duplicate policy sample key: {key}")
        selected.append(dict(row))
        needed.add(key)
        if max_records is not None and len(selected) >= max_records:
            break
    if not selected:
        raise ValueError("joined manifest selection contains no policy rows")

    source_actions = _source_action_index(paths_by_task, needed, tau=tau)
    counters = {"selected_rows": 0, "train_rows": 0, "validation_rows": 0, "test_rows": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_jsonl(output_path) as handle:
        for row in selected:
            key = _sample_key(row["task_id"], row["episode_id"], row["current_step"])
            source_split, action_target = source_actions[key]
            if source_split != str(row["split"]):
                raise ValueError(
                    f"joined/source split mismatch for {key}: {row['split']} != {source_split}"
                )
            # The target graph is intentionally omitted.  Future actions are
            # labels for the policy, never part of the model-input view.
            emitted = {
                "schema": SCHEMA,
                "sample_id": str(row["sample_id"]),
                "task_id": int(row["task_id"]),
                "episode_id": str(row["episode_id"]),
                "demo_key": str(row["demo_key"]),
                "split": str(row["split"]),
                "tau": tau,
                "prev_step": int(row["prev_step"]),
                "current_step": int(row["current_step"]),
                "target_step": int(row["target_step"]),
                "past_action_window": row["past_action_window"],
                "graph_prev": row["graph_prev"],
                "graph_t": row["graph_t"],
                "action_target_window": action_target,
                "source_action_sha256": canonical_sha256(action_target),
            }
            write_json_line(handle, emitted)
            counters["selected_rows"] += 1
            counters[f"{emitted['split']}_rows"] += 1
    report = {
        "schema": "phase5-policy-manifest.v1",
        "status": "completed",
        "tau": tau,
        "task_id": task_filter,
        "splits": sorted(splits),
        "joined_manifest": str(joined_path),
        "output": str(output_path),
        "counters": counters,
        "future_action_in_model_input": False,
        "target_graph_emitted": False,
    }
    write_json(qa_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = build_policy_manifest(load_json_config(args.config))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
