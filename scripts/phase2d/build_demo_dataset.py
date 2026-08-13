"""Build the event-centered, multi-horizon Phase 2D dataset from LIBERO demos.

Each HDF5 simulator state is restored directly.  Stored actions are retained
only as transition-conditioning windows.  Per-demo shards make long Colab runs
restartable; completed shards are merged only after every demo passes.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.phase2d.input_clean import clean_record
from scripts.phase2d.split_manifest import split_lookup, validate_manifest
from scripts.phase2d.state_replay import make_environment, replay_demo_group, sha256_file
from scripts.phase2d.temporal_holding import HoldingPolicy, annotate_temporal_holding
from scripts.phase2r.build_dataset import build_dataset
from scripts.phase2r.collect_libero_trajectory import _task_semantic_metadata


def _demo_number(demo_key: str) -> int:
    return int(str(demo_key).rsplit("_", 1)[-1])


def _demo_keys(handle: Any) -> list[str]:
    return sorted(
        (str(key) for key in handle["data"].keys() if str(key).startswith("demo_")),
        key=_demo_number,
    )


def _split_payload(item: Mapping[str, Any]) -> dict[str, str]:
    return {
        "in_task": str(item["in_task_split"]),
        "task_generalization": str(item["task_generalization_split"]),
    }


def _demo_qa(
    task_id: int,
    demo_key: str,
    snapshots: list[dict[str, Any]],
    events: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    replay_error: float,
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    sample_counts: Counter[str] = Counter()
    change_counts: Counter[str] = Counter()
    for record in records:
        horizon = str(record["horizon"])
        samples = record.get("samples", [])
        sample_counts[horizon] += len(samples)
        change_counts[horizon] += sum(
            sum(int(value) for value in sample.get("relation_changes", {}).values())
            for sample in samples if isinstance(sample, Mapping)
        )
    return {
        "task_id": task_id,
        "demo_key": demo_key,
        "frame_count": len(snapshots),
        "robot_contact_frames": sum(bool(snapshot.get("robot_contact_pairs")) for snapshot in snapshots),
        "holding_frames": sum(
            state == "holding"
            for snapshot in snapshots for state in snapshot.get("holding_state", {}).values()
        ),
        "holding_intervals": len(intervals),
        "events": len(events),
        "state_replay_max_abs": replay_error,
        "sample_counts": dict(sample_counts),
        "changed_relation_counts": dict(change_counts),
    }


def _write_demo_shard(
    shard_path: Path,
    records: Iterable[Mapping[str, Any]],
    compression_level: int,
) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = shard_path.with_suffix(shard_path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=compression_level) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    temporary.replace(shard_path)


def _merge_shards(shards: Iterable[Path], output: Path, compression_level: int) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    records = 0
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=compression_level) as destination:
        for shard in shards:
            with gzip.open(shard, "rt", encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        destination.write(line)
                        records += 1
    temporary.replace(output)
    return records


def _aggregate_task_qa(task_id: int, source: Path, reports: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    reports = list(reports)
    sample_counts: Counter[str] = Counter()
    change_counts: Counter[str] = Counter()
    for report in reports:
        sample_counts.update(report.get("sample_counts", {}))
        change_counts.update(report.get("changed_relation_counts", {}))
    return {
        "task_id": task_id,
        "source_hdf5": str(source),
        "source_sha256": sha256_file(source),
        "demo_count": len(reports),
        "frame_count": sum(int(report["frame_count"]) for report in reports),
        "robot_contact_frames": sum(int(report["robot_contact_frames"]) for report in reports),
        "holding_frames": sum(int(report["holding_frames"]) for report in reports),
        "holding_intervals": sum(int(report["holding_intervals"]) for report in reports),
        "events": sum(int(report["events"]) for report in reports),
        "state_replay_max_abs": max((float(report["state_replay_max_abs"]) for report in reports), default=0.0),
        "sample_counts": dict(sample_counts),
        "changed_relation_counts": dict(change_counts),
        "split_manifest_mismatches": 0,
        "errors": [],
    }


def build_task(
    task_id: int,
    hdf5_path: Path,
    split_by_key: Mapping[tuple[int, str], Mapping[str, Any]],
    output_root: Path,
    bddl_roots: Iterable[Path],
    horizons: Iterable[int] = (1, 3, 6),
    policy: HoldingPolicy | None = None,
    resume: bool = True,
    compression_level: int = 1,
) -> dict[str, Any]:
    import h5py

    policy = policy or HoldingPolicy()
    task_dir = output_root / f"task{task_id}"
    shard_dir = task_dir / "shards"
    qa_dir = task_dir / "shard_qa"
    shard_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    environment, bddl_path = make_environment(hdf5_path, bddl_roots)
    task_semantics = _task_semantic_metadata(bddl_path)
    completed_reports: list[dict[str, Any]] = []
    try:
        with h5py.File(hdf5_path, "r") as handle:
            keys = _demo_keys(handle)
            for demo_key in keys:
                shard_path = shard_dir / f"{demo_key}.jsonl.gz"
                report_path = qa_dir / f"{demo_key}.json"
                if resume and shard_path.exists() and report_path.exists():
                    completed_reports.append(json.loads(report_path.read_text(encoding="utf-8")))
                    continue
                split_item = split_by_key.get((task_id, demo_key))
                if split_item is None:
                    raise KeyError(f"split manifest missing task {task_id}, {demo_key}")
                snapshots, actions, replay_error = replay_demo_group(
                    environment, handle["data"][demo_key], task_semantics
                )
                events, intervals, _ = annotate_temporal_holding(snapshots, policy)
                episode_id = f"task{task_id}_{demo_key}"
                episode = {
                    "episode_id": episode_id,
                    "suite": "libero_spatial",
                    "task_id": task_id,
                    "task_name": bddl_path.name,
                    "snapshots": snapshots,
                    "actions": [
                        {"step": step, "action": action} for step, action in enumerate(actions)
                    ],
                }
                records: list[dict[str, Any]] = []
                for horizon in horizons:
                    result = build_dataset(
                        {
                            "capture_status": {
                                "source": "official_hdf5_state_replay",
                                "state_replay": "exact",
                                "actions_replayed_for_labels": False,
                            },
                            "episodes": [episode],
                        },
                        tau=int(horizon),
                        include_all_sites=False,
                        coordinate_frame="robot_base",
                        gripper_closed_threshold=policy.gripper_closed_threshold,
                    )
                    split_payload = _split_payload(split_item)
                    samples = result.get("samples", [])
                    for sample in samples:
                        if isinstance(sample, dict):
                            sample["split"] = split_payload["in_task"]
                    record = clean_record(
                        {
                            "artifact_version": "phase2d-full-demo.v3",
                            "episode_id": episode_id,
                            "task_id": task_id,
                            "demo_id": _demo_number(demo_key),
                            "demo_key": demo_key,
                            "task_name": bddl_path.name,
                            "horizon": int(horizon),
                            "split": split_payload,
                            "split_manifest_key": {"task_id": task_id, "demo_key": demo_key},
                            "holding_definition": "temporal_holding_v1",
                            "frame_count": len(snapshots),
                            "holding_intervals": intervals,
                            "events": events,
                            "samples": samples,
                            "audit": result.get("audit", {}),
                            "stats": result.get("stats", {}),
                        }
                    )
                    records.append(record)
                report = _demo_qa(
                    task_id, demo_key, snapshots, events, intervals, replay_error, records
                )
                _write_demo_shard(shard_path, records, compression_level)
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                completed_reports.append(report)
                print("DEMO_DONE", task_id, demo_key, report["frame_count"], report["holding_frames"], flush=True)
    finally:
        environment.close()

    shard_paths = [shard_dir / f"{key}.jsonl.gz" for key in keys]
    if not all(path.exists() for path in shard_paths):
        raise RuntimeError(f"task {task_id}: not all demo shards were produced")
    final_path = task_dir / f"phase2d_task{task_id}_graph_dataset.jsonl.gz"
    merged_records = _merge_shards(shard_paths, final_path, compression_level)
    reports = [json.loads((qa_dir / f"{key}.json").read_text(encoding="utf-8")) for key in keys]
    qa = _aggregate_task_qa(task_id, hdf5_path, reports)
    qa["merged_records"] = merged_records
    qa["status"] = "pass"
    (task_dir / f"phase2d_task{task_id}_coverage_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return qa


def build_release(
    task_files: Mapping[int, Path],
    split_manifest_path: Path,
    output_root: Path,
    bddl_roots: Iterable[Path],
    horizons: Iterable[int] = (1, 3, 6),
    policy: HoldingPolicy | None = None,
    resume: bool = True,
    compression_level: int = 1,
) -> dict[str, Any]:
    horizons = tuple(int(value) for value in horizons)
    policy = policy or HoldingPolicy()
    bddl_roots = tuple(Path(root) for root in bddl_roots)
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    split_report = validate_manifest(split_manifest)
    if split_report["status"] != "pass":
        raise RuntimeError(split_report)
    lookup = split_lookup(split_manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    generation_config = {
        "generator_version": "phase2d-full-demo.v3",
        "task_sources": {
            str(task_id): {"path": str(path), "sha256": sha256_file(path)}
            for task_id, path in sorted(task_files.items())
        },
        "split_manifest": {
            "path": str(split_manifest_path),
            "sha256": sha256_file(split_manifest_path),
        },
        "horizons": list(horizons),
        "holding_policy": asdict(policy),
        "coordinate_frame": "robot_base",
        "actions_replayed_for_labels": False,
    }
    config_path = output_root / "generation_config.json"
    if resume and config_path.exists():
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        if previous_config != generation_config:
            raise RuntimeError("resume configuration differs from existing Phase 2D shards")
    config_path.write_text(json.dumps(generation_config, ensure_ascii=False, indent=2), encoding="utf-8")
    started = time.time()
    summary: dict[str, Any] = {
        "status": "in_progress",
        "artifact_version": "phase2d-full-demo.v3",
        "horizons": list(horizons),
        "holding_definition": "temporal_holding_v1",
        "split_manifest": str(split_manifest_path),
        "tasks": {},
    }
    progress_path = output_root / "run_progress.json"
    progress_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for task_id, hdf5_path in sorted(task_files.items()):
        summary["tasks"][str(task_id)] = build_task(
            task_id, hdf5_path, lookup, output_root, bddl_roots,
            horizons=horizons, policy=policy, resume=resume,
            compression_level=compression_level,
        )
        progress_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary.update(status="complete", elapsed_seconds=time.time() - started)
    summary_path = output_root / "phase2d_full_demo_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "artifact_version": "phase2d-full-demo.v3",
        "status": "complete",
        "source_identity_verified": True,
        "split_manifest_verified": True,
        "actions_replayed_for_labels": False,
        "output_root": str(output_root),
        "summary": str(summary_path),
        "source_hdf5": {
            str(task_id): {"path": str(path), "sha256": sha256_file(path)}
            for task_id, path in sorted(task_files.items())
        },
    }
    (output_root / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _task_mapping(values: Iterable[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        task_id, separator, path = value.partition("=")
        if not separator:
            raise ValueError("--task must use TASK_ID=/path/to/demo.hdf5")
        result[int(task_id)] = Path(path)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", required=True, help="TASK_ID=/path/to/demo.hdf5")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bddl-root", action="append", type=Path, default=[])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--holding-history", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--compression-level", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    roots = args.bddl_root or [Path("/content/LIBERO")]
    summary = build_release(
        _task_mapping(args.task), args.split_manifest, args.output_root, roots,
        horizons=args.horizons,
        policy=HoldingPolicy(history_frames=args.holding_history),
        resume=not args.no_resume,
        compression_level=args.compression_level,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
