"""Build the Phase 3C causal joined manifest from Phase 2D samples.

The builder uses the left sample only for the already-executed action window
and the right sample only for the future target graph.  The right sample's
future action window is never copied into the joined record.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .contracts import (
    ACTION_DIM,
    PHASE3C_SCHEMA_VERSION,
    PRIMARY_RELATIONS,
    TAU,
    JoinCounters,
    assert_causal_input,
    canonical_sha256,
    parse_action_window,
    relation_targets,
    scene_max_displacement,
    support_report,
    validate_graph,
)
from .io import atomic_json, atomic_jsonl, iter_phase2d_samples, load_json_config, write_json_line


def _int_field(sample: Mapping[str, Any], key: str) -> int:
    try:
        return int(sample[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"sample missing integer {key}: {sample.get('sample_id')}") from exc


def _episode_key(sample: Mapping[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(sample.get("episode_id")),
        int(sample.get("task_id")),
        str(sample.get("demo_key", "")),
        str(sample.get("split")),
    )


def _sample_key(sample: Mapping[str, Any]) -> tuple[tuple[str, int, str, str], int, int]:
    return _episode_key(sample), _int_field(sample, "start_step"), _int_field(sample, "tau")


def _task_id(sample: Mapping[str, Any]) -> int:
    try:
        return int(sample["task_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Phase 2D sample requires integer task_id") from exc


def _joined_sample_id(
    sample: Mapping[str, Any], *, prev_step: int, current_step: int, target_step: int, tau: int
) -> str:
    episode = str(sample.get("episode_id") or f"task{_task_id(sample)}_{sample.get('demo_key', 'unknown')}")
    safe_episode = episode.replace("/", "_").replace(" ", "_")
    return f"{safe_episode}_prev{prev_step}_cur{current_step}_next{target_step}_tau{tau}"


def _validate_phase2d_sample(sample: Mapping[str, Any], *, tau: int) -> None:
    if _int_field(sample, "tau") != tau:
        raise ValueError(f"sample tau mismatch: expected {tau}, got {sample.get('tau')}")
    validate_graph(sample.get("graph_t"), name="sample.graph_t")
    validate_graph(sample.get("graph_target"), name="sample.graph_target")
    parse_action_window(sample.get("action_window"), tau=tau)
    if "future_action" in sample or "action_future" in sample:
        raise ValueError("Phase 2D sample already contains a forbidden future-action field")


def _build_record(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    tau: int,
    relations: tuple[str, ...],
) -> dict[str, Any]:
    prev_step = _int_field(left, "start_step")
    current_step = _int_field(left, "target_step")
    target_step = _int_field(right, "target_step")
    current_graph = left["graph_target"]
    if canonical_sha256(current_graph) != canonical_sha256(right["graph_t"]):
        raise ValueError("left.graph_target and right.graph_t hashes do not match")
    target_graph = right["graph_target"]
    past_action_window = parse_action_window(left["action_window"], tau=tau)
    labels = relation_targets(current_graph, target_graph, relations=relations)
    record: dict[str, Any] = {
        "schema": PHASE3C_SCHEMA_VERSION,
        "sample_id": _joined_sample_id(
            left,
            prev_step=prev_step,
            current_step=current_step,
            target_step=target_step,
            tau=tau,
        ),
        "task_id": _task_id(left),
        "episode_id": str(left.get("episode_id")),
        "demo_key": str(left.get("demo_key", "")),
        "split": str(left.get("split")),
        "tau": tau,
        "prev_step": prev_step,
        "current_step": current_step,
        "target_step": target_step,
        "past_action_window": past_action_window,
        "graph_prev": left["graph_t"],
        "graph_t": current_graph,
        "target": {
            "graph": target_graph,
            "relation_any_change": labels["relation_any_change"],
            "relation_valid": labels["relation_valid"],
            "edge_diagnostics": labels["edge_diagnostics"],
            "candidate_edge_count": labels["candidate_edge_count"],
            "scene_max_displacement_m": scene_max_displacement(current_graph, target_graph),
        },
        "hashes": {
            "left_graph_t": canonical_sha256(left["graph_t"]),
            "left_graph_target": canonical_sha256(left["graph_target"]),
            "right_graph_t": canonical_sha256(right["graph_t"]),
            "right_graph_target": canonical_sha256(right["graph_target"]),
            "left_source": canonical_sha256(
                {"path": left.get("_source_path"), "line": left.get("_source_line")}
            ),
            "right_source": canonical_sha256(
                {"path": right.get("_source_path"), "line": right.get("_source_line")}
            ),
        },
    }
    assert_causal_input(record)
    return record


def build_joined_manifest(
    input_paths: Iterable[Path],
    output_path: Path,
    qa_path: Path,
    *,
    tau: int = TAU,
    relations: tuple[str, ...] = PRIMARY_RELATIONS,
    max_output_records: int | None = None,
) -> dict[str, Any]:
    if tau < 1:
        raise ValueError("tau must be positive")
    if not relations:
        raise ValueError("at least one relation is required")
    paths = tuple(Path(path) for path in input_paths)
    if not paths:
        raise ValueError("at least one Phase 2D input path is required")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    counters = {
        "source_records": 0,
        "source_samples": 0,
        "candidate_left_samples": 0,
        "joined_samples": 0,
        "boundary_drops": 0,
        "missing_right_samples": 0,
        "hash_mismatches": 0,
        "duplicate_left_keys": 0,
        "invalid_samples": 0,
        "emitted_future_action_fields": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_text: str | None = None
    error: BaseException | None = None
    # Phase 2D's canonical artifact is one demo record per JSONL line.  Keep
    # only that demo in memory so the 800+ MB artifact remains streamable.
    try:
        with atomic_jsonl(output_path) as output:
            for path in paths:
                for source_record in _iter_source_records(path, tau=tau, counters=counters):
                    counters["source_records"] += 1
                    samples = source_record
                    by_start: dict[int, Mapping[str, Any]] = {}
                    for sample in samples:
                        counters["source_samples"] += 1
                        try:
                            _validate_phase2d_sample(sample, tau=tau)
                            start = _int_field(sample, "start_step")
                            if start in by_start:
                                counters["duplicate_left_keys"] += 1
                                raise ValueError(f"duplicate episode start_step={start}")
                            by_start[start] = sample
                        except (KeyError, TypeError, ValueError):
                            counters["invalid_samples"] += 1
                            raise
                    for left_start in sorted(by_start):
                        left = by_start[left_start]
                        current_step = _int_field(left, "target_step")
                        counters["candidate_left_samples"] += 1
                        right = by_start.get(current_step)
                        if right is None:
                            # A terminal window has no next window; this is an
                            # expected boundary drop, not a malformed sample.
                            counters["boundary_drops"] += 1
                            continue
                        if _episode_key(left) != _episode_key(right):
                            counters["missing_right_samples"] += 1
                            raise ValueError("joined samples do not share episode/task/demo/split")
                        if _int_field(right, "start_step") != current_step:
                            counters["missing_right_samples"] += 1
                            raise ValueError("right sample start_step does not equal left target_step")
                        if canonical_sha256(left["graph_target"]) != canonical_sha256(right["graph_t"]):
                            counters["hash_mismatches"] += 1
                            raise ValueError(
                                f"graph hash mismatch at {_episode_key(left)} current_step={current_step}"
                            )
                        try:
                            record = _build_record(left, right, tau=tau, relations=relations)
                        except ValueError:
                            if canonical_sha256(left["graph_target"]) != canonical_sha256(right["graph_t"]):
                                counters["hash_mismatches"] += 1
                            raise
                        if any(key in record for key in ("future_action", "action_future")):
                            counters["emitted_future_action_fields"] += 1
                            raise AssertionError("joined record emitted a future action field")
                        write_json_line(output, record)
                        counters["joined_samples"] += 1
                        if max_output_records is not None and counters["joined_samples"] >= max_output_records:
                            break
                    if max_output_records is not None and counters["joined_samples"] >= max_output_records:
                        break
                if max_output_records is not None and counters["joined_samples"] >= max_output_records:
                    break
    except Exception as exc:
        error = exc
        error_text = f"{type(exc).__name__}: {exc}"

    # Re-read the committed output for support counts.  This keeps the join
    # memory bounded by one demo while retaining an auditable report.
    from .io import iter_json_objects

    output_exists = output_path.exists()
    report = {
        "contract": "phase3c-causal-join.v1",
        "schema": PHASE3C_SCHEMA_VERSION,
        "tau": tau,
        "relations": list(relations),
        "inputs": [str(path) for path in paths],
        "counters": counters,
        "support": support_report(iter_json_objects(output_path)) if output_exists else support_report(()),
        "status": "pass" if error_text is None and counters["joined_samples"] > 0 else "fail",
        "error": error_text,
        "output": str(output_path),
    }
    with atomic_json(qa_path) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    if error is not None:
        raise error
    return report


def _iter_source_records(
    path: Path, *, tau: int, counters: dict[str, int]
) -> Iterable[list[dict[str, Any]]]:
    """Yield one demo's samples at a time from the canonical Phase 2D format."""

    pending: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in iter_phase2d_samples((path,)):
        # `iter_phase2d_samples` flattens nested records.  The canonical file
        # is ordered by demo, so flush when the episode changes.  The pending
        # map also makes bare sample JSONL fixtures work in tests.
        key = _episode_key(sample)
        if pending and key not in pending:
            for rows in pending.values():
                yield rows
            pending.clear()
        pending[key].append(sample)
    for rows in pending.values():
        yield rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qa-output", type=Path)
    parser.add_argument("--tau", type=int)
    parser.add_argument("--relations", nargs="+")
    parser.add_argument("--max-output-records", type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config: dict[str, Any] = load_json_config(args.config) if args.config else {}
    paths = tuple(args.input or tuple(Path(value) for value in config.get("inputs", [])))
    output = args.output or (Path(config["output"]) if config.get("output") else None)
    qa_output = args.qa_output or (Path(config["qa_output"]) if config.get("qa_output") else None)
    if output is None or qa_output is None:
        raise SystemExit("--output and --qa-output are required (directly or in --config)")
    tau = int(args.tau if args.tau is not None else config.get("tau", TAU))
    relations = tuple(args.relations or config.get("relations", PRIMARY_RELATIONS))
    report = build_joined_manifest(
        paths,
        output,
        qa_output,
        tau=tau,
        relations=relations,
        max_output_records=args.max_output_records,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
