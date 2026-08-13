"""Validate Phase 2R scale-up frame stability and relation coverage."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

try:
    from .validate_pilot import validate_payload
except ImportError:  # pragma: no cover - CLI execution from scripts/
    from validate_phase2r_pilot import validate_payload


SEMANTIC_EDGE_RELATIONS = ("on", "inside", "holding")
NODE_RELATIONS = ("open", "close")


def _task_key(record: Mapping[str, Any]) -> str:
    suite = record.get("suite")
    task_id = record.get("task_id")
    return f"{suite}:{task_id}" if suite is not None else str(task_id)


def _range(values: list[list[float]]) -> list[float] | None:
    if not values:
        return None
    dimensions = len(values[0])
    return [max(row[index] for row in values) - min(row[index] for row in values) for index in range(dimensions)]


def _frame_audit(capture: Mapping[str, Any]) -> dict[str, Any]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for episode in capture.get("episodes", []):
        task_key = _task_key(episode)
        for snapshot in episode.get("snapshots", []):
            by_task[task_key].append(snapshot)
    task_results: dict[str, Any] = {}
    for task_id, snapshots in sorted(by_task.items()):
        poses = [snapshot.get("robot_base_pose") for snapshot in snapshots]
        valid_poses = [pose for pose in poses if isinstance(pose, Mapping)]
        positions = [pose.get("position") for pose in valid_poses if isinstance(pose.get("position"), list)]
        quaternions = [pose.get("quaternion") for pose in valid_poses if isinstance(pose.get("quaternion"), list)]
        position_range = _range(positions)
        quaternion_range = _range(quaternions)
        task_results[task_id] = {
            "snapshot_count": len(snapshots),
            "robot_base_pose_present": len(valid_poses),
            "geometry_present": sum(bool(snapshot.get("geometry")) for snapshot in snapshots),
            "position_range": position_range,
            "quaternion_range": quaternion_range,
            "stable": (
                len(valid_poses) == len(snapshots)
                and position_range is not None
                and quaternion_range is not None
                and all(value == 0.0 for value in position_range + quaternion_range)
            ),
        }
    return {
        "by_task": task_results,
        "all_tasks_stable": bool(task_results) and all(result["stable"] for result in task_results.values()),
    }


def _relation_coverage(dataset: Mapping[str, Any]) -> dict[str, Any]:
    edge_counts: dict[str, Counter[str]] = defaultdict(Counter)
    node_counts: dict[str, Counter[str]] = defaultdict(Counter)
    changed: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in dataset.get("samples", []):
        task_id = _task_key(sample)
        for graph in (sample.get("graph_t", {}), sample.get("graph_target", {})):
            for edge in graph.get("edges", []):
                for relation in SEMANTIC_EDGE_RELATIONS:
                    record = edge.get("relations", {}).get(relation, {})
                    if record.get("valid"):
                        edge_counts[task_id][relation] += 1
            for node_records in graph.get("node_semantics", {}).values():
                for relation in NODE_RELATIONS:
                    record = node_records.get(relation, {})
                    if record.get("valid"):
                        node_counts[task_id][relation] += 1
        for key, value in sample.get("relation_changes", {}).items():
            if not value or "::" not in key:
                continue
            relation = key.rsplit("::", 1)[1]
            if relation in set(SEMANTIC_EDGE_RELATIONS) | set(NODE_RELATIONS):
                changed[task_id][relation] += 1
    return {
        "edge_valid_by_task": {task: dict(counts) for task, counts in sorted(edge_counts.items())},
        "node_valid_by_task": {task: dict(counts) for task, counts in sorted(node_counts.items())},
        "changed_by_task": {task: dict(counts) for task, counts in sorted(changed.items())},
    }


def validate_scaleup(capture: Mapping[str, Any], dataset: Mapping[str, Any]) -> dict[str, Any]:
    structural = validate_payload(dataset)
    frame = _frame_audit(capture)
    coverage = _relation_coverage(dataset)
    task_ids = sorted({_task_key(episode) for episode in capture.get("episodes", [])})
    semantic_edge_total = sum(
        sum(values.values()) for values in coverage["edge_valid_by_task"].values()
    )
    semantic_node_total = sum(
        sum(values.values()) for values in coverage["node_valid_by_task"].values()
    )
    gates = {
        "structural_validation": structural["status"] == "pass",
        "task_count_at_least_two": len(task_ids) >= 2,
        "robot_base_pose_present_and_stable_per_task": frame["all_tasks_stable"],
        "geometry_metadata_present": all(
            result["geometry_present"] == result["snapshot_count"]
            for result in frame["by_task"].values()
        ),
        "semantic_coverage_exists": semantic_edge_total > 0 or semantic_node_total > 0,
    }
    return {
        "validation_version": "phase2r-scaleup.v1",
        "status": "pass" if all(gates.values()) else "fail",
        "task_ids": task_ids,
        "structural_validation": structural,
        "frame_audit": frame,
        "relation_coverage": coverage,
        "gate_results": gates,
        "interpretation": {
            "inside_open_close_may_remain_sparse": True,
            "unknown_is_not_false": True,
            "broad_semantic_claim_requires_relation_critical_tasks": True,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    report = validate_scaleup(capture, dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "task_ids": report["task_ids"],
        "frame": report["frame_audit"],
        "coverage": report["relation_coverage"],
        "gates": report["gate_results"],
    }, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
