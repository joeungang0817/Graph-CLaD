"""Build a reproducible 90-item manual holding weak-label audit bundle.

The bundle contains ten onset, ten release, and ten contact-without-holding
hard-negative items for each of tasks 0/1/2.  Candidates are clustered within
episode/object/event type to reduce duplicate overlapping windows, then chosen
with deterministic episode round-robin.  The output is evidence for human
review, not an automatic claim that the weak labels are correct.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase3.build_eval_manifest import (
    load_samples,
    payload_hash,
    sample_id,
)


EVENT_TYPES = ("onset", "release", "hard_negative")


def _relation(edge: Mapping[str, Any], name: str) -> tuple[bool, bool, Any]:
    record = (edge.get("relations", {}) or {}).get(name, {})
    if not isinstance(record, Mapping):
        return False, False, None
    return bool(record.get("value")), bool(record.get("valid")), record.get("evidence")


def _edge_map(graph: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(edge.get("source")), str(edge.get("target"))): edge
        for edge in graph.get("edges", [])
        if isinstance(edge, Mapping)
    }


def _node_map(graph: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(node.get("node_id")): node
        for node in graph.get("nodes", [])
        if isinstance(node, Mapping)
    }


def _is_robot_object(edge: Mapping[str, Any]) -> bool:
    source = str(edge.get("source"))
    target = str(edge.get("target"))
    pair = edge.get("node_type_pair")
    if isinstance(pair, Sequence) and not isinstance(pair, (str, bytes)) and len(pair) == 2:
        types = {str(pair[0]), str(pair[1])}
        return types == {"robot", "object"}
    return source.startswith("robot") != target.startswith("robot")


def _graph_digest(graph: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        graph,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selected_step_graphs(
    samples: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[int, str, int], Mapping[str, Any]], dict[str, Any]]:
    """Collect only graph frames required by the selected audit windows."""
    requested = {
        (int(item["task_id"]), str(item["episode_id"]), step)
        for item in selected
        for step in range(int(item["start_step"]), int(item["target_step"]) + 1)
    }
    graphs: dict[tuple[int, str, int], Mapping[str, Any]] = {}
    digests: dict[tuple[int, str, int], str] = {}
    conflicts: list[dict[str, Any]] = []
    for sample in samples:
        task_id = int(sample.get("task_id", -1))
        episode_id = str(sample.get("episode_id", ""))
        candidates = (
            (int(sample.get("start_step", -1)), sample.get("graph_t", {}) or {}),
            (int(sample.get("target_step", -1)), sample.get("graph_target", {}) or {}),
        )
        for step, graph in candidates:
            key = (task_id, episode_id, step)
            if key not in requested or not isinstance(graph, Mapping) or not graph:
                continue
            digest = _graph_digest(graph)
            if key in digests and digests[key] != digest:
                conflicts.append(
                    {
                        "task_id": task_id,
                        "episode_id": episode_id,
                        "step": step,
                        "first_sha256": digests[key],
                        "new_sha256": digest,
                    }
                )
                continue
            graphs[key] = graph
            digests[key] = digest
    missing = sorted(requested - set(graphs))
    qa = {
        "requested_step_graphs": len(requested),
        "available_step_graphs": len(graphs),
        "missing_step_graphs": [
            {"task_id": task, "episode_id": episode, "step": step}
            for task, episode, step in missing
        ],
        "conflicting_step_graphs": conflicts,
        "status": "pass" if not missing and not conflicts else "fail",
    }
    return graphs, qa


def _trajectory_point(
    graph: Mapping[str, Any],
    step: int,
    edge_key: tuple[str, str],
) -> dict[str, Any]:
    nodes = _node_map(graph)
    edge = _edge_map(graph).get(edge_key, {})
    source = nodes.get(edge_key[0], {})
    target = nodes.get(edge_key[1], {})
    source_features = source.get("features", {}) or {}
    target_features = target.get("features", {}) or {}
    holding, holding_valid, holding_evidence = _relation(edge, "holding")
    contact, contact_valid, _ = _relation(edge, "contact")
    holding_record = (edge.get("relations", {}) or {}).get("holding", {}) or {}
    edge_features = edge.get("features", {}) or {}
    return {
        "step": int(step),
        "source_position": source_features.get("position"),
        "source_position_valid": source_features.get("position_valid"),
        "target_position": target_features.get("position"),
        "target_position_valid": target_features.get("position_valid"),
        "relative_position": edge_features.get("relative_position"),
        "distance": edge_features.get("distance"),
        "gripper_qpos": source_features.get("gripper_qpos"),
        "gripper_qpos_valid": source_features.get("gripper_qpos_valid"),
        "holding": holding,
        "holding_valid": holding_valid,
        "holding_state": holding_record.get("state"),
        "holding_confidence": holding_record.get("confidence"),
        "holding_evidence": holding_evidence,
        "contact": contact if contact_valid else None,
        "contact_valid": contact_valid,
    }


def _candidate_rows(
    samples: Sequence[Mapping[str, Any]],
    task_id: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for sample in samples:
        if int(sample.get("task_id", -1)) != task_id or int(sample.get("tau", -1)) != 6:
            continue
        current_edges = _edge_map(sample.get("graph_t", {}) or {})
        future_edges = _edge_map(sample.get("graph_target", {}) or {})
        for edge_key in sorted(set(current_edges) & set(future_edges)):
            current_edge = current_edges[edge_key]
            future_edge = future_edges[edge_key]
            if not _is_robot_object(current_edge):
                continue
            current_holding, current_valid, _ = _relation(current_edge, "holding")
            future_holding, future_valid, _ = _relation(future_edge, "holding")
            if not (current_valid and future_valid):
                continue
            current_contact, current_contact_valid, _ = _relation(current_edge, "contact")
            future_contact, future_contact_valid, _ = _relation(future_edge, "contact")
            event_type = None
            if not current_holding and future_holding:
                event_type = "onset"
            elif current_holding and not future_holding:
                event_type = "release"
            elif (
                not current_holding
                and not future_holding
                and (
                    (current_contact_valid and current_contact)
                    or (future_contact_valid and future_contact)
                )
            ):
                event_type = "hard_negative"
            if event_type is None:
                continue
            candidates.append(
                {
                    "sample": sample,
                    "sample_id": sample_id(sample),
                    "task_id": task_id,
                    "episode_id": str(sample["episode_id"]),
                    "start_step": int(sample["start_step"]),
                    "target_step": int(sample["target_step"]),
                    "tau": int(sample["tau"]),
                    "edge_source": edge_key[0],
                    "edge_target": edge_key[1],
                    "event_type": event_type,
                    "current_holding": current_holding,
                    "future_holding": future_holding,
                    "current_contact": current_contact if current_contact_valid else None,
                    "future_contact": future_contact if future_contact_valid else None,
                }
            )
    return candidates


def _cluster_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[
            (
                str(candidate["episode_id"]),
                str(candidate["edge_source"]),
                str(candidate["edge_target"]),
                str(candidate["event_type"]),
            )
        ].append(candidate)
    clusters: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: (int(row["target_step"]), str(row["sample_id"])))
        current: list[Mapping[str, Any]] = []
        last_step: int | None = None
        for row in rows:
            step = int(row["target_step"])
            if current and last_step is not None and step - last_step > int(row["tau"]):
                clusters.append(_make_cluster(key, current, len(clusters)))
                current = []
            current.append(row)
            last_step = step
        if current:
            clusters.append(_make_cluster(key, current, len(clusters)))
    return clusters


def _make_cluster(
    key: tuple[str, str, str, str],
    rows: Sequence[Mapping[str, Any]],
    index: int,
) -> dict[str, Any]:
    representative = min(
        rows,
        key=lambda row: hashlib.sha256(str(row["sample_id"]).encode("utf-8")).hexdigest(),
    )
    return {
        **dict(representative),
        "event_cluster_id": (
            f"{key[0]}|{key[1]}|{key[2]}|{key[3]}|cluster{index}"
        ),
        "cluster_window_count": len(rows),
        "cluster_target_step_min": min(int(row["target_step"]) for row in rows),
        "cluster_target_step_max": max(int(row["target_step"]) for row in rows),
    }


def _episode_round_robin(
    clusters: Sequence[Mapping[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    by_episode: dict[str, deque[Mapping[str, Any]]] = {}
    for episode in sorted({str(cluster["episode_id"]) for cluster in clusters}):
        rows = [cluster for cluster in clusters if str(cluster["episode_id"]) == episode]
        rows.sort(
            key=lambda row: hashlib.sha256(
                str(row["event_cluster_id"]).encode("utf-8")
            ).hexdigest()
        )
        by_episode[episode] = deque(rows)
    selected: list[dict[str, Any]] = []
    episodes = sorted(by_episode)
    while len(selected) < count and episodes:
        next_episodes: list[str] = []
        for episode in episodes:
            queue = by_episode[episode]
            if queue and len(selected) < count:
                selected.append(dict(queue.popleft()))
            if queue:
                next_episodes.append(episode)
        episodes = next_episodes
    return selected


def _evidence(
    item: Mapping[str, Any],
    audit_id: str,
    step_graphs: Mapping[tuple[int, str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    sample = item["sample"]
    current_graph = sample.get("graph_t", {}) or {}
    future_graph = sample.get("graph_target", {}) or {}
    edge_key = (str(item["edge_source"]), str(item["edge_target"]))
    current_edge = _edge_map(current_graph)[edge_key]
    future_edge = _edge_map(future_graph)[edge_key]
    current_nodes = _node_map(current_graph)
    future_nodes = _node_map(future_graph)
    trajectory = []
    missing_trajectory_steps = []
    for step in range(int(item["start_step"]), int(item["target_step"]) + 1):
        graph = step_graphs.get(
            (int(item["task_id"]), str(item["episode_id"]), step)
        )
        if graph is None:
            missing_trajectory_steps.append(step)
            continue
        trajectory.append(_trajectory_point(graph, step, edge_key))
    return {
        "audit_id": audit_id,
        "task_id": int(item["task_id"]),
        "event_type": str(item["event_type"]),
        "sample_id": str(item["sample_id"]),
        "payload_sha256": payload_hash(sample),
        "episode_id": str(item["episode_id"]),
        "split": sample.get("split"),
        "start_step": int(item["start_step"]),
        "target_step": int(item["target_step"]),
        "tau": int(item["tau"]),
        "event_cluster_id": str(item["event_cluster_id"]),
        "cluster_window_count": int(item["cluster_window_count"]),
        "cluster_target_step_range": [
            int(item["cluster_target_step_min"]),
            int(item["cluster_target_step_max"]),
        ],
        "edge": {"source": edge_key[0], "target": edge_key[1]},
        "current": {
            "source_node": current_nodes.get(edge_key[0]),
            "target_node": current_nodes.get(edge_key[1]),
            "edge": current_edge,
        },
        "future": {
            "source_node": future_nodes.get(edge_key[0]),
            "target_node": future_nodes.get(edge_key[1]),
            "edge": future_edge,
        },
        "action_window": sample.get("action_window", []),
        "trajectory": trajectory,
        "trajectory_qa": {
            "requested_steps": int(item["tau"]) + 1,
            "available_steps": len(trajectory),
            "missing_steps": missing_trajectory_steps,
            "status": "pass" if not missing_trajectory_steps else "fail",
        },
        "target_categories": sample.get("target_categories", []),
        "review_contract": {
            "accepted_decisions": ["pass", "label_error", "ambiguous"],
            "accepted_error_types": [
                "false_onset",
                "missed_onset",
                "false_release",
                "missed_release",
                "hard_negative_is_holding",
                "contact_mapping_error",
                "temporal_alignment_error",
                "insufficient_evidence",
                "other",
            ],
        },
    }


def build_audit(
    natural_root: Path,
    output_root: Path,
    task_ids: Iterable[int] = (0, 1, 2),
    per_event: int | None = 10,
) -> dict[str, Any]:
    task_ids = [int(value) for value in task_ids]
    samples = load_samples(natural_root, task_ids)
    selected: list[dict[str, Any]] = []
    support: dict[str, Any] = {}
    shortages: list[str] = []
    for task_id in task_ids:
        candidates = _candidate_rows(samples, task_id)
        clusters = _cluster_candidates(candidates)
        support[str(task_id)] = {}
        for event_type in EVENT_TYPES:
            event_clusters = [
                cluster for cluster in clusters if cluster["event_type"] == event_type
            ]
            if per_event is None:
                chosen = sorted(
                    (dict(cluster) for cluster in event_clusters),
                    key=lambda row: str(row["event_cluster_id"]),
                )
            else:
                chosen = _episode_round_robin(event_clusters, per_event)
            support[str(task_id)][event_type] = {
                "candidate_windows": sum(
                    1 for row in candidates if row["event_type"] == event_type
                ),
                "event_clusters": len(event_clusters),
                "episodes": len(
                    {str(row["episode_id"]) for row in event_clusters}
                ),
                "selected": len(chosen),
            }
            if per_event is not None and len(chosen) != per_event:
                shortages.append(
                    f"task{task_id} {event_type}: selected {len(chosen)}/{per_event}"
                )
            selected.extend(chosen)

    output_root.mkdir(parents=True, exist_ok=True)
    evidence_path = output_root / "holding_weak_label_audit_evidence_v1.jsonl.gz"
    review_path = output_root / "holding_weak_label_audit_review_v1.csv"
    manifest_path = output_root / "holding_weak_label_audit_manifest_v1.json"
    step_graphs, trajectory_qa = _selected_step_graphs(samples, selected)
    evidence_rows: list[dict[str, Any]] = []
    for index, item in enumerate(selected, 1):
        evidence_rows.append(
            _evidence(item, f"HQA-{index:03d}", step_graphs)
        )
    with gzip.open(evidence_path, "wt", encoding="utf-8") as handle:
        for row in evidence_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    review_fields = [
        "audit_id",
        "task_id",
        "event_type",
        "sample_id",
        "episode_id",
        "start_step",
        "target_step",
        "edge_source",
        "edge_target",
        "reviewer_decision",
        "label_error_type",
        "reviewer_notes",
    ]
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        for row in evidence_rows:
            writer.writerow(
                {
                    "audit_id": row["audit_id"],
                    "task_id": row["task_id"],
                    "event_type": row["event_type"],
                    "sample_id": row["sample_id"],
                    "episode_id": row["episode_id"],
                    "start_step": row["start_step"],
                    "target_step": row["target_step"],
                    "edge_source": row["edge"]["source"],
                    "edge_target": row["edge"]["target"],
                    "reviewer_decision": "",
                    "label_error_type": "",
                    "reviewer_notes": "",
                }
            )
    manifest = {
        "protocol": "phase3-holding-weak-label-manual-audit-v1",
        "status": (
            "ready_for_manual_review"
            if not shortages and trajectory_qa["status"] == "pass"
            else "insufficient_support"
        ),
        "source_root": str(natural_root),
        "task_ids": task_ids,
        "event_types": list(EVENT_TYPES),
        "per_task_event_count": per_event,
        "all_event_clusters": per_event is None,
        "expected_items": (
            len(evidence_rows)
            if per_event is None
            else len(task_ids) * len(EVENT_TYPES) * per_event
        ),
        "selected_items": len(evidence_rows),
        "selection": (
            "all tau6 event clusters"
            if per_event is None
            else "tau6 event clustering plus deterministic episode round-robin"
        ),
        "support": support,
        "shortages": shortages,
        "trajectory_qa": trajectory_qa,
        "evidence": str(evidence_path),
        "review_csv": str(review_path),
        "review_summary": {
            "status": "pending",
            "reviewed": 0,
            "passed": 0,
            "label_errors": 0,
            "ambiguous": 0,
            "pass_rate": None,
            "error_type_counts": {},
        },
        "claim_limit": "Selection and evidence export do not constitute manual validation.",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--per-event", type=int, default=10)
    parser.add_argument(
        "--all-clusters",
        action="store_true",
        help="Export one deterministic representative from every event cluster.",
    )
    args = parser.parse_args()
    manifest = build_audit(
        args.natural_root,
        args.output_root,
        task_ids=args.tasks,
        per_event=None if args.all_clusters else args.per_event,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "selected_items": manifest["selected_items"],
                "expected_items": manifest["expected_items"],
                "review_csv": manifest["review_csv"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if manifest["status"] == "ready_for_manual_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
