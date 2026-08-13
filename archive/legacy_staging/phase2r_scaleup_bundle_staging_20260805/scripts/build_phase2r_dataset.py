"""Build action-conditioned oracle graph transitions for Phase 2R."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .graph_extractor import _make_edges, extract_graph_snapshot
    from .phase2r_relation_handlers import (
        GEOMETRIC_RELATIONS,
        MARGIN,
        RELATIONS,
        relation_labels,
        node_semantic_records,
        transform_snapshot_to_robot_base,
    )
except ImportError:  # pragma: no cover - CLI execution from scripts/
    from graph_extractor import _make_edges, extract_graph_snapshot
    from phase2r_relation_handlers import (
        GEOMETRIC_RELATIONS,
        MARGIN,
        RELATIONS,
        relation_labels,
        node_semantic_records,
        transform_snapshot_to_robot_base,
    )


def _select_nodes(graph: Mapping[str, Any], include_all_sites: bool) -> list[dict[str, Any]]:
    nodes = []
    for original in graph["nodes"]:
        node = copy.deepcopy(original)
        if not include_all_sites and node["node_type"] == "site":
            continue
        # v2 removes the task-derived shortcut from the core feature vector.
        node["features"]["is_object_of_interest"] = 0
        if len(node["feature_vector"]) > 4:
            node["feature_vector"][4] = 0.0
        nodes.append(node)
    return nodes


def graph_with_relations(
    snapshot: Mapping[str, Any],
    include_all_sites: bool = False,
    coordinate_frame: str = "robot_base",
) -> dict[str, Any]:
    model_snapshot = (
        transform_snapshot_to_robot_base(snapshot)
        if coordinate_frame == "robot_base"
        else copy.deepcopy(dict(snapshot))
    )
    model_snapshot["coordinate_frame"] = coordinate_frame
    base = extract_graph_snapshot(model_snapshot)
    nodes = _select_nodes(base, include_all_sites=include_all_sites)
    node_by_id = {node["node_id"]: node for node in nodes}
    edges = _make_edges(nodes)
    for edge in edges:
        edge["relations"] = relation_labels(
            node_by_id[edge["source"]],
            node_by_id[edge["target"]],
            model_snapshot,
        )
        edge["semantic_relation"] = edge["relations"]
    return {
        "step": int(snapshot.get("step", 0)),
        "nodes": nodes,
        "edges": edges,
        "node_feature_dim": base["node_feature_dim"],
        "node_selection": "task_relevant_entities_without_sites" if not include_all_sites else "all_sites",
        "coordinate_frame": coordinate_frame,
        "node_semantics": node_semantic_records(model_snapshot),
    }


def _split_episode_ids(episode_ids: list[str]) -> dict[str, str]:
    ordered = sorted(episode_ids)
    if len(ordered) < 3:
        return {episode_id: "train" for episode_id in ordered}
    assignments = {}
    for index, episode_id in enumerate(ordered):
        if index == len(ordered) - 1:
            split = "test"
        elif index == len(ordered) - 2:
            split = "validation"
        else:
            split = "train"
        assignments[episode_id] = split
    return assignments


def _edge_map(graph: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(edge["source"], edge["target"]): edge for edge in graph["edges"]}


def build_dataset(
    payload: Mapping[str, Any],
    tau: int = 6,
    include_all_sites: bool = False,
    coordinate_frame: str = "robot_base",
) -> dict[str, Any]:
    episodes = payload.get("episodes", [])
    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)):
        raise ValueError("trajectory payload must contain an episodes sequence")
    episode_ids = [str(episode["episode_id"]) for episode in episodes]
    split_by_episode = _split_episode_ids(episode_ids)
    samples = []
    episode_audits = []
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        snapshots = episode.get("snapshots", [])
        actions = episode.get("actions", [])
        if len(snapshots) <= tau or len(actions) < tau:
            episode_audits.append({"episode_id": episode_id, "eligible_samples": 0, "reason": "too_short"})
            continue
        graphs = [
            graph_with_relations(
                snapshot,
                include_all_sites=include_all_sites,
                coordinate_frame=coordinate_frame,
            )
            for snapshot in snapshots
        ]
        eligible = 0
        change_count = 0
        for start in range(0, len(snapshots) - tau):
            current = graphs[start]
            future = graphs[start + tau]
            current_edges = _edge_map(current)
            future_edges = _edge_map(future)
            common_keys = sorted(set(current_edges) & set(future_edges))
            relation_changes = {}
            relation_valid = {}
            for key in common_keys:
                now = current_edges[key]["relations"]
                later = future_edges[key]["relations"]
                for relation in RELATIONS:
                    now_record = now[relation]
                    later_record = later[relation]
                    label_key = f"{key[0]}->{key[1]}::{relation}"
                    valid = int(now_record["valid"] and later_record["valid"])
                    relation_valid[label_key] = valid
                    relation_changes[label_key] = int(
                        valid and now_record["value"] != later_record["value"]
                    )
            sample = {
                "episode_id": episode_id,
                "split": split_by_episode[episode_id],
                "task_id": episode.get("task_id"),
                "task_name": episode.get("task_name"),
                "start_step": start,
                "target_step": start + tau,
                "tau": tau,
                "action_window": [entry["action"] for entry in actions[start : start + tau]],
                "graph_t": current,
                "graph_target": future,
                "relation_valid": relation_valid,
                "relation_changes": relation_changes,
            }
            samples.append(sample)
            eligible += 1
            change_count += sum(relation_changes.values())
        episode_audits.append({"episode_id": episode_id, "eligible_samples": eligible, "change_count": change_count})

    stats = defaultdict(Counter)
    for sample in samples:
        for label_key, valid in sample["relation_valid"].items():
            _, relation = label_key.split("::", 1)
            stats[relation]["valid"] += int(valid)
            if valid:
                stats[relation]["changed"] += int(sample["relation_changes"][label_key])
    return {
        "dataset_version": "phase2r.v2-robot-base",
        "source_capture_status": payload.get("capture_status"),
        "privileged_oracle": True,
        "tau": tau,
        "margin": MARGIN,
        "coordinate_frame": coordinate_frame,
        "frame_contract": {
            "primary": "robot_base",
            "source_capture_frame": "raw_world",
            "quaternion_convention": "MuJoCo body_xquat raw wxyz ordering",
            "transform": "subtract robot0_base position then inverse-rotate",
            "requires_robot_base_pose": coordinate_frame == "robot_base",
        },
        "relation_definitions": {
            relation: (
                "robot_base_axis_margin"
                if relation in GEOMETRIC_RELATIONS
                else (
                    "mujoco_contact_or_capability_handler"
                    if relation == "contact"
                    else (
                        "libero_wrapper_then_geometry_fallback"
                        if relation in {"on", "inside"}
                        else (
                            "gripper_contact_and_closed_qpos"
                            if relation == "holding"
                            else "node_state_only_not_pairwise"
                        )
                    )
                )
            )
            for relation in RELATIONS
        },
        "split": {
            "unit": "episode",
            "assignments": split_by_episode,
            "normalization_fit_on": "train_only",
            "threshold_fit_on": "train_only",
        },
        "episodes": episode_audits,
        "samples": samples,
        "stats": {relation: dict(counter) for relation, counter in stats.items()},
        "audit": {
            "episode_count": len(episode_ids),
            "sample_count": len(samples),
            "changed_relation_count": sum(
                sum(sample["relation_changes"].values()) for sample in samples
            ),
            "nonzero_action_count": sum(
                any(abs(float(value)) > 0.0 for value in action)
                for episode in episodes
                for action_entry in episode.get("actions", [])
                for action in [action_entry.get("action", [])]
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tau", type=int, default=6)
    parser.add_argument("--include-all-sites", action="store_true")
    parser.add_argument(
        "--coordinate-frame",
        choices=["robot_base", "raw_world"],
        default="robot_base",
        help="Model-facing frame; robot_base requires robot_base_pose in every snapshot.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    dataset = build_dataset(
        payload,
        tau=args.tau,
        include_all_sites=args.include_all_sites,
        coordinate_frame=args.coordinate_frame,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dataset["audit"], ensure_ascii=False))
    print(json.dumps({"stats": dataset["stats"], "split": dataset["split"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
