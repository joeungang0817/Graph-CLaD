"""Validate Phase 2R pilot integrity, relations, and change attribution."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


INVERSE_RELATIONS = {
    "left": "right",
    "right": "left",
    "front": "behind",
    "behind": "front",
    "above": "below",
    "below": "above",
    "contact": "contact",
}
UNDIRECTED_FAMILIES = {
    "left": "horizontal_x",
    "right": "horizontal_x",
    "front": "depth_y",
    "behind": "depth_y",
    "above": "vertical_z",
    "below": "vertical_z",
    "contact": "contact",
}


def _edge_map(graph: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(edge["source"]), str(edge["target"])): edge for edge in graph.get("edges", [])}


def _node_type_map(graph: Mapping[str, Any]) -> dict[str, str]:
    return {str(node["node_id"]): str(node["node_type"]) for node in graph.get("nodes", [])}


def _record_is_valid(record: Mapping[str, Any]) -> bool:
    return bool(record.get("valid", 0))


def _record_value(record: Mapping[str, Any]) -> Any:
    return record.get("value")


def _changed_key_parts(key: str) -> tuple[str, str, str] | None:
    try:
        pair, relation = key.split("::", 1)
        source, target = pair.split("->", 1)
        return source, target, relation
    except ValueError:
        return None


def validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    samples = payload.get("samples", [])
    split = payload.get("split", {})
    assignments = split.get("assignments", {})
    split_sets = defaultdict(set)
    for episode_id, split_name in assignments.items():
        split_sets[str(split_name)].add(str(episode_id))
    split_overlap = []
    split_names = sorted(split_sets)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = split_sets[left] & split_sets[right]
            split_overlap.extend(f"{left}:{episode}" for episode in sorted(overlap))
    if split_overlap:
        errors.append(f"episode appears in multiple splits: {split_overlap}")

    audit = {
        "sample_count": len(samples),
        "episode_count": len(assignments),
        "tau": int(payload.get("tau", 0)),
        "invalid_samples": 0,
        "duplicate_node_graphs": 0,
        "edge_reverse_violations": 0,
        "relation_inverse_violations": 0,
        "feature_dim_violations": 0,
        "action_length_violations": 0,
        "temporal_alignment_violations": 0,
    }
    changed_by_relation = Counter()
    changed_by_type_pair = Counter()
    changed_robot = 0
    changed_nonrobot = 0
    changed_undirected: set[tuple[str, str, str]] = set()
    valid_semantic = Counter()
    graph_count = 0

    for sample_index, sample in enumerate(samples):
        tau = int(sample.get("tau", audit["tau"]))
        if len(sample.get("action_window", [])) != tau:
            audit["action_length_violations"] += 1
            errors.append(f"sample {sample_index}: action window length != tau")
        if int(sample.get("target_step", -1)) - int(sample.get("start_step", -1)) != tau:
            audit["temporal_alignment_violations"] += 1
            errors.append(f"sample {sample_index}: target/start step mismatch")
        current = sample.get("graph_t", {})
        target = sample.get("graph_target", {})
        for graph_name, graph in (("current", current), ("target", target)):
            graph_count += 1
            nodes = graph.get("nodes", [])
            node_ids = [str(node.get("node_id")) for node in nodes]
            if len(node_ids) != len(set(node_ids)):
                audit["duplicate_node_graphs"] += 1
                errors.append(f"sample {sample_index} {graph_name}: duplicate node id")
            for node in nodes:
                if len(node.get("feature_vector", [])) != int(graph.get("node_feature_dim", 24)):
                    audit["feature_dim_violations"] += 1
                    errors.append(f"sample {sample_index} {graph_name}: feature dimension mismatch")
            edges = _edge_map(graph)
            for source, target_id in edges:
                if (target_id, source) not in edges:
                    audit["edge_reverse_violations"] += 1
            if graph_name == "current":
                current_edges = edges
                current_types = _node_type_map(graph)
            else:
                target_edges = edges
        if set(current_types) != set(_node_type_map(target)):
            audit["invalid_samples"] += 1
            errors.append(f"sample {sample_index}: node identity changed across target")
        if set(current_edges) != set(target_edges):
            audit["invalid_samples"] += 1
            errors.append(f"sample {sample_index}: edge identity changed across target")

        for (source, target_id), edge in current_edges.items():
            reverse = current_edges.get((target_id, source))
            if reverse is None:
                continue
            relations = edge.get("relations", {})
            reverse_relations = reverse.get("relations", {})
            for relation, inverse in INVERSE_RELATIONS.items():
                left = relations.get(relation, {})
                right = reverse_relations.get(inverse, {})
                if _record_is_valid(left) and _record_is_valid(right):
                    if _record_value(left) != _record_value(right):
                        audit["relation_inverse_violations"] += 1
            for relation, record in relations.items():
                if _record_is_valid(record) and relation in {"on", "inside", "holding", "open", "close"}:
                    valid_semantic[relation] += 1

        for key, changed in sample.get("relation_changes", {}).items():
            if not changed:
                continue
            parts = _changed_key_parts(str(key))
            if parts is None:
                errors.append(f"sample {sample_index}: malformed relation change key {key!r}")
                continue
            source, target_id, relation = parts
            changed_by_relation[relation] += 1
            source_type = current_types.get(source, "unknown")
            target_type = current_types.get(target_id, "unknown")
            changed_by_type_pair[f"{source_type}->{target_type}"] += 1
            if source == "robot0" or target_id == "robot0":
                changed_robot += 1
            else:
                changed_nonrobot += 1
            family = UNDIRECTED_FAMILIES.get(relation)
            if family is not None:
                changed_undirected.add((family, *sorted((source, target_id))))

    if audit["edge_reverse_violations"]:
        errors.append("directed topology is not reciprocal")
    if audit["relation_inverse_violations"]:
        errors.append("inverse relation consistency failed")
    if audit["feature_dim_violations"]:
        errors.append("feature dimension consistency failed")
    if audit["action_length_violations"] or audit["temporal_alignment_violations"]:
        errors.append("action/target temporal alignment failed")
    if not valid_semantic:
        warnings.append("semantic relation labels are still entirely invalid")
    coordinate_frame = str(payload.get("coordinate_frame", ""))
    if coordinate_frame.startswith("raw_world_pilot"):
        warnings.append("coordinate frame is raw-world pilot; freeze robot-base/task-local before scale-up")
    total_changed = changed_robot + changed_nonrobot
    robot_share = changed_robot / total_changed if total_changed else 0.0
    if robot_share > 0.5:
        warnings.append("more than half of directed relation changes involve robot0; inspect for motion-induced changes")

    return {
        "validation_version": "phase2r-validation.v1",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "audit": audit,
        "change_attribution": {
            "directed_changed_count": total_changed,
            "undirected_changed_pair_count": len(changed_undirected),
            "robot_involved_directed_count": changed_robot,
            "nonrobot_directed_count": changed_nonrobot,
            "robot_involved_share": robot_share,
            "by_relation": dict(changed_by_relation),
            "by_node_type_pair": dict(changed_by_type_pair),
        },
        "semantic_valid_counts": dict(valid_semantic),
        "graph_count_checked": graph_count,
        "gate_results": {
            "episode_split_disjoint": not split_overlap,
            "temporal_alignment": audit["action_length_violations"] == 0 and audit["temporal_alignment_violations"] == 0,
            "node_alignment": audit["invalid_samples"] == 0,
            "directed_topology_reciprocal": audit["edge_reverse_violations"] == 0,
            "inverse_relation_consistency": audit["relation_inverse_violations"] == 0,
            "pilot_has_nonzero_changed_relations": bool(changed_by_relation),
            "semantic_label_contract": bool(valid_semantic),
            "coordinate_frame_finalized": not coordinate_frame.startswith("raw_world_pilot"),
        },
    }


def make_change_plot(payload: Mapping[str, Any], output_path: Path) -> bool:
    """Render one changed sample as a compact top-down before/after audit plot."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False
    sample = next(
        (sample for sample in payload.get("samples", []) if sum(sample.get("relation_changes", {}).values()) > 0),
        None,
    )
    if sample is None:
        return False
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, graph, title in zip(
        axes,
        (sample["graph_t"], sample["graph_target"]),
        (f"before step {sample['start_step']}", f"after step {sample['target_step']}"),
    ):
        for node in graph.get("nodes", []):
            position = node.get("features", {}).get("position", [0.0, 0.0, 0.0])
            color = {"robot": "black", "object": "tab:blue", "fixture": "tab:orange", "site": "tab:green"}.get(node.get("node_type"), "gray")
            axis.scatter(position[0], position[1], c=color, s=55)
            axis.text(position[0], position[1], str(node.get("node_id"))[:16], fontsize=7)
        axis.set_title(title)
        axis.set_xlabel("world x")
        axis.set_ylabel("world y")
        axis.grid(alpha=0.25)
    figure.suptitle(f"Phase 2R pilot audit: {sample['episode_id']} tau={sample['tau']}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = validate_payload(payload)
    if args.plot_output:
        report["plot_created"] = make_change_plot(payload, args.plot_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "audit": report["audit"], "change_attribution": report["change_attribution"], "warnings": report["warnings"]}, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
