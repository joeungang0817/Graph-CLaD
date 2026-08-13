"""Run fixed-manifest holder-object smoke comparisons.

The default keeps the original v1 path reproducible. ``--feature-contract
holder_object_v2`` enables the compact pair state, contact-aware edge, and
structured action variants without changing the frozen sample manifest.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping


def _load_probe():
    path = Path(__file__).with_name("offline_probe.py")
    spec = importlib.util.spec_from_file_location("phase3_offline_probe", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample_id(sample: Mapping[str, Any]) -> str:
    return "|".join(
        str(sample[key])
        for key in ("suite", "task_id", "episode_id", "start_step", "target_step", "tau")
    )


def _read_requested(root: Path, task_id: int, requested: set[str]) -> list[dict[str, Any]]:
    path = root / f"task{task_id}" / f"phase2d_task{task_id}_graph_dataset.jsonl.gz"
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            for raw in payload.get("samples", []):
                sample = dict(raw)
                sample["suite"] = str(sample.get("suite") or "libero_spatial")
                sample["task_id"] = int(sample.get("task_id", task_id))
                sample["episode_id"] = str(sample.get("episode_id"))
                if _sample_id(sample) in requested:
                    rows.append(sample)
    return rows


def _robot_object_topology(
    sample: Mapping[str, Any],
    topology: str,
    prune_non_pair_nodes: bool = False,
) -> dict[str, Any]:
    """Build sparse or complete robot/object messages with pair-only targets.

    Complete topology adds object-object edges only as message context. The
    prediction-edge contract remains robot-object for both topologies so the
    complete model does not receive extra supervised targets.
    """

    if topology not in {"sparse", "complete"}:
        raise ValueError(f"unknown robot/object topology={topology}")

    output = dict(sample)
    graph = copy.deepcopy(sample["graph_t"])
    target_graph = copy.deepcopy(sample["graph_target"])
    node_types = {str(node["node_id"]): str(node.get("node_type")) for node in graph.get("nodes", [])}
    retained_node_ids = {
        node_id for node_id, node_type in node_types.items()
        if node_type in {"robot", "object"}
    }
    target_keys = {
        (str(edge.get("source")), str(edge.get("target")))
        for edge in target_graph.get("edges", [])
    }

    def is_holder_object(edge: Mapping[str, Any]) -> bool:
        pair = {node_types.get(str(edge.get("source"))), node_types.get(str(edge.get("target")))}
        return pair == {"robot", "object"}

    def keep_message(edge: Mapping[str, Any]) -> bool:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        key = (source, target)
        if key not in target_keys or source == target:
            return False
        if topology == "sparse":
            return is_holder_object(edge)
        return source in retained_node_ids and target in retained_node_ids

    graph["edges"] = [
        edge for edge in graph.get("edges", []) if keep_message(edge)
    ]
    message_keys = [
        (str(edge.get("source")), str(edge.get("target")))
        for edge in graph["edges"]
    ]
    message_key_set = set(message_keys)
    target_graph["edges"] = [
        edge
        for edge in target_graph.get("edges", [])
        if (str(edge.get("source")), str(edge.get("target"))) in message_key_set
    ]
    output["prediction_edge_keys"] = [
        [source, target]
        for source, target in message_keys
        if {
            node_types.get(source),
            node_types.get(target),
        } == {"robot", "object"}
    ]
    if prune_non_pair_nodes:
        graph["nodes"] = [
            node for node in graph.get("nodes", [])
            if str(node.get("node_id")) in retained_node_ids
        ]
        target_graph["nodes"] = [
            node for node in target_graph.get("nodes", [])
            if str(node.get("node_id")) in retained_node_ids
        ]
    output["graph_t"] = graph
    output["graph_target"] = target_graph
    return output


def _holder_object_only(
    sample: Mapping[str, Any],
    prune_non_pair_nodes: bool = False,
) -> dict[str, Any]:
    """Compatibility wrapper for the sparse holder-object topology."""

    return _robot_object_topology(
        sample, "sparse", prune_non_pair_nodes=prune_non_pair_nodes
    )


def _manifest_rows(manifest: Mapping[str, Any], fold_name: str) -> dict[str, list[dict[str, Any]]]:
    fold = next(item for item in manifest["folds"] if item["name"] == fold_name)
    return {
        role: list(fold["sample_keys"][role])
        for role in ("train", "validation", "natural_test", "challenge_test")
    }


def run_smoke(
    manifest_path: Path,
    output_path: Path,
    fold_name: str,
    seed: int,
    feature_contract: str = "legacy_v1",
) -> dict[str, Any]:
    import torch

    probe = _load_probe()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass":
        raise ValueError(f"manifest status is not pass: {manifest.get('status')} {manifest.get('warnings')}")
    if feature_contract not in {"legacy_v1", "holder_object_v2"}:
        raise ValueError(f"unknown feature_contract={feature_contract}")
    is_v2 = feature_contract == "holder_object_v2"
    fold = next(item for item in manifest["folds"] if item["name"] == fold_name)
    roots = manifest["source_roots"]
    role_keys = _manifest_rows(manifest, fold_name)
    requested_by_source: dict[str, set[str]] = {"natural": set(), "target_aligned": set()}
    for role, keys in role_keys.items():
        for key in keys:
            requested_by_source["natural" if key["source"] == "natural" else "target_aligned"].add(key["sample_id"])
    rows_by_id: dict[str, dict[str, Any]] = {}
    for source_name, root_value in roots.items():
        if source_name == "split_manifest":
            continue
        root = Path(root_value)
        for task_id in (0, 1, 2):
            for sample in _read_requested(root, task_id, requested_by_source[source_name]):
                rows_by_id[f"{source_name}|{_sample_id(sample)}"] = sample

    dataset_samples: list[dict[str, Any]] = []
    challenge_samples: list[dict[str, Any]] = []
    for role in ("train", "validation", "natural_test", "challenge_test"):
        for key in role_keys[role]:
            source = "natural" if key["source"] == "natural" else "target_aligned"
            raw = rows_by_id[f"{source}|{key['sample_id']}"]
            sample = _holder_object_only(raw, prune_non_pair_nodes=is_v2)
            sample["split"] = "test" if role in {"natural_test", "challenge_test"} else role
            if role == "challenge_test":
                challenge_samples.append(sample)
            else:
                dataset_samples.append(sample)

    model_ids = (
        [
            "p0_flat_mlp",
            "b1_pair_feature_mlp_v2",
            "g1_sparse_holder_object_gnn",
            "g3_action_conditioned_holder_object_gnn",
            "g2_flat_action_holder_object_gnn_v2",
            "g2_structured_action_holder_object_gnn",
            "g3v2_action_film_holder_object_gnn",
        ]
        if is_v2 else
        [
            "p0_flat_mlp",
            "b1_target_object_mlp",
            "g1_sparse_holder_object_gnn",
            "g3_action_conditioned_holder_object_gnn",
        ]
    )
    base_config = {
        "target_relations": manifest["relations"],
        "models": model_ids,
        "seeds": [seed],
        "parameter_match": False,
        "hidden_dim": 48,
        "batch_size": 64,
        "epochs": 10,
        "patience": 3,
        "learning_rate": 0.001,
        "current_loss_weight": 0.25,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "node_feature_contract": "holder_object_v2" if is_v2 else "legacy_v1",
        "edge_feature_contract": "holder_object_v2" if is_v2 else "geometry_v1",
        "action_normalization_contract": "channel_v2" if is_v2 else "flat_position_v1",
    }
    load_options = {
        "relations": manifest["relations"],
        "node_feature_contract": base_config["node_feature_contract"],
        "edge_feature_contract": base_config["edge_feature_contract"],
    }
    records = probe.load_probe_records({"samples": dataset_samples}, **load_options)
    challenge_records = probe.load_probe_records({"samples": challenge_samples}, **load_options)
    if is_v2:
        action_layouts = {
            (int(row.get("action_steps", 0)), int(row.get("action_dim", 0)))
            for row in records + challenge_records
        }
        if action_layouts != {(6, 7)}:
            raise ValueError(
                "holder_object_v2 expects the frozen 6x7 LIBERO action window; "
                f"got {sorted(action_layouts)}"
            )
    train_records = [row for row in records if row["split"] == "train"]
    val_records = [row for row in records if row["split"] == "validation"]
    natural_records = [row for row in records if row["split"] == "test"]
    shape = probe.ProbeShape(
        max_nodes=max(len(row["node_features"]) for row in records),
        max_edges=max(len(row["edge_src"]) for row in records),
        node_dim=len(records[0]["node_features"][0]),
        edge_dim=len(records[0]["edge_geometry"][0]),
        action_dim=max(len(row["actions"]) for row in records),
        relation_dim=len(manifest["relations"]),
        action_steps=max(int(row.get("action_steps", 0)) for row in records),
        action_step_dim=max(int(row.get("action_dim", 0)) for row in records),
    )
    normalization = probe._normalization(
        train_records, base_config["action_normalization_contract"]
    )
    train_loader = probe._loader(train_records, shape, normalization, 64, shuffle=True)
    val_loader = probe._loader(val_records, shape, normalization, 64, shuffle=False)
    natural_loader = probe._loader(natural_records, shape, normalization, 64, shuffle=False)
    challenge_loader = probe._loader(challenge_records, shape, normalization, 64, shuffle=False)
    device = base_config["device"]
    results: list[dict[str, Any]] = []
    evaluation_modes = ["correct", "no_action", "shuffled_action", "shuffled_edge"]
    if is_v2:
        evaluation_modes.extend([
            "physical_zero_action",
            "reversed_action",
            "shuffled_gripper",
            "shuffled_arm",
        ])
    for model_id in base_config["models"]:
        probe._set_seed(seed)
        model, training = probe.train_one(
            model_id, shape, train_records, val_loader, train_loader, device,
            base_config["hidden_dim"], base_config["epochs"], base_config["patience"],
            base_config["learning_rate"], base_config["current_loss_weight"],
            relations=manifest["relations"],
        )
        natural_eval = {
            mode: probe.evaluate_model(
                model,
                natural_loader,
                device,
                mode=mode,
                relations=manifest["relations"],
                holding_threshold=training["holding_threshold"],
            )
            for mode in evaluation_modes
        }
        challenge_eval = {
            mode: probe.evaluate_model(
                model,
                challenge_loader,
                device,
                mode=mode,
                relations=manifest["relations"],
                holding_threshold=training["holding_threshold"],
            )
            for mode in evaluation_modes
        }
        results.append({"model_id": model_id, "seed": seed, "training": training, "natural": natural_eval, "challenge": challenge_eval})
        print(json.dumps({
            "model_id": model_id,
            "natural_changed_f1": natural_eval["correct"]["changed_relation"]["macro_f1"],
            "challenge_changed_f1": challenge_eval["correct"]["changed_relation"]["macro_f1"],
            "natural_holding_change_event_f1": natural_eval["correct"]["holding"]["change_event"]["f1"],
            "challenge_holding_change_event_f1": challenge_eval["correct"]["holding"]["change_event"]["f1"],
            "holding_threshold": training["holding_threshold"],
        }, ensure_ascii=False))
    output = {
        "protocol": f"phase3B-R1-holder-action-smoke-{'v2' if is_v2 else 'v1'}",
        "fold": fold,
        "seed": seed,
        "model_scope": (
            "B0/B1, sparse G1 flat late action, sparse G3 flat gated action, "
            "matched G2-flat/G2-structured late-action blocks, and G3v2 "
            "structured pair-conditioned FiLM"
            if is_v2 else
            "B0 flat, B1 current-information robot-object pair MLP, G1 one-layer sparse holder-object GNN with global action concat, G3 one-layer sparse holder-object GNN with action-conditioned edge messages and gate"
        ),
        "target_selection": "current graph robot-object pair restriction; graph_target and target_categories are not used for input selection",
        "feature_scope": (
            "compact node state (type, validity, gripper qpos/aperture, joint velocity); "
            "relative position/distance, current contact and edge direction; structured "
            "6x7 action window with step mask; no current holding, future field, past graph "
            "history, relative velocity, or object-following stability"
            if is_v2 else
            "legacy node vector, relative position/distance, flattened action window; "
            "no current contact edge input, history, relative velocity, or stability"
        ),
        "feature_contract": {
            "node": base_config["node_feature_contract"],
            "edge": base_config["edge_feature_contract"],
            "action_normalization": base_config["action_normalization_contract"],
            "current_holding_is_input": False,
            "node_feature_names": list(probe.HOLDER_OBJECT_V2_NODE_FEATURES) if is_v2 else None,
            "edge_feature_names": list(probe.HOLDER_OBJECT_V2_EDGE_FEATURES) if is_v2 else None,
        },
        "split_counts": {"train": len(train_records), "validation": len(val_records), "natural_test": len(natural_records), "challenge_test": len(challenge_records)},
        "shape": shape.__dict__,
        "config": base_config,
        "evaluation_modes": evaluation_modes,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", default="test_task0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--feature-contract",
        choices=("legacy_v1", "holder_object_v2"),
        default="legacy_v1",
    )
    args = parser.parse_args()
    result = run_smoke(
        args.manifest, args.output, args.fold, args.seed, args.feature_contract
    )
    print(json.dumps({"status": "completed", "output": str(args.output), "split_counts": result["split_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
