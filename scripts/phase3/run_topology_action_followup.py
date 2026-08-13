"""Run the S-0, C-L, and C-E fixed-manifest follow-up smoke.

S-0 removes the action encoder from the matched v2 sparse block. C-L and C-E
use the same robot/object nodes and prediction targets as their sparse
counterparts, but add object-object message edges. C-L uses structured late
action and C-E uses the same encoder with pair-conditioned FiLM.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping


def _load_smoke_module():
    path = Path(__file__).with_name("run_holder_action_smoke.py")
    spec = importlib.util.spec_from_file_location("phase3_holder_action_smoke", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _selected_samples(
    manifest: Mapping[str, Any], fold_name: str
) -> dict[str, list[dict[str, Any]]]:
    smoke = _load_smoke_module()
    role_keys = smoke._manifest_rows(manifest, fold_name)
    requested_by_source: dict[str, set[str]] = {
        "natural": set(),
        "target_aligned": set(),
    }
    for keys in role_keys.values():
        for key in keys:
            source = "natural" if key["source"] == "natural" else "target_aligned"
            requested_by_source[source].add(key["sample_id"])

    rows_by_id: dict[str, dict[str, Any]] = {}
    for source_name, root_value in manifest["source_roots"].items():
        if source_name == "split_manifest":
            continue
        for task_id in (0, 1, 2):
            rows = smoke._read_requested(
                Path(root_value), task_id, requested_by_source[source_name]
            )
            for sample in rows:
                rows_by_id[f"{source_name}|{smoke._sample_id(sample)}"] = sample

    selected: dict[str, list[dict[str, Any]]] = {}
    for role, keys in role_keys.items():
        selected[role] = []
        for key in keys:
            source = "natural" if key["source"] == "natural" else "target_aligned"
            selected[role].append(rows_by_id[f"{source}|{key['sample_id']}"])
    return selected


def _topology_records(
    selected: Mapping[str, list[dict[str, Any]]],
    topology: str,
    relations: list[str],
    smoke,
    probe,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    load_options = {
        "relations": relations,
        "node_feature_contract": "holder_object_v2",
        "edge_feature_contract": "holder_object_v2",
    }
    for role, samples in selected.items():
        prepared: list[dict[str, Any]] = []
        for raw in samples:
            sample = smoke._robot_object_topology(
                raw, topology, prune_non_pair_nodes=True
            )
            sample["split"] = (
                "test" if role in {"natural_test", "challenge_test"} else role
            )
            prepared.append(sample)
        output[role] = probe.load_probe_records(
            {"samples": prepared}, **load_options
        )
    return output


def _shape(records_by_role: Mapping[str, list[dict[str, Any]]], relation_dim: int, probe):
    records = [row for rows in records_by_role.values() for row in rows]
    return probe.ProbeShape(
        max_nodes=max(len(row["node_features"]) for row in records),
        max_edges=max(len(row["edge_src"]) for row in records),
        node_dim=len(records[0]["node_features"][0]),
        edge_dim=len(records[0]["edge_geometry"][0]),
        action_dim=max(len(row["actions"]) for row in records),
        relation_dim=relation_dim,
        action_steps=max(int(row.get("action_steps", 0)) for row in records),
        action_step_dim=max(int(row.get("action_dim", 0)) for row in records),
    )


def _holding_f1(evaluation: Mapping[str, Any]) -> float:
    return float(evaluation["holding"]["change_event"]["f1"])


def run_followup(
    manifest_path: Path,
    output_path: Path,
    fold_name: str,
    seed: int,
    comparison_ids: list[str] | None = None,
    parameter_match: bool = False,
    target_parameter_count: int = 60235,
) -> dict[str, Any]:
    import torch

    smoke = _load_smoke_module()
    probe = smoke._load_probe()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass":
        raise ValueError(f"manifest status is not pass: {manifest.get('status')}")
    relations = [str(value) for value in manifest["relations"]]
    model_specs = [
        {
            "comparison_id": "S-0",
            "model_id": "s0_no_action_holder_object_gnn_v2",
            "topology": "sparse",
            "action_routing": "none",
        },
        {
            "comparison_id": "S-0-G1",
            "model_id": "s0_g1_no_action_holder_object_gnn_v2",
            "topology": "sparse",
            "action_routing": "none_g1_exact_ablation",
        },
        {
            "comparison_id": "C-L",
            "model_id": "c_l_complete_late_action_gnn_v2",
            "topology": "complete",
            "action_routing": "structured_late",
        },
        {
            "comparison_id": "C-E",
            "model_id": "c_e_complete_action_film_gnn_v2",
            "topology": "complete",
            "action_routing": "structured_edge_film",
        },
    ]
    if comparison_ids:
        requested = set(comparison_ids)
        known = {spec["comparison_id"] for spec in model_specs}
        if not requested <= known:
            raise ValueError(f"unknown comparison IDs: {sorted(requested - known)}")
        model_specs = [
            spec for spec in model_specs if spec["comparison_id"] in requested
        ]

    selected = _selected_samples(manifest, fold_name)
    sparse = _topology_records(selected, "sparse", relations, smoke, probe)
    sparse_shape = _shape(sparse, len(relations), probe)
    topologies = {"sparse": sparse}
    shapes = {"sparse": sparse_shape}
    if any(spec["topology"] == "complete" for spec in model_specs):
        complete = _topology_records(selected, "complete", relations, smoke, probe)
        complete_shape = _shape(complete, len(relations), probe)
        if (
            sparse_shape.node_dim != complete_shape.node_dim
            or sparse_shape.edge_dim != complete_shape.edge_dim
            or sparse_shape.action_dim != complete_shape.action_dim
            or sparse_shape.max_nodes != complete_shape.max_nodes
        ):
            raise ValueError("sparse and complete contracts differ beyond edge count")
        topologies["complete"] = complete
        shapes["complete"] = complete_shape

    action_layouts = {
        (int(row.get("action_steps", 0)), int(row.get("action_dim", 0)))
        for topology_rows in topologies.values()
        for rows in topology_rows.values()
        for row in rows
    }
    if action_layouts != {(6, 7)}:
        raise ValueError(f"expected frozen 6x7 action window, got {sorted(action_layouts)}")

    # Freeze preprocessing to the sparse holder-object training distribution.
    # C-L/C-E therefore differ from S-LS/S-EF only by message topology.
    normalization = probe._normalization(sparse["train"], "channel_v2")
    bundles: dict[str, dict[str, Any]] = {}
    for topology, rows in topologies.items():
        shape = shapes[topology]
        bundles[topology] = {
            "train": probe._loader(rows["train"], shape, normalization, 64, True),
            "validation": probe._loader(
                rows["validation"], shape, normalization, 64, False
            ),
            "natural": probe._loader(
                rows["natural_test"], shape, normalization, 64, False
            ),
            "challenge": probe._loader(
                rows["challenge_test"], shape, normalization, 64, False
            ),
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for spec in model_specs:
        if parameter_match:
            spec["hidden_dim"] = probe._select_hidden_dims(
                [spec["model_id"]],
                shapes[spec["topology"]],
                {
                    "parameter_match": True,
                    "target_parameter_count": target_parameter_count,
                    "candidate_hidden_dims": list(range(40, 81)),
                },
            )[spec["model_id"]]
        else:
            spec["hidden_dim"] = 48
    evaluation_modes = [
        "correct",
        "no_action",
        "shuffled_action",
        "shuffled_edge",
        "physical_zero_action",
        "reversed_action",
        "shuffled_gripper",
        "shuffled_arm",
    ]
    results: list[dict[str, Any]] = []
    for spec in model_specs:
        topology = spec["topology"]
        rows = topologies[topology]
        loaders = bundles[topology]
        shape = shapes[topology]
        probe._set_seed(seed)
        model, training = probe.train_one(
            spec["model_id"],
            shape,
            rows["train"],
            loaders["validation"],
            loaders["train"],
            device,
            spec["hidden_dim"],
            10,
            3,
            0.001,
            0.25,
            relations=relations,
        )
        evaluations: dict[str, dict[str, Any]] = {}
        for split in ("natural", "challenge"):
            evaluations[split] = {
                mode: probe.evaluate_model(
                    model,
                    loaders[split],
                    device,
                    mode=mode,
                    relations=relations,
                    holding_threshold=training["holding_threshold"],
                )
                for mode in evaluation_modes
            }
        action_invariance = None
        if spec["comparison_id"] in {"S-0", "S-0-G1"}:
            action_modes = [
                mode for mode in evaluation_modes
                if mode not in {"correct", "shuffled_edge"}
            ]
            action_invariance = max(
                abs(
                    _holding_f1(evaluations[split]["correct"])
                    - _holding_f1(evaluations[split][mode])
                )
                for split in ("natural", "challenge")
                for mode in action_modes
            )
            if action_invariance > 1e-12:
                raise AssertionError(
                    f"S-0 changed under action-only controls: {action_invariance}"
                )
        result = {
            **spec,
            "seed": seed,
            "shape": shape.__dict__,
            "training": training,
            "action_invariance_max_abs_event_f1_delta": action_invariance,
            **evaluations,
        }
        results.append(result)
        print(json.dumps({
            "comparison_id": spec["comparison_id"],
            "model_id": spec["model_id"],
            "topology": topology,
            "hidden_dim": spec["hidden_dim"],
            "parameter_count": training["parameter_count"],
            "holding_threshold": training["holding_threshold"],
            "natural_holding_change_event_f1": _holding_f1(evaluations["natural"]["correct"]),
            "challenge_holding_change_event_f1": _holding_f1(evaluations["challenge"]["correct"]),
        }, ensure_ascii=False), flush=True)

    output = {
        "protocol": "phase3B-R1-topology-action-followup-smoke-v2",
        "fold": fold_name,
        "seed": seed,
        "relations": relations,
        "split_counts": {role: len(rows) for role, rows in sparse.items()},
        "parameter_match": parameter_match,
        "target_parameter_count": target_parameter_count if parameter_match else None,
        "hidden_dims": {
            spec["comparison_id"]: spec["hidden_dim"] for spec in model_specs
        },
        "normalization_source": "sparse holder-object train split, reused by complete topology",
        "topology_contract": {
            "nodes": "identical robot/object nodes",
            "prediction_edges": "robot-object only for loss and evaluation",
            "sparse_message_edges": "robot-object directed edges",
            "complete_message_edges": "all directed non-self robot/object pairs",
        },
        "model_specs": model_specs,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", default="test_task0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--only",
        action="append",
        choices=("S-0", "S-0-G1", "C-L", "C-E"),
        help="Run only the selected comparison ID; repeat to select multiple.",
    )
    parser.add_argument("--parameter-match", action="store_true")
    parser.add_argument("--target-parameter-count", type=int, default=60235)
    args = parser.parse_args()
    result = run_followup(
        args.manifest,
        args.output,
        args.fold,
        args.seed,
        comparison_ids=args.only,
        parameter_match=args.parameter_match,
        target_parameter_count=args.target_parameter_count,
    )
    print(json.dumps({
        "status": "completed",
        "output": str(args.output),
        "models": [item["comparison_id"] for item in result["results"]],
        "split_counts": result["split_counts"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
