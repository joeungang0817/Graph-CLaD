"""Run the paired task-0 three-seed action-semantics gate.

The gate separates four explanations for G1's smoke-test gain:

* G1-correct: the original sample-aligned action pathway;
* S-0-G1: the exact G1-style block trained without an action pathway;
* G1-constant: an equal-parameter G1 branch that receives one fixed action
  template for every sample;
* G1-train-shuffled: the original G1 trained with the batch action marginal
  preserved but every scene-action pairing cyclically shifted.

All models reuse the frozen manifest, sparse holder-object topology, train-only
normalization, validation-selected holding threshold, and natural/challenge
test sample IDs.  Results are checkpointed after every run for Colab recovery.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase3 import run_topology_action_followup as followup


COMPARISONS = (
    {
        "comparison_id": "G1-correct",
        "model_id": "g1_sparse_holder_object_gnn",
        "action_information": "sample_aligned",
        "training_action_mode": "correct",
    },
    {
        "comparison_id": "S-0-G1",
        "model_id": "s0_g1_no_action_holder_object_gnn_v2",
        "action_information": "none",
        "training_action_mode": "correct",
    },
    {
        "comparison_id": "G1-constant",
        "model_id": "g1_constant_action_holder_object_gnn_v2",
        "action_information": "constant_template",
        "training_action_mode": "correct",
    },
    {
        "comparison_id": "G1-train-shuffled",
        "model_id": "g1_sparse_holder_object_gnn",
        "action_information": "marginal_preserved_alignment_destroyed",
        "training_action_mode": "shuffled_batch",
    },
)


def _holding_metric(evaluation: Mapping[str, Any], metric: str) -> float | None:
    holding = evaluation["holding"]
    if metric == "event_f1":
        value = holding["change_event"]["f1"]
    elif metric == "event_pr_auc":
        value = holding["change_event"]["pr_auc"]
    elif metric == "onset_f1":
        value = holding["onset"]["f1"]
    elif metric == "release_f1":
        value = holding["release"]["f1"]
    elif metric == "hard_negative_fpr":
        value = holding["hard_negative"]["false_positive_rate"]
    else:
        raise ValueError(metric)
    return None if value is None else float(value)


def _summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = (
        "event_f1",
        "event_pr_auc",
        "onset_f1",
        "release_f1",
        "hard_negative_fpr",
    )
    by_key = {
        (str(item["comparison_id"]), int(item["seed"])): item for item in results
    }
    seeds = sorted({int(item["seed"]) for item in results})
    aggregates: dict[str, Any] = {}
    for comparison in (str(spec["comparison_id"]) for spec in COMPARISONS):
        aggregates[comparison] = {}
        for split in ("natural", "challenge"):
            aggregates[comparison][split] = {}
            for metric in metrics:
                values = [
                    _holding_metric(by_key[(comparison, seed)][split]["correct"], metric)
                    for seed in seeds
                    if (comparison, seed) in by_key
                ]
                finite = [value for value in values if value is not None]
                aggregates[comparison][split][metric] = {
                    "mean": mean(finite) if finite else None,
                    "std": pstdev(finite) if len(finite) > 1 else (0.0 if finite else None),
                    "n": len(finite),
                    "values": finite,
                }

    paired: dict[str, Any] = {}
    for control in ("S-0-G1", "G1-constant", "G1-train-shuffled"):
        key = f"G1-correct_minus_{control}"
        paired[key] = {}
        for split in ("natural", "challenge"):
            differences = []
            for seed in seeds:
                if ("G1-correct", seed) not in by_key or (control, seed) not in by_key:
                    continue
                correct = _holding_metric(
                    by_key[("G1-correct", seed)][split]["correct"], "event_f1"
                )
                baseline = _holding_metric(
                    by_key[(control, seed)][split]["correct"], "event_f1"
                )
                if correct is not None and baseline is not None:
                    differences.append(correct - baseline)
            paired[key][split] = {
                "mean": mean(differences) if differences else None,
                "std": (
                    pstdev(differences)
                    if len(differences) > 1
                    else (0.0 if differences else None)
                ),
                "n": len(differences),
                "values": differences,
                "positive_count": sum(value > 0 for value in differences),
            }
    return {"aggregates": aggregates, "paired_event_f1": paired}


def _write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def run_gate(
    manifest_path: Path,
    output_path: Path,
    fold_name: str = "test_task0",
    seeds: Sequence[int] = (0, 1, 2),
    resume: bool = False,
) -> dict[str, Any]:
    import torch

    smoke = followup._load_smoke_module()
    probe = smoke._load_probe()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass":
        raise ValueError(f"manifest status is not pass: {manifest.get('status')}")
    relations = [str(value) for value in manifest["relations"]]
    selected = followup._selected_samples(manifest, fold_name)
    records = followup._topology_records(
        selected, "sparse", relations, smoke, probe
    )
    shape = followup._shape(records, len(relations), probe)
    if (shape.action_steps, shape.action_step_dim) != (6, 7):
        raise ValueError(
            "expected frozen 6x7 action window, got "
            f"{shape.action_steps}x{shape.action_step_dim}"
        )

    normalization = probe._normalization(records["train"], "channel_v2")
    target_parameter_count = probe._parameter_count(
        probe.RelationalDynamicsProbe(
            "g1_sparse_holder_object_gnn", shape, hidden_dim=48
        )
    )
    s0_hidden_dim = probe._select_hidden_dims(
        ["s0_g1_no_action_holder_object_gnn_v2"],
        shape,
        {
            "parameter_match": True,
            "target_parameter_count": target_parameter_count,
            "candidate_hidden_dims": list(range(40, 81)),
        },
    )["s0_g1_no_action_holder_object_gnn_v2"]
    specs = []
    for template in COMPARISONS:
        spec = dict(template)
        spec["hidden_dim"] = s0_hidden_dim if spec["comparison_id"] == "S-0-G1" else 48
        model = probe.RelationalDynamicsProbe(
            spec["model_id"], shape, hidden_dim=spec["hidden_dim"]
        )
        spec["parameter_count"] = probe._parameter_count(model)
        spec["parameter_difference_from_g1"] = (
            spec["parameter_count"] - target_parameter_count
        )
        specs.append(spec)

    prior_results: list[dict[str, Any]] = []
    if resume and output_path.exists():
        prior = json.loads(output_path.read_text(encoding="utf-8"))
        if prior.get("protocol") != "phase3B-R1-action-semantics-gate-v1":
            raise ValueError("resume output uses a different protocol")
        if prior.get("fold") != fold_name or prior.get("seeds") != list(seeds):
            raise ValueError("resume output fold/seeds do not match")
        prior_results = list(prior.get("results", []))
    completed = {
        (str(item["comparison_id"]), int(item["seed"])) for item in prior_results
    }
    results = prior_results
    evaluation_modes = (
        "correct",
        "no_action",
        "shuffled_action",
        "shuffled_edge",
        "physical_zero_action",
        "reversed_action",
        "shuffled_gripper",
        "shuffled_arm",
    )
    expected_runs = len(specs) * len(seeds)

    def payload(status: str) -> dict[str, Any]:
        return {
            "protocol": "phase3B-R1-action-semantics-gate-v1",
            "status": status,
            "fold": fold_name,
            "seeds": list(seeds),
            "manifest": str(manifest_path),
            "relations": relations,
            "split_counts": {role: len(rows) for role, rows in records.items()},
            "shape": shape.__dict__,
            "topology": "sparse holder-object directed edges",
            "normalization": "channel_v2 fit on train split only",
            "checkpoint": "validation holding change-event F1",
            "threshold": "fit on validation correct-action output, frozen on tests",
            "target_parameter_count": target_parameter_count,
            "model_specs": specs,
            "evaluation_modes": list(evaluation_modes),
            "completed_runs": len(results),
            "expected_runs": expected_runs,
            "results": results,
            "summary": _summary(results),
        }

    _write_checkpoint(output_path, payload("running"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for seed in seeds:
        for spec in specs:
            run_key = (str(spec["comparison_id"]), int(seed))
            if run_key in completed:
                continue
            probe._set_seed(int(seed))
            train_loader = probe._loader(
                records["train"], shape, normalization, 64, True
            )
            validation_loader = probe._loader(
                records["validation"], shape, normalization, 64, False
            )
            natural_loader = probe._loader(
                records["natural_test"], shape, normalization, 64, False
            )
            challenge_loader = probe._loader(
                records["challenge_test"], shape, normalization, 64, False
            )
            model, training = probe.train_one(
                spec["model_id"],
                shape,
                records["train"],
                validation_loader,
                train_loader,
                device,
                spec["hidden_dim"],
                10,
                3,
                0.001,
                0.25,
                relations=relations,
                training_action_mode=spec["training_action_mode"],
            )
            evaluations: dict[str, Any] = {}
            for split, loader in (
                ("natural", natural_loader),
                ("challenge", challenge_loader),
            ):
                evaluations[split] = {
                    mode: probe.evaluate_model(
                        model,
                        loader,
                        device,
                        mode=mode,
                        relations=relations,
                        holding_threshold=training["holding_threshold"],
                    )
                    for mode in evaluation_modes
                }
            action_invariance = None
            if spec["comparison_id"] in {"S-0-G1", "G1-constant"}:
                action_modes = [
                    mode
                    for mode in evaluation_modes
                    if mode not in {"correct", "shuffled_edge"}
                ]
                action_invariance = max(
                    abs(
                        _holding_metric(evaluations[split]["correct"], "event_f1")
                        - _holding_metric(evaluations[split][mode], "event_f1")
                    )
                    for split in ("natural", "challenge")
                    for mode in action_modes
                )
                if action_invariance > 1e-12:
                    raise AssertionError(
                        f"{spec['comparison_id']} changed under action controls: "
                        f"{action_invariance}"
                    )
            result = {
                **spec,
                "seed": int(seed),
                "training": training,
                "action_invariance_max_abs_event_f1_delta": action_invariance,
                **evaluations,
            }
            results.append(result)
            completed.add(run_key)
            _write_checkpoint(output_path, payload("running"))
            print(
                json.dumps(
                    {
                        "comparison_id": spec["comparison_id"],
                        "seed": seed,
                        "params": training["parameter_count"],
                        "threshold": training["holding_threshold"],
                        "training_action_mode": training["training_action_mode"],
                        "natural_event_f1": _holding_metric(
                            evaluations["natural"]["correct"], "event_f1"
                        ),
                        "challenge_event_f1": _holding_metric(
                            evaluations["challenge"]["correct"], "event_f1"
                        ),
                        "completed_runs": len(results),
                        "expected_runs": expected_runs,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    final = payload("completed")
    _write_checkpoint(output_path, final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", default="test_task0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_gate(
        args.manifest,
        args.output,
        fold_name=args.fold,
        seeds=args.seeds,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "completed_runs": result["completed_runs"],
                "expected_runs": result["expected_runs"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
