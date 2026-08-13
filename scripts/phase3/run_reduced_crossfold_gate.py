"""Run the parameter-matched reduced Phase 3B cross-fold gate.

The four paired conditions are B1-v2, G1-correct, exact G1-style S-0, and G1
trained with batch-shuffled actions.  Every fold uses the frozen manifest,
sparse topology, validation-fitted threshold, global episode-disjoint action
shuffle, and edge shuffle.  A JSON checkpoint and a model checkpoint are saved
after every fold/seed/model run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase3 import run_topology_action_followup as followup


MODEL_TEMPLATES = (
    {
        "comparison_id": "B1-v2",
        "model_id": "b1_pair_feature_mlp_v2",
        "action_information": "pair_feature_with_aligned_action",
        "training_action_mode": "correct",
    },
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
        "comparison_id": "G1-train-shuffled",
        "model_id": "g1_sparse_holder_object_gnn",
        "action_information": "alignment_destroyed",
        "training_action_mode": "shuffled_batch",
    },
)
EVAL_MODES = (
    "correct",
    "global_shuffled_action",
    "shuffled_action",
    "no_action",
    "physical_zero_action",
    "reversed_action",
    "shuffled_gripper",
    "shuffled_arm",
    "shuffled_edge",
)
METRICS = (
    "event_f1",
    "event_pr_auc",
    "onset_f1",
    "release_f1",
    "hard_negative_fpr",
)


def _metric(evaluation: Mapping[str, Any], metric: str) -> float | None:
    paths = {
        "event_f1": ("change_event", "f1"),
        "event_pr_auc": ("change_event", "pr_auc"),
        "onset_f1": ("onset", "f1"),
        "release_f1": ("release", "f1"),
        "hard_negative_fpr": ("hard_negative", "false_positive_rate"),
    }
    value = evaluation["holding"][paths[metric][0]][paths[metric][1]]
    return None if value is None else float(value)


def _stats(values: Sequence[float | None]) -> dict[str, Any]:
    finite = [float(value) for value in values if value is not None]
    return {
        "mean": mean(finite) if finite else None,
        "std": pstdev(finite) if len(finite) > 1 else (0.0 if finite else None),
        "n": len(finite),
        "values": finite,
    }


def _summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    models = [str(item["comparison_id"]) for item in MODEL_TEMPLATES]
    summary: dict[str, Any] = {
        "raw_9_run": {},
        "fold_means_over_3_seeds": {},
        "paired_event_f1": {},
        "global_shuffle_event_pr_auc": {},
    }
    for model in models:
        model_rows = [item for item in results if item["comparison_id"] == model]
        summary["raw_9_run"][model] = {
            split: {
                metric: _stats(
                    [_metric(row[split]["correct"], metric) for row in model_rows]
                )
                for metric in METRICS
            }
            for split in ("natural", "challenge")
        }
        summary["fold_means_over_3_seeds"][model] = {}
        for fold in sorted({str(item["fold"]) for item in model_rows}):
            fold_rows = [item for item in model_rows if item["fold"] == fold]
            summary["fold_means_over_3_seeds"][model][fold] = {
                split: {
                    metric: _stats(
                        [_metric(row[split]["correct"], metric) for row in fold_rows]
                    )
                    for metric in METRICS
                }
                for split in ("natural", "challenge")
            }

    for control in ("B1-v2", "S-0-G1", "G1-train-shuffled"):
        key = f"G1-correct_minus_{control}"
        summary["paired_event_f1"][key] = {}
        for split in ("natural", "challenge"):
            differences = []
            for g1 in results:
                if g1["comparison_id"] != "G1-correct":
                    continue
                control_row = next(
                    (
                        row
                        for row in results
                        if row["comparison_id"] == control
                        and row["fold"] == g1["fold"]
                        and row["seed"] == g1["seed"]
                    ),
                    None,
                )
                if control_row is not None:
                    differences.append(
                        _metric(g1[split]["correct"], "event_f1")
                        - _metric(control_row[split]["correct"], "event_f1")
                    )
            summary["paired_event_f1"][key][split] = {
                **_stats(differences),
                "positive_count": sum(value > 0 for value in differences),
            }

    for model in models:
        summary["global_shuffle_event_pr_auc"][model] = {}
        for split in ("natural", "challenge"):
            differences = []
            for row in results:
                if row["comparison_id"] != model:
                    continue
                correct = _metric(row[split]["correct"], "event_pr_auc")
                shuffled = _metric(
                    row[split]["global_shuffled_action"], "event_pr_auc"
                )
                if correct is not None and shuffled is not None:
                    differences.append(correct - shuffled)
            summary["global_shuffle_event_pr_auc"][model][split] = {
                **_stats(differences),
                "positive_count": sum(value > 0 for value in differences),
            }
    return summary


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def run_gate(
    manifest_path: Path,
    output_path: Path,
    checkpoint_dir: Path,
    folds: Sequence[str] = ("test_task0", "test_task1", "test_task2"),
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
    prior_results: list[dict[str, Any]] = []
    if resume and output_path.exists():
        prior = json.loads(output_path.read_text(encoding="utf-8"))
        if prior.get("protocol") != "phase3B-R1-reduced-crossfold-gate-v1":
            raise ValueError("resume output uses a different protocol")
        if prior.get("folds") != list(folds) or prior.get("seeds") != list(seeds):
            raise ValueError("resume output folds/seeds do not match")
        prior_results = list(prior.get("results", []))
    results = prior_results
    completed = {
        (str(row["fold"]), int(row["seed"]), str(row["comparison_id"]))
        for row in results
    }
    expected_runs = len(folds) * len(seeds) * len(MODEL_TEMPLATES)

    def payload(status: str) -> dict[str, Any]:
        return {
            "protocol": "phase3B-R1-reduced-crossfold-gate-v1",
            "status": status,
            "folds": list(folds),
            "seeds": list(seeds),
            "manifest": str(manifest_path),
            "relations": relations,
            "models": list(MODEL_TEMPLATES),
            "evaluation_modes": list(EVAL_MODES),
            "expected_runs": expected_runs,
            "completed_runs": len(results),
            "checkpoint_dir": str(checkpoint_dir),
            "results": results,
            "summary": _summary(results),
        }

    _write_json(output_path, payload("running"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold_name in folds:
        selected = followup._selected_samples(manifest, fold_name)
        records = followup._topology_records(
            selected, "sparse", relations, smoke, probe
        )
        shape = followup._shape(records, len(relations), probe)
        if (shape.action_steps, shape.action_step_dim) != (6, 7):
            raise ValueError(f"{fold_name}: expected frozen 6x7 action window")
        normalization = probe._normalization(records["train"], "channel_v2")
        target_count = probe._parameter_count(
            probe.RelationalDynamicsProbe(
                "g1_sparse_holder_object_gnn", shape, hidden_dim=48
            )
        )
        hidden_dims = probe._select_hidden_dims(
            [template["model_id"] for template in MODEL_TEMPLATES],
            shape,
            {
                "parameter_match": True,
                "target_parameter_count": target_count,
                "candidate_hidden_dims": list(range(40, 81)),
            },
        )
        specs = []
        for template in MODEL_TEMPLATES:
            spec = dict(template)
            spec["hidden_dim"] = hidden_dims[spec["model_id"]]
            model_probe = probe.RelationalDynamicsProbe(
                spec["model_id"], shape, hidden_dim=spec["hidden_dim"]
            )
            spec["parameter_count"] = probe._parameter_count(model_probe)
            spec["target_parameter_count"] = target_count
            spec["parameter_difference_from_g1"] = (
                spec["parameter_count"] - target_count
            )
            specs.append(spec)
        for seed in seeds:
            for spec in specs:
                run_key = (fold_name, int(seed), str(spec["comparison_id"]))
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
                evaluations = {}
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
                        for mode in EVAL_MODES
                    }
                action_invariance = None
                if spec["comparison_id"] == "S-0-G1":
                    action_invariance = max(
                        abs(
                            _metric(evaluations[split]["correct"], "event_f1")
                            - _metric(evaluations[split][mode], "event_f1")
                        )
                        for split in ("natural", "challenge")
                        for mode in EVAL_MODES
                        if mode not in {"correct", "shuffled_edge"}
                    )
                    if action_invariance > 1e-12:
                        raise AssertionError(
                            f"S-0-G1 changed under action controls: {action_invariance}"
                        )
                checkpoint_path = checkpoint_dir / (
                    f"{fold_name}_{spec['comparison_id']}_seed{seed}.pt"
                )
                torch.save(
                    {
                        "protocol": "phase3B-R1-reduced-crossfold-gate-v1",
                        "fold": fold_name,
                        "seed": int(seed),
                        "comparison_id": spec["comparison_id"],
                        "model_id": spec["model_id"],
                        "shape": shape.__dict__,
                        "hidden_dim": spec["hidden_dim"],
                        "relations": relations,
                        "training": training,
                        "normalization": {
                            key: np.asarray(value).tolist()
                            for key, value in normalization.items()
                        },
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                    },
                    checkpoint_path,
                )
                result = {
                    **spec,
                    "fold": fold_name,
                    "seed": int(seed),
                    "checkpoint": str(checkpoint_path),
                    "shape": shape.__dict__,
                    "split_counts": {role: len(rows) for role, rows in records.items()},
                    "training": training,
                    "action_invariance_max_abs_event_f1_delta": action_invariance,
                    **evaluations,
                }
                results.append(result)
                completed.add(run_key)
                _write_json(output_path, payload("running"))
                print(
                    json.dumps(
                        {
                            "fold": fold_name,
                            "seed": seed,
                            "comparison_id": spec["comparison_id"],
                            "hidden_dim": spec["hidden_dim"],
                            "params": training["parameter_count"],
                            "natural_event_f1": _metric(
                                evaluations["natural"]["correct"], "event_f1"
                            ),
                            "challenge_event_f1": _metric(
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
    _write_json(output_path, final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--folds", nargs="+", default=["test_task0", "test_task1", "test_task2"]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_gate(
        args.manifest,
        args.output,
        args.checkpoint_dir,
        folds=args.folds,
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
