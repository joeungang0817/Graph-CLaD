"""Run the corrected Phase 3 holding architecture gate.

This runner is deliberately versioned separately from
``run_reduced_crossfold_gate.py``.  It uses natural validation for checkpoint
selection and threshold fitting, a common action-free current-relation head,
threshold-free holding event PR-AUC as the checkpoint criterion, and durable
per-pair prediction artifacts for calibration and hierarchical analysis.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase3 import run_topology_action_followup as followup


METRIC_PATHS = {
    "event_pr_auc": (
        "holding",
        "conditional_oracle_current_change_event",
        "pr_auc",
    ),
    "event_f1": (
        "holding",
        "conditional_oracle_current_change_event",
        "f1",
    ),
    "end_to_end_event_pr_auc": (
        "holding",
        "end_to_end_change_event",
        "pr_auc",
    ),
    "end_to_end_event_f1": (
        "holding",
        "end_to_end_change_event",
        "f1",
    ),
    "onset_f1": ("holding", "onset", "f1"),
    "release_f1": ("holding", "release", "f1"),
    "hard_negative_fpr": (
        "holding",
        "hard_negative",
        "false_positive_rate",
    ),
    "event_brier": (
        "holding",
        "conditional_oracle_current_change_event",
        "brier_score",
    ),
    "event_ece": (
        "holding",
        "conditional_oracle_current_change_event",
        "ece",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _expand_config_environment(value: Any) -> Any:
    """Expand ${VAR} references in a config without changing non-string values."""

    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_config_environment(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _expand_config_environment(item)
            for key, item in value.items()
        }
    return value


def _validate_persistent_output_root(
    output_root: Path,
    runtime_config: Mapping[str, Any],
) -> None:
    """Reject output paths outside an explicitly declared persistent root.

    Colab's historical ``require_persistent_drive_output`` check remains
    unchanged for legacy configs.  New Linux/SSH configs can instead declare
    ``require_persistent_output`` and one or more
    ``persistent_output_roots`` after environment expansion.
    """

    if not runtime_config.get("require_persistent_output", False):
        return
    configured_roots = runtime_config.get("persistent_output_roots", [])
    if not isinstance(configured_roots, Sequence) or isinstance(
        configured_roots, (str, bytes)
    ):
        raise ValueError("persistent_output_roots must be a list of paths")
    if not configured_roots:
        raise ValueError(
            "require_persistent_output is enabled but no persistent_output_roots were configured"
        )
    resolved_output = output_root.expanduser().resolve()
    roots = [Path(str(value)).expanduser().resolve() for value in configured_roots]
    if not any(
        resolved_output == root or root in resolved_output.parents for root in roots
    ):
        raise ValueError(
            "output root is outside declared persistent storage roots: "
            f"{resolved_output} not under {[str(root) for root in roots]}"
        )


def _metric(evaluation: Mapping[str, Any], name: str) -> float | None:
    value: Any = evaluation
    for key in METRIC_PATHS[name]:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
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
    models = sorted({str(row["comparison_id"]) for row in results})
    output: dict[str, Any] = {
        "primary_metric": "natural_test holding conditional/oracle-current event PR-AUC",
        "models": {},
        "paired_differences": {},
        "interpretation_contract": {
            "natural_test": "primary held-out evaluation",
            "challenge_stress": (
                "future-event-selected subset of natural held-out episodes; "
                "stress analysis only and not an independent test"
            ),
        },
    }
    for model in models:
        model_rows = [row for row in results if row["comparison_id"] == model]
        output["models"][model] = {
            view: {
                metric: _stats(
                    [_metric(row[view]["correct"], metric) for row in model_rows]
                )
                for metric in METRIC_PATHS
            }
            for view in ("natural_test", "challenge_stress")
        }

    for control in ("B1-v2", "S-0-G1", "G1-train-shuffled"):
        comparison = f"G1-late-action_minus_{control}"
        output["paired_differences"][comparison] = {}
        for view in ("natural_test", "challenge_stress"):
            output["paired_differences"][comparison][view] = {}
            for metric in ("event_pr_auc", "event_f1", "release_f1", "hard_negative_fpr"):
                differences: list[float] = []
                for g1 in results:
                    if g1["comparison_id"] != "G1-late-action":
                        continue
                    matched = next(
                        (
                            row
                            for row in results
                            if row["comparison_id"] == control
                            and row["fold"] == g1["fold"]
                            and row["seed"] == g1["seed"]
                        ),
                        None,
                    )
                    if matched is None:
                        continue
                    left = _metric(g1[view]["correct"], metric)
                    right = _metric(matched[view]["correct"], metric)
                    if left is not None and right is not None:
                        differences.append(left - right)
                output["paired_differences"][comparison][view][metric] = {
                    **_stats(differences),
                    "positive_count": sum(value > 0 for value in differences),
                }
    return output


def _pair_local_factorial_summary(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report predeclared within-fold contrasts for the H0--H3 factorial."""

    contrasts = {
        "history_without_action_H1_minus_H0": ("H1-history", "H0-state"),
        "action_without_history_H2_minus_H0": ("H2-action", "H0-state"),
        "action_with_history_H3_minus_H1": ("H3-history-action", "H1-history"),
        "history_with_action_H3_minus_H2": ("H3-history-action", "H2-action"),
        "joint_H3_minus_H0": ("H3-history-action", "H0-state"),
    }
    output: dict[str, Any] = {}
    for contrast, (left_id, right_id) in contrasts.items():
        output[contrast] = {}
        for view in ("natural_test", "challenge_stress"):
            output[contrast][view] = {}
            for metric in (
                "event_pr_auc",
                "event_f1",
                "release_f1",
                "hard_negative_fpr",
                "event_brier",
                "event_ece",
            ):
                differences: list[float] = []
                for left in results:
                    if left["comparison_id"] != left_id:
                        continue
                    right = next(
                        (
                            row
                            for row in results
                            if row["comparison_id"] == right_id
                            and row["fold"] == left["fold"]
                            and row["seed"] == left["seed"]
                        ),
                        None,
                    )
                    if right is None:
                        continue
                    left_value = _metric(left[view]["correct"], metric)
                    right_value = _metric(right[view]["correct"], metric)
                    if left_value is not None and right_value is not None:
                        differences.append(left_value - right_value)
                output[contrast][view][metric] = {
                    **_stats(differences),
                    "positive_count": sum(value > 0 for value in differences),
                }
    return output


def _write_predictions(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    run_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for row in rows:
            payload = {**run_metadata, **dict(row)}
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)
    return {
        "path": str(path),
        "rows": len(rows),
        "sha256": _sha256(path),
        "format": "gzip_jsonl",
        "unit": "valid holding prediction pair",
    }


def _extract_prediction_artifact(
    evaluation: dict[str, Any],
    path: Path,
    run_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    rows = evaluation.pop("prediction_rows", [])
    return _write_predictions(path, rows, run_metadata)


def _snapshot_sources(
    snapshot_dir: Path,
    config_path: Path,
    manifest_path: Path,
    config: Mapping[str, Any],
) -> dict[str, str]:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        Path(__file__),
        PROJECT_ROOT / "scripts" / "phase3" / "offline_probe.py",
        PROJECT_ROOT / "scripts" / "phase3" / "build_eval_manifest.py",
        PROJECT_ROOT / "scripts" / "phase3" / "run_topology_action_followup.py",
        PROJECT_ROOT / "scripts" / "phase3" / "run_holder_action_smoke.py",
        config_path,
        manifest_path,
    ]
    if config.get("preprocessing", {}).get("input_contract") == "pair_local_causal_history_v1":
        sources.append(
            PROJECT_ROOT / "scripts" / "phase3" / "pair_local_temporal.py"
        )
    hashes: dict[str, str] = {}
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(source)
        destination = snapshot_dir / source.name
        shutil.copy2(source, destination)
        hashes[destination.name] = _sha256(destination)
    return hashes


def _validate_manifest(manifest: Mapping[str, Any], folds: Sequence[str]) -> None:
    if manifest.get("status") != "pass":
        raise ValueError(f"manifest status is not pass: {manifest.get('status')}")
    validation = manifest.get("validation_protocol", {})
    if validation.get("source") != "natural" or validation.get("event_enrichment"):
        raise ValueError("corrected gate requires non-enriched natural validation")
    by_name = {str(fold["name"]): fold for fold in manifest.get("folds", [])}
    for fold_name in folds:
        fold = by_name.get(fold_name)
        if fold is None:
            raise ValueError(f"manifest is missing fold {fold_name}")
        if fold.get("role_sources", {}).get("validation") != "natural":
            raise ValueError(f"{fold_name}: validation source is not natural")
        overlap = fold.get("overlap_payload_hash_qa", {})
        if not overlap.get("challenge_subset_of_natural"):
            raise ValueError(f"{fold_name}: stress view is not a natural-test subset")
        if int(overlap.get("payload_hash_mismatch_count", -1)) != 0:
            raise ValueError(f"{fold_name}: overlap payload hash QA failed")


def run_gate(
    config_path: Path,
    manifest_path: Path | None = None,
    output_root: Path | None = None,
    *,
    allow_cpu: bool = False,
) -> dict[str, Any]:
    import torch

    config = _expand_config_environment(
        json.loads(config_path.read_text(encoding="utf-8"))
    )
    supported_protocols = {
        "phase3B-R1-corrected-smoke-v2",
        "phase3B-R1-corrected-threefold-seed0-v2",
        "phase3-pair-local-temporal-smoke-v1",
        "phase3-pair-local-temporal-threefold-seed0-v1",
        "phase3-pair-local-temporal-action-alignment-seed0-v1",
    }
    if config.get("protocol") not in supported_protocols:
        raise ValueError(f"unexpected config protocol: {config.get('protocol')}")
    manifest_path = manifest_path or Path(str(config["manifest"]))
    artifact_config = dict(config.get("artifacts", {}))
    output_root = output_root or Path(str(artifact_config["output_root"]))
    runtime_config = dict(config.get("runtime", {}))
    if (
        runtime_config.get("require_persistent_drive_output", False)
        and not allow_cpu
        and not str(output_root).startswith("/content/drive/")
    ):
        raise ValueError(f"output root is not persistent Drive storage: {output_root}")
    _validate_persistent_output_root(output_root, runtime_config)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = _expand_config_environment(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    folds = [str(value) for value in config.get("folds", ["test_task1"])]
    seeds = [int(value) for value in config.get("seeds", [0])]
    _validate_manifest(manifest, folds)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if runtime_config.get("require_cuda", False) and device != "cuda" and not allow_cpu:
        raise RuntimeError("corrected smoke requires CUDA; use --allow-cpu only for local QA")

    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_root / str(artifact_config.get("checkpoints", "checkpoints"))
    prediction_dir = output_root / str(artifact_config.get("predictions", "predictions"))
    snapshot_dir = output_root / str(artifact_config.get("code_snapshot", "code_snapshot"))
    result_path = output_root / str(
        artifact_config.get("result", "phase3_corrected_smoke_task1_seed0_v2.json")
    )
    runtime_manifest_path = output_root / str(
        artifact_config.get("runtime_manifest", "runtime_manifest.json")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    snapshot_hashes = _snapshot_sources(
        snapshot_dir, config_path, manifest_path, config
    )
    config_hash = _sha256(config_path)
    manifest_hash = _sha256(manifest_path)

    smoke = followup._load_smoke_module()
    probe = smoke._load_probe()
    relations = [str(value) for value in manifest["relations"]]
    training_config = dict(config.get("training", {}))
    parameter_config = dict(config.get("parameter_matching", {}))
    current_head_contract = str(training_config["current_head_contract"])
    checkpoint_criterion = str(training_config["checkpoint_criterion"])
    model_templates = [dict(value) for value in config["models"]]
    preprocessing_config = dict(config.get("preprocessing", {}))
    pair_local_input = (
        preprocessing_config.get("input_contract")
        == "pair_local_causal_history_v1"
    )
    history_qa: dict[str, Any] | None = None
    history_feature_names: list[str] = []

    results: list[dict[str, Any]] = []
    for fold_name in folds:
        selected = followup._selected_samples(manifest, fold_name)
        history_dim = 0
        if pair_local_input:
            from scripts.phase3 import pair_local_temporal as temporal

            natural_root = Path(str(preprocessing_config["natural_root"]))
            lookback_steps = int(preprocessing_config.get("lookback_steps", 3))
            all_selected = [
                sample for rows in selected.values() for sample in rows
            ]
            graph_index, index_qa = temporal.load_requested_graph_history(
                natural_root,
                all_selected,
                lookback_steps=lookback_steps,
            )
            if index_qa["status"] != "pass":
                raise ValueError(f"{fold_name}: causal history QA failed: {index_qa}")
            attached_by_role: dict[str, list[dict[str, Any]]] = {}
            sample_qas: list[dict[str, Any]] = []
            for role, samples in selected.items():
                attached_by_role[role] = []
                for sample in samples:
                    attached, sample_qa = temporal.attach_causal_pair_history(
                        sample,
                        graph_index,
                        lookback_steps=lookback_steps,
                    )
                    if sample_qa["future_frame_reads"] != 0:
                        raise AssertionError("causal history read a future frame")
                    attached_by_role[role].append(attached)
                    sample_qas.append(sample_qa)
            selected = attached_by_role
            history_dim = temporal.HISTORY_DIM
            history_feature_names = list(temporal.HISTORY_FEATURE_NAMES)
            history_qa = {
                **index_qa,
                "samples": len(sample_qas),
                "sample_future_frame_reads": sum(
                    int(row["future_frame_reads"]) for row in sample_qas
                ),
                "feature_key": temporal.HISTORY_FEATURE_KEY,
                "feature_names": history_feature_names,
                "source_root": str(natural_root),
                "label_fields_read": [],
            }
        records = followup._topology_records(
            selected, "sparse", relations, smoke, probe
        )
        if pair_local_input:
            # Rebuild records with the appended edge-history contract.  The
            # topology function remains responsible only for pruning.
            records = {}
            load_options = {
                "relations": relations,
                "node_feature_contract": "holder_object_v2",
                "edge_feature_contract": "holder_object_causal_history_v1",
            }
            for role, samples in selected.items():
                prepared: list[dict[str, Any]] = []
                for raw in samples:
                    sample = smoke._robot_object_topology(
                        raw, "sparse", prune_non_pair_nodes=True
                    )
                    sample["split"] = (
                        "test"
                        if role in {"natural_test", "challenge_test"}
                        else role
                    )
                    prepared.append(sample)
                records[role] = probe.load_probe_records(
                    {"samples": prepared}, **load_options
                )
        shape = followup._shape(records, len(relations), probe)
        shape.history_dim = history_dim
        if (shape.action_steps, shape.action_step_dim) != (6, 7):
            raise ValueError(f"{fold_name}: expected frozen 6x7 action window")
        normalization = probe._normalization(records["train"], "channel_v2")
        reference_model = str(
            parameter_config.get("reference_model_id", "g1_sparse_holder_object_gnn")
        )
        reference_hidden = int(parameter_config.get("reference_hidden_dim", 48))
        target_count = probe._parameter_count(
            probe.RelationalDynamicsProbe(
                reference_model,
                shape,
                hidden_dim=reference_hidden,
                current_head_contract=current_head_contract,
            )
        )
        model_ids = sorted({str(spec["model_id"]) for spec in model_templates})
        hidden_dims = probe._select_hidden_dims(
            model_ids,
            shape,
            {
                "parameter_match": bool(parameter_config.get("enabled", True)),
                "target_parameter_count": target_count,
                "candidate_hidden_dims": parameter_config.get(
                    "candidate_hidden_dims", list(range(40, 81))
                ),
                "current_head_contract": current_head_contract,
            },
        )
        specs: list[dict[str, Any]] = []
        for template in model_templates:
            spec = dict(template)
            spec["hidden_dim"] = hidden_dims[str(spec["model_id"])]
            candidate = probe.RelationalDynamicsProbe(
                str(spec["model_id"]),
                shape,
                hidden_dim=int(spec["hidden_dim"]),
                current_head_contract=current_head_contract,
            )
            spec["parameter_count"] = probe._parameter_count(candidate)
            spec["target_parameter_count"] = target_count
            spec["parameter_difference_from_g1"] = (
                spec["parameter_count"] - target_count
            )
            maximum_relative_difference = float(
                parameter_config.get("maximum_relative_difference", 1.0)
            )
            relative_difference = abs(
                spec["parameter_difference_from_g1"]
            ) / max(target_count, 1)
            spec["parameter_relative_difference_from_target"] = (
                relative_difference
            )
            if relative_difference > maximum_relative_difference:
                raise ValueError(
                    f"{fold_name}/{spec['comparison_id']}: parameter match "
                    f"difference {relative_difference:.3%} exceeds "
                    f"{maximum_relative_difference:.3%}"
                )
            specs.append(spec)

        for seed in seeds:
            for spec in specs:
                probe._set_seed(seed)
                training_records = records["train"]
                action_shuffle_qa = None
                if spec["training_action_mode"] == "episode_disjoint_matched":
                    training_records, action_shuffle_qa = (
                        probe.attach_training_action_donors(
                            records["train"], relations=relations
                        )
                    )
                train_loader = probe._loader(
                    training_records,
                    shape,
                    normalization,
                    int(training_config.get("batch_size", 64)),
                    True,
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
                    str(spec["model_id"]),
                    shape,
                    training_records,
                    validation_loader,
                    train_loader,
                    device,
                    int(spec["hidden_dim"]),
                    int(training_config.get("epochs", 10)),
                    int(training_config.get("patience", 3)),
                    float(training_config.get("learning_rate", 0.001)),
                    float(training_config.get("current_loss_weight", 0.25)),
                    relations=relations,
                    training_action_mode=str(spec["training_action_mode"]),
                    current_head_contract=current_head_contract,
                    checkpoint_criterion=checkpoint_criterion,
                )
                if training["checkpoint_metric"] != "holding_event_pr_auc":
                    raise AssertionError(
                        f"{fold_name}/{spec['comparison_id']}: wrong checkpoint "
                        f"metric {training['checkpoint_metric']}"
                    )

                run_metadata = {
                    "protocol": str(config["protocol"]),
                    "fold": fold_name,
                    "seed": seed,
                    "comparison_id": str(spec["comparison_id"]),
                    "model_id": str(spec["model_id"]),
                }
                prediction_artifacts: dict[str, Any] = {}
                validation_evaluation = probe.evaluate_model(
                    model,
                    validation_loader,
                    device,
                    mode="correct",
                    relations=relations,
                    holding_threshold=training["holding_threshold"],
                    current_holding_threshold=training[
                        "current_holding_threshold"
                    ],
                    include_predictions=True,
                    prediction_view="natural_validation",
                )
                validation_filename = (
                    f"{fold_name}_{spec['comparison_id']}_seed{seed}_"
                    "natural_validation_correct.jsonl.gz"
                )
                prediction_artifacts["natural_validation.correct"] = (
                    _extract_prediction_artifact(
                        validation_evaluation,
                        prediction_dir / validation_filename,
                        run_metadata,
                    )
                )

                evaluations: dict[str, dict[str, Any]] = {}
                for view, loader in (
                    ("natural_test", natural_loader),
                    ("challenge_stress", challenge_loader),
                ):
                    evaluations[view] = {}
                    for mode in config["evaluation"]["control_modes"]:
                        evaluation = probe.evaluate_model(
                            model,
                            loader,
                            device,
                            mode=str(mode),
                            relations=relations,
                            holding_threshold=training["holding_threshold"],
                            current_holding_threshold=training[
                                "current_holding_threshold"
                            ],
                            include_predictions=True,
                            prediction_view=view,
                        )
                        filename = (
                            f"{fold_name}_{spec['comparison_id']}_seed{seed}_"
                            f"{view}_{mode}.jsonl.gz"
                        )
                        prediction_artifacts[f"{view}.{mode}"] = (
                            _extract_prediction_artifact(
                                evaluation,
                                prediction_dir / filename,
                                run_metadata,
                            )
                        )
                        evaluations[view][str(mode)] = evaluation

                if pair_local_input:
                    uses_action = bool(getattr(model, "uses_action", False))
                    uses_history = bool(getattr(model, "uses_history", False))
                    for view in ("natural_test", "challenge_stress"):
                        correct_pr_auc = _metric(
                            evaluations[view]["correct"], "event_pr_auc"
                        )
                        if not uses_action and "global_shuffled_action" in evaluations[view]:
                            shuffled_action_pr_auc = _metric(
                                evaluations[view]["global_shuffled_action"],
                                "event_pr_auc",
                            )
                            if correct_pr_auc != shuffled_action_pr_auc:
                                raise AssertionError(
                                    f"{spec['comparison_id']} action invariance failed"
                                )
                        for history_mode in (
                            "no_history",
                            "global_shuffled_history",
                        ):
                            if not uses_history and history_mode in evaluations[view]:
                                history_pr_auc = _metric(
                                    evaluations[view][history_mode],
                                    "event_pr_auc",
                                )
                                if correct_pr_auc != history_pr_auc:
                                    raise AssertionError(
                                        f"{spec['comparison_id']} history invariance failed"
                                    )

                if spec["comparison_id"] == "S-0-G1":
                    for view in ("natural_test", "challenge_stress"):
                        correct = _metric(
                            evaluations[view]["correct"], "event_pr_auc"
                        )
                        shuffled = _metric(
                            evaluations[view]["global_shuffled_action"],
                            "event_pr_auc",
                        )
                        if correct != shuffled:
                            raise AssertionError(
                                f"S-0 action invariance failed on {view}: "
                                f"{correct} != {shuffled}"
                            )

                checkpoint_path = checkpoint_dir / (
                    f"{fold_name}_{spec['comparison_id']}_seed{seed}.pt"
                )
                torch.save(
                    {
                        **run_metadata,
                        "config_sha256": config_hash,
                        "manifest_sha256": manifest_hash,
                        "shape": shape.__dict__,
                        "relations": relations,
                        "hidden_dim": int(spec["hidden_dim"]),
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
                    "seed": seed,
                    "shape": shape.__dict__,
                    "split_counts": {
                        role: len(rows) for role, rows in records.items()
                    },
                    "causal_history_qa": history_qa,
                    "validation_source": "natural",
                    "training": training,
                    "training_action_shuffle_qa": action_shuffle_qa,
                    "checkpoint": {
                        "path": str(checkpoint_path),
                        "sha256": _sha256(checkpoint_path),
                    },
                    "prediction_artifacts": prediction_artifacts,
                    "natural_validation": validation_evaluation,
                    **evaluations,
                }
                results.append(result)
                partial = {
                    "protocol": str(config["protocol"]),
                    "status": "running",
                    "config": config,
                    "config_path": str(config_path),
                    "config_sha256": config_hash,
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": manifest_hash,
                    "device": device,
                    "results": results,
                    "summary": {
                        **_summary(results),
                        **(
                            {
                                "pair_local_factorial_contrasts":
                                _pair_local_factorial_summary(results)
                            }
                            if pair_local_input
                            else {}
                        ),
                    },
                }
                _write_json(result_path, partial)
                print(
                    json.dumps(
                        {
                            **run_metadata,
                            "params": training["parameter_count"],
                            "threshold": training["holding_threshold"],
                            "natural_event_pr_auc": _metric(
                                evaluations["natural_test"]["correct"],
                                "event_pr_auc",
                            ),
                            "natural_event_f1": _metric(
                                evaluations["natural_test"]["correct"],
                                "event_f1",
                            ),
                            "natural_release_f1": _metric(
                                evaluations["natural_test"]["correct"],
                                "release_f1",
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    runtime_manifest = {
        "protocol": str(config["protocol"]),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "output_root": str(output_root),
        "code_snapshot": str(snapshot_dir),
        "snapshot_sha256": snapshot_hashes,
        "result_path": str(result_path),
        "checkpoints": str(checkpoint_dir),
        "predictions": str(prediction_dir),
        "causal_history_qa": history_qa,
    }
    _write_json(runtime_manifest_path, runtime_manifest)
    if len(folds) == 1 and len(seeds) == 1:
        scope_limit = (
            "One fold/seed smoke is an architecture/protocol gate, not a "
            "generalization conclusion."
        )
    elif len(seeds) == 1:
        scope_limit = (
            "This multi-fold screen uses one training seed; task folds are the "
            "outer evaluation units and the result is not a generalization "
            "conclusion."
        )
    else:
        scope_limit = (
            "Training seeds within a task fold share held-out episodes and are "
            "not independent test samples."
        )
    final = {
        "protocol": str(config["protocol"]),
        "status": "completed",
        "config": config,
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "device": device,
        "runtime_manifest": str(runtime_manifest_path),
        "results": results,
        "summary": {
            **_summary(results),
            **(
                {
                    "pair_local_factorial_contrasts":
                    _pair_local_factorial_summary(results)
                }
                if pair_local_input
                else {}
            ),
        },
        "claim_limits": [
            "The change_event compatibility field is conditional on oracle current holding.",
            "The challenge_stress view is selected from natural held-out episodes using future-event information and is not an independent generalization test.",
            scope_limit,
            *(
                [
                    "Causal history is reconstructed only from episode frames at or before current t; frame availability and conflicts are stored in causal_history_qa.",
                    "H0--H3 are a pair-local architecture factorial and do not establish the final CLaD representation claim.",
                    "Weak-label audit subsets are internal-consistency sensitivity groups, not human ground truth.",
                ]
                if pair_local_input
                else []
            ),
        ],
    }
    _write_json(result_path, final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU and a non-Drive output only for local protocol QA.",
    )
    args = parser.parse_args()
    result = run_gate(
        args.config,
        manifest_path=args.manifest,
        output_root=args.output_root,
        allow_cpu=args.allow_cpu,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "protocol": result["protocol"],
                "runs": len(result["results"]),
                "runtime_manifest": result["runtime_manifest"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
