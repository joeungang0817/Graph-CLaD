"""Run the reproducible Phase 3 task-family-held-out experiment.

This is the durable replacement for the large exploratory Colab cell.  It
reads the Phase 2D input-clean JSONL files directly from a mounted Drive path,
builds three deterministic task-family folds, and writes a checkpoint after
each completed fold.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_probe():
    """Load the sibling probe both as a package module and as a standalone file."""

    try:
        from . import offline_probe

        return offline_probe
    except ImportError:
        probe_path = Path(__file__).with_name("offline_probe.py")
        spec = importlib.util.spec_from_file_location("phase3_offline_probe", probe_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load probe module from {probe_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


def _load_sampling():
    """Load the sibling sampling module as a package or standalone file."""

    try:
        from . import sampling

        return sampling
    except ImportError:
        sampling_path = Path(__file__).with_name("sampling.py")
        spec = importlib.util.spec_from_file_location("phase3_sampling", sampling_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load sampling module from {sampling_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


def _read_family_samples(dataset_root: Path, task_ids: tuple[int, ...]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    source_counts: dict[str, int] = {}
    for task_id in task_ids:
        path = dataset_root / f"task{task_id}" / f"phase2d_task{task_id}_graph_dataset.jsonl.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        family = f"libero_spatial:{task_id}"
        total = 0
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                for original in payload.get("samples", []):
                    sample = dict(original)
                    sample["suite"] = "libero_spatial"
                    sample["task_id"] = task_id
                    sample["episode_id"] = str(
                        sample.get("episode_id", f"task{task_id}_{payload.get('demo_key', 'unknown')}")
                    )
                    grouped[family][sample["episode_id"]].append(sample)
                    total += 1
        source_counts[family] = total

    return {
        family: [sample for episode in sorted(episodes) for sample in episodes[episode]]
        for family, episodes in grouped.items()
    }, source_counts


def _balanced_cap(samples: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    """Backward-compatible alias for the original episode round-robin cap."""

    return _load_sampling().episode_round_robin_cap(samples, cap)


def _fold_specs() -> list[dict[str, str]]:
    return [
        {"name": "test_task0", "validation": "libero_spatial:1", "test": "libero_spatial:0", "train": "libero_spatial:2"},
        {"name": "test_task1", "validation": "libero_spatial:2", "test": "libero_spatial:1", "train": "libero_spatial:0"},
        {"name": "test_task2", "validation": "libero_spatial:1", "test": "libero_spatial:2", "train": "libero_spatial:0"},
    ]


def default_config() -> dict[str, Any]:
    return {
        "probe_version": "phase3-offline.v1-controlled-taskfamily-inputclean",
        "models": [
            "p0_flat_mlp",
            "p1_node_no_message",
            "p2_gnn_empty_edge",
            "p3_gnn_geometry",
            "p4_gnn_soft_attention",
        ],
        "seeds": [0, 1, 2],
        "parameter_match": True,
        "target_parameter_count": 70000,
        "candidate_hidden_dims": [40, 48, 56, 64, 72, 80, 88, 96],
        "hidden_dim": 64,
        "batch_size": 64,
        "epochs": 40,
        "patience": 8,
        "learning_rate": 0.001,
        "current_loss_weight": 0.25,
        "device": "cuda",
    }


def run_experiment(
    dataset_root: Path,
    output_root: Path,
    max_samples_per_family: int = 600,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    probe = _load_probe()
    sampling_module = _load_sampling()
    base_config = copy.deepcopy(config or default_config())
    dataset_contract = dict(base_config.get("dataset_contract") or {})
    if "target_relations" not in base_config and dataset_contract.get("target_relations"):
        base_config["target_relations"] = list(dataset_contract["target_relations"])
    sampling_config = dict(base_config.get("sampling") or {})
    raw_families, source_counts = _read_family_samples(dataset_root, (0, 1, 2))
    families = sorted(raw_families)
    expected = ["libero_spatial:0", "libero_spatial:1", "libero_spatial:2"]
    if families != expected:
        raise ValueError(f"unexpected task families: {families}")
    family_samples = {
        family: sampling_module.select_with_config(
            raw_families[family],
            max_samples_per_family,
            sampling_config,
        )
        for family in families
    }
    insufficient = {
        family: len(rows)
        for family, rows in family_samples.items()
        if len(rows) < max_samples_per_family
    }
    if insufficient:
        raise ValueError(f"insufficient samples per family: {insufficient}")

    selected_category_counts = {
        family: dict(sampling_module.category_counts(rows))
        for family, rows in family_samples.items()
    }
    folds = list(base_config.get("folds") or _fold_specs())
    output_root.mkdir(parents=True, exist_ok=True)
    fold_summaries: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for fold in folds:
        split_by_family = {
            fold["train"]: "train",
            fold["validation"]: "validation",
            fold["test"]: "test",
        }
        samples: list[dict[str, Any]] = []
        for family in families:
            for original in family_samples[family]:
                sample = dict(original)
                sample["split"] = split_by_family[family]
                samples.append(sample)
        dataset = {
            "dataset_version": str(
                dataset_contract.get(
                    "artifact_version",
                    dataset_contract.get(
                        "source",
                        "phase2d-full-demo.v2-input-clean-controlled-v1",
                    ),
                )
            ),
            "source": str(dataset_contract.get("description", "Phase 2D JSONL")),
            "fold": fold,
            "split": {
                "unit": "task_family",
                "episode_overlap": False,
                "normalization_fit": "train_samples_only",
            },
            "samples": samples,
        }
        fold_config = copy.deepcopy(base_config)
        fold_config["fold"] = fold
        started = time.time()
        report = probe.run_probe(dataset, fold_config)
        reports.append(report)
        rows: list[dict[str, Any]] = []
        for model_result in report["models"]:
            evaluations = model_result["evaluations"]
            correct_holding = evaluations["correct"].get("holding") or {}
            no_action_holding = evaluations["no_action"].get("holding") or {}
            shuffled_action_holding = evaluations["shuffled_action"].get("holding") or {}
            rows.append({
                "model_id": model_result["model_id"],
                "seed": model_result["seed"],
                "hidden_dim": model_result["hidden_dim"],
                "parameter_count": model_result["training"]["parameter_count"],
                "best_val_macro_f1": model_result["training"]["best_val_macro_f1"],
                "checkpoint_metric": model_result["training"].get("checkpoint_metric"),
                "best_checkpoint_score": model_result["training"].get("best_checkpoint_score"),
                "holding_threshold": model_result["training"].get("holding_threshold"),
                "correct_future_macro_f1": evaluations["correct"]["future_relation"]["macro_f1"],
                "correct_changed_macro_f1": evaluations["correct"]["changed_relation"]["macro_f1"],
                "no_action_changed_macro_f1": evaluations["no_action"]["changed_relation"]["macro_f1"],
                "shuffled_action_changed_macro_f1": evaluations["shuffled_action"]["changed_relation"]["macro_f1"],
                "shuffled_edge_changed_macro_f1": evaluations["shuffled_edge"]["changed_relation"]["macro_f1"],
                "correct_per_relation": evaluations["correct"]["future_relation"]["per_relation"],
                "correct_changed_per_relation": evaluations["correct"]["changed_relation"]["per_relation"],
                "correct_holding": correct_holding,
                "holding_change_event_f1": (correct_holding.get("change_event") or {}).get("f1"),
                "holding_onset_f1": (correct_holding.get("onset") or {}).get("f1"),
                "holding_release_f1": (correct_holding.get("release") or {}).get("f1"),
                "holding_hard_negative_fpr": (correct_holding.get("hard_negative") or {}).get("false_positive_rate"),
                "holding_change_delta_correct_minus_no_action": (
                    ((correct_holding.get("change_event") or {}).get("f1") or 0.0)
                    - ((no_action_holding.get("change_event") or {}).get("f1") or 0.0)
                    if correct_holding and no_action_holding else None
                ),
                "holding_change_delta_correct_minus_shuffled_action": (
                    ((correct_holding.get("change_event") or {}).get("f1") or 0.0)
                    - ((shuffled_action_holding.get("change_event") or {}).get("f1") or 0.0)
                    if correct_holding and shuffled_action_holding else None
                ),
            })
        fold_summaries.append({
            "fold": fold,
            "split_counts": report["split_counts"],
            "shape": report["shape"],
            "model_rows": rows,
            "elapsed_sec": round(time.time() - started, 2),
        })
        (output_root / "phase3_controlled_taskfamily_checkpoint.json").write_text(
            json.dumps({
                "status": "running",
                "completed_folds": fold_summaries,
                "completed_fold_count": len(fold_summaries),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    final_report = {
        "status": "completed",
        "protocol": str(
            base_config.get("protocol", "phase3-controlled-taskfamily-inputclean-v1")
        ),
        "dataset_root": str(dataset_root),
        "max_samples_per_family": max_samples_per_family,
        "source_counts": source_counts,
        "selected_counts": {family: len(family_samples[family]) for family in families},
        "selected_category_counts": selected_category_counts,
        "config": base_config,
        "folds": fold_summaries,
        "full_reports": reports,
    }
    (output_root / "phase3_controlled_taskfamily_report.json").write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "phase3_controlled_taskfamily_config.json").write_text(
        json.dumps({
            "protocol": final_report["protocol"],
            "folds": folds,
            "config": base_config,
            "source_counts": source_counts,
            "selected_counts": {family: len(family_samples[family]) for family in families},
            "selected_category_counts": selected_category_counts,
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "status": "completed",
        "fold_count": len(fold_summaries),
        "run_count": sum(len(fold["model_rows"]) for fold in fold_summaries),
        "output": str(output_root / "phase3_controlled_taskfamily_report.json"),
        "folds": [
            {
                "name": fold["fold"]["name"],
                "split_counts": fold["split_counts"],
                "elapsed_sec": fold["elapsed_sec"],
            }
            for fold in fold_summaries
        ],
    }
    (output_root / "phase3_controlled_taskfamily_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return final_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples-per-family", type=int, default=600)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
    report = run_experiment(
        args.dataset_root,
        args.output_root,
        max_samples_per_family=args.max_samples_per_family,
        config=config,
    )
    print(json.dumps({
        "status": report["status"],
        "fold_count": len(report["folds"]),
        "run_count": sum(len(fold["model_rows"]) for fold in report["folds"]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
