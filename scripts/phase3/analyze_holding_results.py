"""Summarize holding and control metrics from a controlled Phase 3 report."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def _metric(values: Iterable[Any]) -> dict[str, float | int | None]:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return {
        "mean": statistics.fmean(finite) if finite else None,
        "std": statistics.stdev(finite) if len(finite) > 1 else 0.0 if finite else None,
        "n": len(finite),
    }


def _relation_metric(
    evaluation: Mapping[str, Any],
    relation: str,
    changed: bool,
) -> Mapping[str, Any]:
    section = evaluation["changed_relation" if changed else "future_relation"]
    return section.get("per_relation", {}).get(relation, {})


def analyze_report(
    report: Mapping[str, Any],
    *,
    source_report: str | None = None,
    relation: str = "holding",
    sampling_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a traceable aggregate without treating target labels as inputs."""

    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_run: list[dict[str, Any]] = []
    relation_present = False
    inside_present = False
    for fold_report in report.get("full_reports", []):
        fold_name = str(fold_report.get("config", {}).get("fold", {}).get("name", "unknown"))
        relations = [str(value) for value in fold_report.get("relations", [])]
        relation_present = relation_present or relation in relations
        inside_present = inside_present or "inside" in relations
        for model_result in fold_report.get("models", []):
            evaluations = model_result["evaluations"]
            correct_eval = evaluations["correct"]
            correct = correct_eval["changed_relation"]["macro_f1"]
            no_action = evaluations["no_action"]["changed_relation"]["macro_f1"]
            shuffled_action = evaluations["shuffled_action"]["changed_relation"]["macro_f1"]
            shuffled_edge = evaluations["shuffled_edge"]["changed_relation"]["macro_f1"]
            relation_future = _relation_metric(correct_eval, relation, False)
            relation_changed = _relation_metric(correct_eval, relation, True)
            holding = correct_eval.get("holding") or {}
            holding_change = holding.get("change_event") or {}
            holding_onset = holding.get("onset") or {}
            holding_release = holding.get("release") or {}
            holding_hard_negative = holding.get("hard_negative") or {}
            holding_future = holding.get("future_state") or {}
            no_action_holding = evaluations["no_action"].get("holding") or {}
            shuffled_action_holding = evaluations["shuffled_action"].get("holding") or {}
            shuffled_edge_holding = evaluations["shuffled_edge"].get("holding") or {}
            holding_change_f1 = holding_change.get("f1")
            row = {
                "fold": fold_name,
                "model_id": model_result["model_id"],
                "seed": model_result["seed"],
                "correct_changed": correct,
                f"{relation}_future_f1": relation_future.get("f1"),
                f"{relation}_future_support": relation_future.get("support", 0),
                f"{relation}_changed_f1": relation_changed.get("f1"),
                f"{relation}_changed_support": relation_changed.get("support", 0),
                "holding_change_event_f1": holding_change_f1,
                "holding_change_event_pr_auc": holding_change.get("pr_auc"),
                "holding_onset_f1": holding_onset.get("f1"),
                "holding_release_f1": holding_release.get("f1"),
                "holding_future_pr_auc": holding_future.get("pr_auc"),
                "holding_hard_negative_fpr": holding_hard_negative.get("false_positive_rate"),
                "action_delta_correct_minus_no_action": correct - no_action,
                "action_delta_correct_minus_shuffled_action": correct - shuffled_action,
                "edge_delta_correct_minus_shuffled_edge": correct - shuffled_edge,
                "holding_change_delta_correct_minus_no_action": (
                    holding_change_f1
                    - ((no_action_holding.get("change_event") or {}).get("f1") or 0.0)
                    if holding_change_f1 is not None and no_action_holding else None
                ),
                "holding_change_delta_correct_minus_shuffled_action": (
                    holding_change_f1
                    - ((shuffled_action_holding.get("change_event") or {}).get("f1") or 0.0)
                    if holding_change_f1 is not None and shuffled_action_holding else None
                ),
                "holding_change_delta_correct_minus_shuffled_edge": (
                    holding_change_f1
                    - ((shuffled_edge_holding.get("change_event") or {}).get("f1") or 0.0)
                    if holding_change_f1 is not None and shuffled_edge_holding else None
                ),
            }
            rows[str(model_result["model_id"])].append(row)
            per_run.append(row)

    if not rows:
        raise ValueError("report contains no model runs")
    if not relation_present:
        raise ValueError(f"relation is absent from report: {relation}")

    model_summary: dict[str, Any] = {}
    aggregate_keys = (
        "correct_changed",
        f"{relation}_future_f1",
        f"{relation}_changed_f1",
        "holding_change_event_f1",
        "holding_change_event_pr_auc",
        "holding_onset_f1",
        "holding_release_f1",
        "holding_future_pr_auc",
        "holding_hard_negative_fpr",
        "action_delta_correct_minus_no_action",
        "action_delta_correct_minus_shuffled_action",
        "edge_delta_correct_minus_shuffled_edge",
        "holding_change_delta_correct_minus_no_action",
        "holding_change_delta_correct_minus_shuffled_action",
        "holding_change_delta_correct_minus_shuffled_edge",
    )
    for model_id, model_rows in sorted(rows.items()):
        model_summary[model_id] = {
            "runs": len(model_rows),
            **{
                key: _metric(row[key] for row in model_rows)
                for key in aggregate_keys
            },
        }

    has_actual_holding_change = any(
        summary["holding_change_event_f1"]["n"] > 0
        for summary in model_summary.values()
    )
    ranking_metric = (
        "holding_change_event_f1" if has_actual_holding_change else "correct_changed"
    )
    ranking = sorted(
        [
            {
                "model_id": model_id,
                "mean": summary[ranking_metric]["mean"],
                "std": summary[ranking_metric]["std"],
            }
            for model_id, summary in model_summary.items()
        ],
        key=lambda item: (
            float(item["mean"]) if item["mean"] is not None else float("-inf")
        ),
        reverse=True,
    )
    return {
        "analysis_version": "phase3-holding-report-analysis.v2",
        "source_report": source_report,
        "protocol": report.get("protocol"),
        "run_count": len(per_run),
        "fold_count": len(report.get("full_reports", [])),
        "relation_contract": {
            "primary_relation": relation,
            "inside_present": inside_present,
        },
        "sampling": dict(report.get("config", {}).get("sampling") or {}),
        "sampling_preflight": dict(sampling_preflight or {}),
        "selected_category_counts": report.get("selected_category_counts", {}),
        "models": model_summary,
        "primary_ranking_metric": ranking_metric,
        "primary_ranking": ranking,
        "changed_relation_ranking": ranking if not has_actual_holding_change else None,
        "per_run": per_run,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--relation", default="holding")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    preflight = (
        json.loads(args.preflight.read_text(encoding="utf-8"))
        if args.preflight
        else None
    )
    analysis = analyze_report(
        report,
        source_report=str(args.report),
        relation=args.relation,
        sampling_preflight=preflight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "saved", "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
