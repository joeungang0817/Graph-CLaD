"""Review all holding event clusters from masked continuous trajectory evidence.

This audit is deliberately separate from the dataset labels.  It never reads
the exported holding evidence flags when forming a decision and never rewrites
train/validation/test targets.  Its outputs are QA triage and sensitivity
groups, not human ground truth.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


DECISIONS = ("likely_pass", "likely_label_error", "ambiguous")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vector(value: Any, expected: int = 3) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) < expected:
        return None
    try:
        result = [float(value[index]) for index in range(expected)]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _subtract(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [float(a) - float(b) for a, b in zip(left, right)]


def _segment_metrics(
    points: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    if not points:
        return {"valid": False, "reason": "empty_segment", "hold_score": 0, "strong_hold": False}
    source = [_vector(point.get("source_position")) for point in points]
    target = [_vector(point.get("target_position")) for point in points]
    if any(value is None for value in source + target):
        return {"valid": False, "reason": "missing_position", "hold_score": 0, "strong_hold": False}
    source_v = [value for value in source if value is not None]
    target_v = [value for value in target if value is not None]
    relative = [_subtract(target_v[index], source_v[index]) for index in range(len(points))]
    relative_span = max(_norm(_subtract(value, relative[0])) for value in relative)
    source_delta = _subtract(source_v[-1], source_v[0])
    target_delta = _subtract(target_v[-1], target_v[0])
    source_displacement = _norm(source_delta)
    target_displacement = _norm(target_delta)
    follow_residual = _norm(_subtract(target_delta, source_delta))

    valid_contacts = [
        bool(point.get("contact"))
        for point in points
        if bool(point.get("contact_valid", point.get("contact") is not None))
    ]
    contact_fraction = (
        sum(valid_contacts) / len(valid_contacts) if valid_contacts else None
    )
    closed_values = []
    qpos_max_abs = []
    for point in points:
        qpos = point.get("gripper_qpos")
        if not isinstance(qpos, Sequence) or isinstance(qpos, (str, bytes)) or not qpos:
            continue
        try:
            magnitude = max(abs(float(value)) for value in qpos)
        except (TypeError, ValueError):
            continue
        qpos_max_abs.append(magnitude)
        closed_values.append(
            magnitude <= float(thresholds["closed_gripper_max_abs_qpos"])
        )
    closed_fraction = sum(closed_values) / len(closed_values) if closed_values else None
    contact_supported = (
        contact_fraction is not None
        and contact_fraction >= float(thresholds["minimum_contact_fraction"])
    )
    closed_supported = (
        closed_fraction is not None
        and closed_fraction >= float(thresholds["minimum_closed_fraction"])
    )
    stable_supported = relative_span <= float(thresholds["stable_relative_span"])
    movement = max(source_displacement, target_displacement)
    following_supported = (
        movement >= float(thresholds["minimum_motion"])
        and follow_residual <= float(thresholds["following_residual"])
    )
    components = {
        "contact": contact_supported,
        "closed_gripper": closed_supported,
        "relative_stability": stable_supported,
        "object_following": following_supported,
    }
    score = sum(bool(value) for value in components.values())
    return {
        "valid": True,
        "frames": len(points),
        "start_step": int(points[0]["step"]),
        "end_step": int(points[-1]["step"]),
        "contact_fraction": contact_fraction,
        "closed_fraction": closed_fraction,
        "qpos_max_abs_mean": sum(qpos_max_abs) / len(qpos_max_abs) if qpos_max_abs else None,
        "relative_span": relative_span,
        "source_displacement": source_displacement,
        "target_displacement": target_displacement,
        "follow_residual": follow_residual,
        "movement": movement,
        "components": components,
        "hold_score": score,
        "strong_hold": score == len(components),
        "static_contact_candidate": (
            contact_supported and closed_supported and stable_supported and not following_supported
        ),
    }


def extract_metrics(
    row: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    trajectory = row.get("trajectory", []) or []
    segment_frames = int(thresholds["segment_frames"])
    if len(trajectory) < segment_frames:
        return {
            "valid": False,
            "reason": "trajectory_too_short",
            "trajectory_frames": len(trajectory),
        }
    early = _segment_metrics(trajectory[:segment_frames], thresholds)
    late = _segment_metrics(trajectory[-segment_frames:], thresholds)
    rolling = [
        _segment_metrics(trajectory[index : index + segment_frames], thresholds)
        for index in range(len(trajectory) - segment_frames + 1)
    ]
    valid_rolling = [value for value in rolling if value.get("valid")]
    contacts = [
        bool(point.get("contact"))
        for point in trajectory
        if bool(point.get("contact_valid", point.get("contact") is not None))
    ]
    return {
        "valid": bool(early.get("valid") and late.get("valid") and valid_rolling),
        "trajectory_frames": len(trajectory),
        "trajectory_qa": row.get("trajectory_qa", {}).get("status"),
        "early": early,
        "late": late,
        "rolling_max_hold_score": max(
            (int(value["hold_score"]) for value in valid_rolling), default=0
        ),
        "rolling_strong_hold_count": sum(
            bool(value["strong_hold"]) for value in valid_rolling
        ),
        "rolling_static_candidate_count": sum(
            bool(value["static_contact_candidate"]) for value in valid_rolling
        ),
        "contact_first": contacts[0] if contacts else None,
        "contact_last": contacts[-1] if contacts else None,
        "contact_transition_count": sum(
            left != right for left, right in zip(contacts, contacts[1:])
        ),
    }


def _metric_reason(prefix: str, metric: Mapping[str, Any]) -> str:
    components = metric.get("components", {}) or {}
    return (
        f"{prefix}:score={metric.get('hold_score')}/4,"
        f"contact={components.get('contact')},closed={components.get('closed_gripper')},"
        f"stable={components.get('relative_stability')},follow={components.get('object_following')},"
        f"span={metric.get('relative_span'):.4f},residual={metric.get('follow_residual'):.4f},"
        f"motion={metric.get('movement'):.4f}"
    )


def review_event(row: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    event = str(row["event_type"])
    if not metrics.get("valid") or metrics.get("trajectory_qa") != "pass":
        return {
            "decision": "ambiguous",
            "confidence": "high",
            "error_type": "insufficient_evidence",
            "reasons": [
                f"trajectory_valid={metrics.get('valid')}",
                f"trajectory_qa={metrics.get('trajectory_qa')}",
            ],
        }
    early = metrics["early"]
    late = metrics["late"]
    reasons = [_metric_reason("early", early), _metric_reason("late", late)]

    if event == "onset":
        if late["strong_hold"] and not early["strong_hold"]:
            return {"decision": "likely_pass", "confidence": "high", "error_type": "", "reasons": reasons}
        if late["strong_hold"] and early["strong_hold"]:
            return {"decision": "ambiguous", "confidence": "medium", "error_type": "temporal_alignment_error", "reasons": reasons + ["strong holding evidence already exists in the early segment"]}
        if int(late["hold_score"]) <= 2:
            return {"decision": "likely_label_error", "confidence": "high" if int(late["hold_score"]) <= 1 else "medium", "error_type": "false_onset", "reasons": reasons + ["late segment lacks sufficient holding evidence"]}
        return {"decision": "ambiguous", "confidence": "medium", "error_type": "insufficient_evidence", "reasons": reasons + ["late segment is a static contact candidate without independent following motion"]}

    if event == "release":
        disengagement = (
            not bool(metrics.get("contact_last"))
            or not bool(late.get("components", {}).get("closed_gripper"))
            or not bool(late.get("components", {}).get("relative_stability"))
            or (
                float(late.get("movement", 0.0)) > 0.0
                and not bool(late.get("components", {}).get("object_following"))
            )
        )
        if early["strong_hold"] and not late["strong_hold"] and disengagement:
            return {"decision": "likely_pass", "confidence": "high", "error_type": "", "reasons": reasons}
        if early["strong_hold"] and late["strong_hold"]:
            return {"decision": "likely_label_error", "confidence": "high", "error_type": "false_release", "reasons": reasons + ["strong holding evidence persists through the late segment"]}
        if int(early["hold_score"]) < 3:
            return {"decision": "ambiguous", "confidence": "medium", "error_type": "insufficient_evidence", "reasons": reasons + ["pre-release holding evidence is weak"]}
        if not late["strong_hold"] and disengagement:
            return {"decision": "likely_pass", "confidence": "medium", "error_type": "", "reasons": reasons + ["release evidence is present but early following motion is limited"]}
        return {"decision": "ambiguous", "confidence": "medium", "error_type": "temporal_alignment_error", "reasons": reasons}

    if event == "hard_negative":
        if int(metrics["rolling_strong_hold_count"]) > 0:
            return {"decision": "likely_label_error", "confidence": "high", "error_type": "hard_negative_is_holding", "reasons": reasons + [f"{metrics['rolling_strong_hold_count']} rolling segments meet all four holding components"]}
        if int(metrics["rolling_max_hold_score"]) <= 2:
            return {"decision": "likely_pass", "confidence": "high", "error_type": "", "reasons": reasons + ["no rolling segment has more than two holding components"]}
        return {"decision": "ambiguous", "confidence": "medium", "error_type": "insufficient_evidence", "reasons": reasons + ["static contact candidate reaches three of four components"]}

    return {
        "decision": "ambiguous",
        "confidence": "high",
        "error_type": "unsupported_event_type",
        "reasons": [f"event_type={event}"],
    }


def _flatten(row: Mapping[str, Any], metrics: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, Any]:
    early = metrics.get("early", {}) or {}
    late = metrics.get("late", {}) or {}
    edge_source = row.get("edge", {}).get("source")
    edge_target = row.get("edge", {}).get("target")
    audit_pair_key = "|".join(
        [str(row.get("sample_id")), str(edge_source), str(edge_target)]
    )
    result = {
        "audit_id": row.get("audit_id"),
        "task_id": row.get("task_id"),
        "event_type": row.get("event_type"),
        "episode_id": row.get("episode_id"),
        "sample_id": row.get("sample_id"),
        "event_cluster_id": row.get("event_cluster_id"),
        "start_step": row.get("start_step"),
        "target_step": row.get("target_step"),
        "edge_source": edge_source,
        "edge_target": edge_target,
        "audit_pair_key": audit_pair_key,
        "decision": review["decision"],
        "confidence": review["confidence"],
        "error_type": review["error_type"],
        "reasons": " | ".join(review["reasons"]),
        "trajectory_frames": metrics.get("trajectory_frames"),
        "early_hold_score": early.get("hold_score"),
        "late_hold_score": late.get("hold_score"),
        "early_relative_span": early.get("relative_span"),
        "late_relative_span": late.get("relative_span"),
        "early_follow_residual": early.get("follow_residual"),
        "late_follow_residual": late.get("follow_residual"),
        "early_movement": early.get("movement"),
        "late_movement": late.get("movement"),
        "early_contact_fraction": early.get("contact_fraction"),
        "late_contact_fraction": late.get("contact_fraction"),
        "early_closed_fraction": early.get("closed_fraction"),
        "late_closed_fraction": late.get("closed_fraction"),
        "rolling_max_hold_score": metrics.get("rolling_max_hold_score"),
        "rolling_strong_hold_count": metrics.get("rolling_strong_hold_count"),
        "contact_first": metrics.get("contact_first"),
        "contact_last": metrics.get("contact_last"),
    }
    return result


def analyze(evidence_path: Path, config_path: Path, output_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    thresholds = config["thresholds"]
    with gzip.open(evidence_path, "rt", encoding="utf-8") as handle:
        evidence = [json.loads(line) for line in handle if line.strip()]
    output_root.mkdir(parents=True, exist_ok=True)
    detailed = []
    flat = []
    for row in evidence:
        metrics = extract_metrics(row, thresholds)
        review = review_event(row, metrics)
        if review["decision"] not in DECISIONS:
            raise ValueError(f"Unexpected review decision: {review['decision']}")
        detailed.append(
            {
                "audit_id": row["audit_id"],
                "task_id": row["task_id"],
                "event_type": row["event_type"],
                "episode_id": row["episode_id"],
                "sample_id": row["sample_id"],
                "event_cluster_id": row["event_cluster_id"],
                "edge_source": row.get("edge", {}).get("source"),
                "edge_target": row.get("edge", {}).get("target"),
                "audit_pair_key": "|".join(
                    [
                        str(row["sample_id"]),
                        str(row.get("edge", {}).get("source")),
                        str(row.get("edge", {}).get("target")),
                    ]
                ),
                "metrics": metrics,
                "review": review,
            }
        )
        flat.append(_flatten(row, metrics, review))

    csv_path = output_root / "ai_assisted_cluster_reviews_v1.csv"
    if flat:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    detailed_path = output_root / "ai_assisted_cluster_reviews_v1.jsonl.gz"
    with gzip.open(detailed_path, "wt", encoding="utf-8") as handle:
        for row in detailed:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    groups = {decision: [] for decision in DECISIONS}
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    confidence_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    for row in detailed:
        decision = row["review"]["decision"]
        groups[decision].append(row["audit_pair_key"])
        counts[f"task{row['task_id']}:{row['event_type']}"][decision] += 1
        confidence_counts[row["review"]["confidence"]] += 1
        if row["review"]["error_type"]:
            error_counts[row["review"]["error_type"]] += 1
    groups_path = output_root / "ai_assisted_sensitivity_groups_v1.json"
    groups_path.write_text(
        json.dumps(
            {
                "protocol": config["protocol"],
                "groups": groups,
                "group_unit": "sample_id|edge_source|edge_target",
                "policy": config["use_policy"],
                "label_policy": config["label_policy"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = {
        "protocol": config["protocol"],
        "rubric_version": config["rubric_version"],
        "status": "completed",
        "clusters": len(detailed),
        "decision_counts": dict(sorted(Counter(row["review"]["decision"] for row in detailed).items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "error_type_counts": dict(sorted(error_counts.items())),
        "task_event_decisions": {
            key: dict(sorted(value.items())) for key, value in sorted(counts.items())
        },
        "evidence": {"path": str(evidence_path), "sha256": _sha256(evidence_path)},
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "artifacts": {
            "reviews_csv": str(csv_path),
            "reviews_jsonl_gz": str(detailed_path),
            "sensitivity_groups": str(groups_path),
        },
        "masked_evidence_fields": config["masked_evidence_fields"],
        "label_policy": config["label_policy"],
        "use_policy": config["use_policy"],
        "claim_limit": config["claim_limit"],
    }
    summary_path = output_root / "ai_assisted_weak_label_audit_summary_v1.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.evidence, args.config, args.output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
