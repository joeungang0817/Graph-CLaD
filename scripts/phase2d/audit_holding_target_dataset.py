"""Audit target-aligned holding coverage under the actual Phase 3 samplers."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.phase3.sampling import category_counts, select_with_config


def _read_task_samples(dataset_root: Path, task_id: int) -> list[dict[str, Any]]:
    path = (
        dataset_root
        / f"task{task_id}"
        / f"phase2d_task{task_id}_graph_dataset.jsonl.gz"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    samples: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for original in record.get("samples", []):
                sample = dict(original)
                sample["episode_id"] = str(
                    sample.get(
                        "episode_id",
                        f"task{task_id}_{record.get('demo_key', 'unknown')}",
                    )
                )
                samples.append(sample)
    return samples


def audit_dataset(
    dataset_root: Path,
    *,
    cap: int = 600,
    category_quota: int = 120,
) -> dict[str, Any]:
    manifest_path = dataset_root / "phase2d_holding_target_dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass":
        raise RuntimeError(
            f"target dataset manifest is not pass: {manifest.get('warnings')}"
        )

    methods = {
        "episode_round_robin": {"method": "episode_round_robin"},
        "category_aware_v1": {
            "method": "category_aware_v1",
            "category_quota": category_quota,
        },
        "category_aware_episode_round_robin_v2": {
            "method": "category_aware_episode_round_robin_v2",
            "category_quota": category_quota,
        },
    }
    task_reports: dict[str, Any] = {}
    warnings: list[str] = []
    required = (
        "future_holding_positive",
        "holding_changed",
        "hard_negative",
        "background",
    )
    for task_id in (0, 1, 2):
        samples = _read_task_samples(dataset_root, task_id)
        method_reports: dict[str, Any] = {}
        for method_name, sampling in methods.items():
            selected = select_with_config(samples, cap, sampling)
            counts: Counter[str] = category_counts(selected)
            method_reports[method_name] = {
                "selected_samples": len(selected),
                "episode_count": len({sample["episode_id"] for sample in selected}),
                "category_counts": dict(counts),
                "multi_label_count_sum": sum(counts.values()),
            }
            if method_name != "episode_round_robin":
                for category in required:
                    if counts[category] < category_quota:
                        warnings.append(
                            f"task{task_id}:{method_name}:{category}={counts[category]}"
                        )
        task_reports[str(task_id)] = {
            "available_samples": len(samples),
            "available_episode_count": len(
                {str(sample["episode_id"]) for sample in samples}
            ),
            "samplers": method_reports,
        }

    return {
        "audit_version": "phase2d-holding-target-sampler-audit.v1",
        "dataset_root": str(dataset_root),
        "manifest": str(manifest_path),
        "cap": cap,
        "category_quota": category_quota,
        "category_counts_are_multilabel": True,
        "tasks": task_reports,
        "warnings": warnings,
        "status": "pass" if not warnings else "fail",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cap", type=int, default=600)
    parser.add_argument("--category-quota", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = audit_dataset(
        args.dataset_root,
        cap=args.cap,
        category_quota=args.category_quota,
    )
    output = args.output or (
        args.dataset_root / "phase2d_holding_target_sampler_audit.json"
    )
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
