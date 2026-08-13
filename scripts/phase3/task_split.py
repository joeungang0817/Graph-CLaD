"""Create a task-family-held-out split for the Phase 3 probe."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping


def task_family_key(record: Mapping[str, Any]) -> str:
    suite = record.get("suite")
    task_id = record.get("task_id")
    if suite is None or task_id is None:
        raise ValueError(f"sample is missing suite/task_id: {record.get('episode_id')}")
    return f"{suite}:{task_id}"


def make_task_family_split(
    dataset: Mapping[str, Any],
    validation_families: list[str],
    test_families: list[str],
) -> dict[str, Any]:
    validation = set(validation_families)
    test = set(test_families)
    if validation & test:
        raise ValueError("validation and test task families must be disjoint")
    output = copy.deepcopy(dict(dataset))
    samples = output.get("samples", [])
    assignments: dict[str, str] = {}
    family_counts: dict[str, int] = {}
    for sample in samples:
        family = task_family_key(sample)
        if family in test:
            split = "test"
        elif family in validation:
            split = "validation"
        else:
            split = "train"
        sample["split"] = split
        episode_id = str(sample["episode_id"])
        previous = assignments.setdefault(episode_id, split)
        if previous != split:
            raise ValueError(f"episode has inconsistent split assignment: {episode_id}")
        family_counts[family] = family_counts.get(family, 0) + 1
    output["split"] = {
        "unit": "task_family_then_episode",
        "assignments": assignments,
        "validation_task_families": sorted(validation),
        "test_task_families": sorted(test),
        "train_task_families": sorted(set(family_counts) - validation - test),
        "normalization_fit_on": "train_only",
        "threshold_fit_on": "train_only",
    }
    output["task_family_split_audit"] = {
        "family_sample_counts": dict(sorted(family_counts.items())),
        "family_count": len(family_counts),
        "validation_sample_count": sum(
            count for family, count in family_counts.items() if family in validation
        ),
        "test_sample_count": sum(
            count for family, count in family_counts.items() if family in test
        ),
    }
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-families", nargs="+", required=True)
    parser.add_argument("--test-families", nargs="+", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dataset = json.loads(args.input.read_text(encoding="utf-8"))
    output = make_task_family_split(
        dataset,
        validation_families=args.validation_families,
        test_families=args.test_families,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["task_family_split_audit"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
