"""Create, validate, and repair episode-level Phase 2D split manifests."""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


def build_manifest(
    task_ids: Iterable[int] = (0, 1, 2),
    demos_per_task: int = 50,
    seed: int = 0,
    train_fraction: float = 0.75,
    validation_fraction: float = 0.125,
    generalization_train_tasks: Iterable[int] = (0, 1),
    generalization_test_tasks: Iterable[int] = (2,),
) -> dict[str, Any]:
    """Assign whole demonstrations before label extraction or normalization."""

    task_ids = tuple(int(task_id) for task_id in task_ids)
    entries = [
        {"task_id": task_id, "demo_key": f"demo_{demo_id}", "demo_id": demo_id}
        for task_id in task_ids for demo_id in range(demos_per_task)
    ]
    shuffled = list(range(len(entries)))
    random.Random(seed).shuffle(shuffled)
    train_count = int(len(entries) * train_fraction + 0.5)
    validation_count = int(len(entries) * validation_fraction + 0.5)
    if train_count + validation_count >= len(entries):
        raise ValueError("split fractions leave no test demonstrations")
    in_task_by_index: dict[int, str] = {}
    for rank, index in enumerate(shuffled):
        if rank < train_count:
            split = "train"
        elif rank < train_count + validation_count:
            split = "validation"
        else:
            split = "test"
        in_task_by_index[index] = split

    train_tasks = {int(value) for value in generalization_train_tasks}
    test_tasks = {int(value) for value in generalization_test_tasks}
    if train_tasks & test_tasks:
        raise ValueError("a task cannot be in both generalization train and test")
    for index, entry in enumerate(entries):
        task_id = int(entry["task_id"])
        if task_id in train_tasks:
            generalization_split = "train"
        elif task_id in test_tasks:
            generalization_split = "test"
        else:
            generalization_split = "validation"
        entry.update(
            in_task_split=in_task_by_index[index],
            task_generalization_split=generalization_split,
        )

    counts = Counter(entry["in_task_split"] for entry in entries)
    generalization_counts = Counter(entry["task_generalization_split"] for entry in entries)
    return {
        "manifest_version": "phase2d-demo-split.v1",
        "seed": seed,
        "unit": "demonstration",
        "created_before_labels": True,
        "episodes": entries,
        "counts": dict(counts),
        "task_generalization_counts": dict(generalization_counts),
        "rules": {
            "adjacent_frames_share_split": True,
            "thresholds_fit_on_train_only": True,
            "normalization_fit_on_train_only": True,
        },
    }


def split_lookup(manifest: Mapping[str, Any]) -> dict[tuple[int, str], Mapping[str, Any]]:
    entries = manifest.get("episodes", [])
    result: dict[tuple[int, str], Mapping[str, Any]] = {}
    for item in entries if isinstance(entries, list) else []:
        if not isinstance(item, Mapping):
            continue
        task_id = int(item["task_id"])
        demo_key = str(item.get("demo_key", f"demo_{int(item['demo_id'])}"))
        key = (task_id, demo_key)
        if key in result:
            raise ValueError(f"duplicate split manifest key: {key}")
        result[key] = item
    return result


def validate_manifest(manifest: Mapping[str, Any], expected_count: int | None = None) -> dict[str, Any]:
    lookup = split_lookup(manifest)
    invalid = []
    for key, item in lookup.items():
        if item.get("in_task_split") not in {"train", "validation", "test"}:
            invalid.append({"key": key, "field": "in_task_split"})
        if item.get("task_generalization_split") not in {"train", "validation", "test"}:
            invalid.append({"key": key, "field": "task_generalization_split"})
    count_ok = expected_count is None or len(lookup) == expected_count
    return {
        "status": "pass" if not invalid and count_ok else "fail",
        "episode_count": len(lookup),
        "expected_count": expected_count,
        "invalid_assignments": invalid,
        "in_task_counts": dict(Counter(item["in_task_split"] for item in lookup.values())),
        "task_generalization_counts": dict(
            Counter(item["task_generalization_split"] for item in lookup.values())
        ),
    }


def repair_null_splits(
    source: Path, output: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Repair legacy null splits using a strict, memory-efficient line rewrite."""

    lookup = split_lookup(manifest)
    null_token = '"split":{"in_task":null,"task_generalization":null}'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    records = replaced = 0
    with gzip.open(source, "rt", encoding="utf-8") as input_file, gzip.open(
        temporary, "wt", encoding="utf-8", compresslevel=1
    ) as output_file:
        for line in input_file:
            records += 1
            try:
                task_id = int(line.split('"task_id":', 1)[1].split(",", 1)[0])
                demo_id = int(line.split('"demo_id":', 1)[1].split(",", 1)[0])
            except (IndexError, ValueError) as error:
                raise RuntimeError(f"record {records}: unable to parse task/demo key") from error
            item = lookup.get((task_id, f"demo_{demo_id}"))
            if item is None:
                raise KeyError(f"split assignment missing for task {task_id}, demo_{demo_id}")
            count = line.count(null_token)
            if count != 1:
                raise RuntimeError(f"record {records}: expected one null split token, found {count}")
            split = json.dumps(
                {
                    "in_task": item["in_task_split"],
                    "task_generalization": item["task_generalization_split"],
                },
                separators=(",", ":"),
            )
            output_file.write(line.replace(null_token, '"split":' + split, 1))
            replaced += 1
    temporary.replace(output)
    return {"status": "pass", "records": records, "replaced": replaced, "output": str(output)}


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-ids", default="0,1,2")
    parser.add_argument("--demos-per-task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generalization-train", default="0,1")
    parser.add_argument("--generalization-test", default="2")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = build_manifest(
        task_ids=_parse_ints(args.task_ids),
        demos_per_task=args.demos_per_task,
        seed=args.seed,
        generalization_train_tasks=_parse_ints(args.generalization_train),
        generalization_test_tasks=_parse_ints(args.generalization_test),
    )
    report = validate_manifest(manifest, len(_parse_ints(args.task_ids)) * args.demos_per_task)
    if report["status"] != "pass":
        raise RuntimeError(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

