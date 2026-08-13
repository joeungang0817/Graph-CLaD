"""Build a balanced event-centered dataset from official Phase 2D demos.

The source remains the existing state-replayed official-demo artifact.  This
script only selects and materializes event windows; it never uses scripted
probe data and never changes the fixed demo-level split.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.phase2d.event_windows import decorate_record_samples


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_name(record: Mapping[str, Any]) -> str:
    split = record.get("split")
    if isinstance(split, Mapping):
        return str(split.get("in_task", "unknown"))
    return str(split or "unknown")


def _sample_key(sample: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        int(sample.get("start_step", 0)),
        int(sample.get("target_step", 0)),
        str(sample.get("event_category", "")),
    )


def _select_samples(
    samples: Iterable[Mapping[str, Any]],
    caps: Mapping[str, int],
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        episode = str(sample.get("episode_id", "unknown"))
        category = str(sample.get("event_category", "ambiguous"))
        horizon = int(sample.get("tau", sample.get("horizon", 0)))
        grouped[(episode, category, horizon)].append(sample)

    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    available_counts: Counter[str] = Counter()
    for (episode, category, horizon), rows in sorted(grouped.items()):
        available_counts[category] += len(rows)
        if category == "ambiguous":
            continue
        cap = int(caps.get(category, 0))
        if cap <= 0:
            continue
        ordered = sorted(rows, key=_sample_key)
        for row in ordered[:cap]:
            selected.append(dict(row))
            selected_counts[category] += 1
    return selected, selected_counts, available_counts


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(path)
    return count


def build_event_dataset(
    task_files: Mapping[int, Path],
    output_root: Path,
    caps: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    caps = dict(caps or {"positive_event": 12, "hard_negative": 12, "background": 6})
    output_root.mkdir(parents=True, exist_ok=True)
    coverage: Counter[str] = Counter()
    available: Counter[str] = Counter()
    split_category_demos: dict[str, set[str]] = defaultdict(set)
    index_rows: list[dict[str, Any]] = []
    task_reports: dict[str, Any] = {}

    for task_id, source in sorted(task_files.items()):
        output_file = output_root / f"task{task_id}" / f"phase2d_task{task_id}_graph_dataset.jsonl.gz"
        task_available: Counter[str] = Counter()
        task_selected: Counter[str] = Counter()
        records = 0
        samples = 0
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_temporary = output_file.with_suffix(output_file.suffix + ".tmp")
        with gzip.open(source, "rt", encoding="utf-8") as handle, gzip.open(
            output_temporary, "wt", encoding="utf-8", compresslevel=1
        ) as output_handle:
            for line in handle:
                if not line.strip():
                    continue
                original = json.loads(line)
                decorated, _ = decorate_record_samples(original)
                selected, selected_counts, available_counts = _select_samples(
                    decorated.get("samples", []), caps
                )
                task_available.update(available_counts)
                available.update(available_counts)
                task_selected.update(selected_counts)
                if not selected:
                    continue
                split = _split_name(decorated)
                for sample in selected:
                    if _split_name({"split": sample.get("split")}) not in {split, "unknown"}:
                        raise ValueError(f"sample split mismatch in task {task_id}: {sample.get('episode_id')}")
                    category = str(sample["event_category"])
                    coverage[f"task{task_id}:{split}:{category}"] += 1
                    split_category_demos[f"task{task_id}:{split}:{category}"].add(
                        str(decorated.get("demo_key", decorated.get("episode_id", "unknown")))
                    )
                    index_rows.append(
                        {
                            "task_id": task_id,
                            "demo_id": decorated.get("demo_id"),
                            "demo_key": decorated.get("demo_key"),
                            "episode_id": sample.get("episode_id"),
                            "horizon": sample.get("horizon", sample.get("tau")),
                            "start_step": sample.get("start_step"),
                            "target_step": sample.get("target_step"),
                            "split": split,
                            "event_category": category,
                            "event_tags": sample.get("event_tags", []),
                            "event_object_ids": sample.get("event_object_ids", []),
                            "source": "official_hdf5_state_replay",
                        }
                    )
                output_record = dict(decorated)
                output_record["samples"] = selected
                output_record["event_sample_counts"] = dict(selected_counts)
                output_handle.write(
                    json.dumps(
                        output_record,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                records += 1
                samples += len(selected)
        output_temporary.replace(output_file)
        written = records
        task_reports[str(task_id)] = {
            "source": str(source),
            "source_sha256": _sha256(source),
            "output": str(output_file),
            "records": records,
            "samples": samples,
            "available_by_category": dict(task_available),
            "selected_by_category": dict(task_selected),
            "written_records": written,
        }

    config = {
        "artifact_version": "phase2d-holding-events.v1+input-clean",
        "source_kind": "official_hdf5_state_replay",
        "holding_relation": "primary",
        "inside_relation": "deferred_unknown",
        "fixed_demo_split_preserved": True,
        "caps_per_demo_horizon": caps,
        "selection": "deterministic_sorted_by_start_target_category",
    }
    config_text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    warnings: list[str] = []
    if not coverage.get("task0:train:positive_event") and not any(
        key.endswith(":positive_event") for key in coverage
    ):
        warnings.append("no holding-positive event windows were selected")
    if not any(key.endswith(":hard_negative") for key in coverage):
        warnings.append("no contact-without-holding hard-negative windows were selected")
    manifest = {
        **config,
        "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "tasks": task_reports,
        "coverage": dict(sorted(coverage.items())),
        "demo_coverage": {
            key: len(value) for key, value in sorted(split_category_demos.items())
        },
        "available_total_by_category": dict(available),
        "index": str(output_root / "phase2d_holding_event_window_index_v1.jsonl.gz"),
        "warnings": warnings,
        "status": "pass" if not warnings else "pass_with_warnings",
    }
    (output_root / "generation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_jsonl_gz(output_root / "phase2d_holding_event_window_index_v1.jsonl.gz", index_rows)
    (output_root / "phase2d_holding_event_dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _task_mapping(values: Iterable[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        task_id, separator, path = value.partition("=")
        if not separator:
            raise ValueError("--task must use TASK_ID=/path/to/input.jsonl.gz")
        result[int(task_id)] = Path(path)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--positive-cap", type=int, default=12)
    parser.add_argument("--hard-negative-cap", type=int, default=12)
    parser.add_argument("--background-cap", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = build_event_dataset(
        _task_mapping(args.task),
        args.output_root,
        caps={
            "positive_event": args.positive_cap,
            "hard_negative": args.hard_negative_cap,
            "background": args.background_cap,
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
