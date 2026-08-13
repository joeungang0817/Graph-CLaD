"""Load and prepare Phase 2D demo-derived samples for Phase 3.

The official-demo artifact is stored as one JSON object per demonstration in
JSONL (optionally gzip-compressed), while ``phase3_offline_probe.py`` consumes
the flattened ``{"samples": [...]}`` contract.  This module is the single
place where that storage-format adaptation happens; notebooks should not
reimplement it in ad-hoc cells.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


VALID_SPLITS = frozenset({"train", "validation", "test"})


def _decode_split(value: Any) -> str | None:
    """Extract a canonical split token from plain or serialized metadata."""

    if isinstance(value, str):
        token = value.strip()
        if token in VALID_SPLITS:
            return token
        try:
            return _decode_split(json.loads(token))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if isinstance(value, Mapping):
        for key in ("in_task", "task_heldout", "task_generalization", "value"):
            decoded = _decode_split(value.get(key))
            if decoded is not None:
                return decoded
    return None


def _read_payloads(path: Path) -> Iterable[Mapping[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    if path.name.endswith((".jsonl", ".jsonl.gz")):
        with opener(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"record {line_number} in {path} is not an object")
                yield payload
        return
    with opener(path, "rt", encoding="utf-8") as stream:
        text = stream.read()
    if not text.strip():
        return
    stripped = text.lstrip()
    if stripped[0] in "[{":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, Mapping):
        yield payload
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        yield from (item for item in payload if isinstance(item, Mapping))
    else:
        raise ValueError(f"unsupported payload in {path}")


def iter_samples(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    """Yield flattened samples while preserving episode-level split metadata."""

    for path in paths:
        for record in _read_payloads(Path(path)):
            outer_split = _decode_split(record.get("split"))
            nested = record.get("samples")
            is_episode_record = isinstance(nested, list)
            items = nested if is_episode_record else [record]
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                sample = dict(item)
                # A demo record's manifest split is authoritative.  The
                # Phase 2 transition builder may assign a temporary nested
                # split while processing one episode at a time.
                split = outer_split if is_episode_record and outer_split else _decode_split(sample.get("split"))
                if split is not None:
                    sample["split"] = split
                yield sample


def load_samples(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Load all flattened samples from one or more JSON/JSONL artifacts."""

    return list(iter_samples(paths))


def build_smoke_dataset(
    paths: Iterable[Path],
    per_split: int = 100,
) -> dict[str, Any]:
    """Create a bounded, split-balanced dataset for a fast smoke run.

    Selection is deterministic in source order.  The function refuses to
    silently fabricate a split, and it requires all three canonical splits so
    a smoke run cannot pass with a train-only sample by accident.
    """

    if per_split <= 0:
        raise ValueError("per_split must be positive")
    buckets = {split: [] for split in sorted(VALID_SPLITS)}
    for sample in iter_samples(paths):
        split = sample.get("split")
        if split in buckets and len(buckets[split]) < per_split:
            buckets[split].append(sample)
        if all(len(rows) >= per_split for rows in buckets.values()):
            break
    missing = {split: len(rows) for split, rows in buckets.items() if len(rows) < per_split}
    if missing:
        raise ValueError(f"insufficient split coverage for smoke dataset: {missing}")
    samples = [sample for split in ("train", "validation", "test") for sample in buckets[split]]
    return {
        "dataset_version": "phase2d-smoke-input.v1",
        "source_format": "phase2d-demo-jsonl",
        "split": {"unit": "episode", "selection": "deterministic_source_order"},
        "split_counts": {split: len(rows) for split, rows in buckets.items()},
        "samples": samples,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-split", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = [path for path in args.input]
    if args.per_split:
        payload = build_smoke_dataset(paths, per_split=args.per_split)
    else:
        samples = load_samples(paths)
        payload = {
            "dataset_version": "phase2d-flattened.v1",
            "source_format": "phase2d-demo-jsonl",
            "samples": samples,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(payload["samples"]), "split_counts": payload.get("split_counts")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
