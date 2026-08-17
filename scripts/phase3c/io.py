"""Streaming JSONL and atomic artifact helpers for Phase 3C."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from datetime import datetime, timezone
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any


VALID_SPLITS = frozenset({"train", "validation", "test"})


def decode_split(value: Any) -> str | None:
    if isinstance(value, str):
        token = value.strip()
        if token in VALID_SPLITS:
            return token
        try:
            return decode_split(json.loads(token))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if isinstance(value, Mapping):
        for key in ("in_task", "task_heldout", "task_generalization", "value"):
            decoded = decode_split(value.get(key))
            if decoded is not None:
                return decoded
    return None


def _open_text(path: Path, mode: str):
    if path.name.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def iter_json_objects(path: Path) -> Iterator[Mapping[str, Any]]:
    """Read a Phase 2D JSON/JSONL(.gz) file without loading it wholly."""

    with _open_text(path, "rt") as handle:
        if path.name.endswith((".jsonl", ".jsonl.gz")):
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                yield value
            return
        text = handle.read()
    if not text.strip():
        return
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                yield row
        return
    if isinstance(value, Mapping):
        yield value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise ValueError(f"{path}[{index}] is not a JSON object")
            yield item
    else:
        raise ValueError(f"unsupported JSON root in {path}")


def iter_phase2d_samples(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    """Flatten Phase 2D demo records while preserving authoritative outer split."""

    for path in paths:
        for line_number, record in enumerate(iter_json_objects(Path(path)), 1):
            outer_split = decode_split(record.get("split"))
            nested = record.get("samples")
            items = nested if isinstance(nested, list) else [record]
            for item_index, item in enumerate(items):
                if not isinstance(item, Mapping):
                    raise ValueError(f"{path}:{line_number}[{item_index}] sample is not an object")
                sample = dict(item)
                split = outer_split if isinstance(nested, list) and outer_split else decode_split(sample.get("split"))
                if split is None:
                    raise ValueError(f"{path}:{line_number}[{item_index}] has no canonical split")
                sample["split"] = split
                sample["_source_path"] = str(path)
                sample["_source_line"] = line_number
                yield sample


def load_json_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"config root must be an object: {path}")
    return value


@contextmanager
def atomic_json(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            yield handle
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def atomic_jsonl(path: Path):
    """Yield a text stream and replace the destination only on clean exit."""

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".jsonl.gz.tmp" if path.name.endswith(".gz") else ".jsonl.tmp"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=suffix, dir=str(path.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        if path.name.endswith(".gz"):
            with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                yield handle
        else:
            with temporary.open("w", encoding="utf-8") as handle:
                yield handle
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_json_line(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    )


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    with atomic_json(path) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def set_run_state(root: Path, status: str, details: Mapping[str, Any]) -> Path:
    """Atomically expose one unambiguous RUNNING/COMPLETED/FAILED marker."""

    normalized = str(status).upper()
    allowed = ("RUNNING", "COMPLETED", "FAILED")
    if normalized not in allowed:
        raise ValueError(f"unsupported run state: {status}")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in allowed:
        if name != normalized:
            (root / f"{name}.json").unlink(missing_ok=True)
    target = root / f"{normalized}.json"
    write_json(
        target,
        {
            "schema": "phase3c-run-state.v1",
            "status": normalized.lower(),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **dict(details),
        },
    )
    return target
