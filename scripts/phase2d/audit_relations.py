"""Audit Phase 2D event-centered relation coverage and sample contracts.

The full-demo converter writes one JSON object per demonstration as JSONL
(optionally gzip-compressed).  This audit deliberately treats unsupported
relations as ``valid=0`` rather than silently converting them to negatives.
It also checks that the action-window metadata is present and that goal/task
metadata does not leak into graph inputs.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


RELATIONS = (
    "left",
    "right",
    "front",
    "behind",
    "above",
    "below",
    "contact",
    "on",
    "inside",
    "holding",
    "open",
    "close",
)
EVENT_KEYS = ("event_type", "type", "kind", "event", "name")
STATE_KEYS = ("state", "holding_state", "from_state", "to_state")
LEAK_KEYS = {"is_object_of_interest", "task_semantics", "bddl", "goal", "goal_state"}


def _read_text(path: Path) -> str:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return stream.read()


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load JSON, JSONL, or gzipped JSONL into demonstration records."""

    text = _read_text(path)
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped[0] in "[{":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, Mapping):
        for key in ("records", "episodes", "demos", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError(f"unsupported dataset payload in {path}")
    return [row for row in payload if isinstance(row, dict)]


def _as_bool(value: Any) -> bool:
    return value is True or value == 1 or value == "true"


def _task_key(record: Mapping[str, Any]) -> str:
    if record.get("suite") is not None:
        return f"{record.get('suite')}:{record.get('task_id')}"
    return str(record.get("task_id", "unknown"))


def _demo_key(record: Mapping[str, Any], fallback: str) -> str:
    for key in ("demo_key", "demo_id", "episode_id", "id"):
        if record.get(key) is not None:
            return str(record[key])
    return fallback


def _list_field(record: Mapping[str, Any], *keys: str) -> list[Any]:
    for key in keys:
        value = record.get(key)
        if isinstance(value, list):
            return value
    return []


def _relation_records(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[str, Mapping[str, Any], tuple[str, ...]]]:
    """Yield relation label records wherever they occur in a graph/frame."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            next_path = path + (key_text,)
            if key_text in RELATIONS and isinstance(child, Mapping):
                if "valid" in child or "value" in child or "definition_version" in child:
                    yield key_text, child, next_path
            yield from _relation_records(child, next_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _relation_records(child, path + (str(index),))


def _record_counts(value: Any) -> dict[str, Counter[str]]:
    counts = {relation: Counter() for relation in RELATIONS}
    seen: set[tuple[int, str]] = set()
    for relation, label, path in _relation_records(value):
        marker = (id(label), relation)
        if marker in seen:
            continue
        seen.add(marker)
        counts[relation]["records"] += 1
        counts[relation]["valid"] += int(_as_bool(label.get("valid")))
        counts[relation]["positive"] += int(_as_bool(label.get("valid")) and _as_bool(label.get("value")))
        counts[relation]["invalid"] += int(not _as_bool(label.get("valid")))
    return counts


def _merge_counts(target: Counter[str], source: Counter[str]) -> None:
    for key, value in source.items():
        target[key] += value


def _event_name(event: Mapping[str, Any]) -> str | None:
    for key in EVENT_KEYS:
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    relation = event.get("relation")
    if isinstance(relation, str) and relation:
        return relation
    return None


def _holding_evidence(label: Mapping[str, Any]) -> Counter[str]:
    evidence = label.get("evidence")
    if not isinstance(evidence, Mapping):
        return Counter()
    result = Counter()
    for key in ("finger_contact", "closed_gripper", "relative_pose_stable", "object_followed_eef"):
        if key in evidence:
            result[key] += int(_as_bool(evidence[key]))
    return result


def _walk_keys(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[str, ...]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            next_path = path + (str(key),)
            yield next_path
            yield from _walk_keys(child, next_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_keys(child, path + (str(index),))


def _input_leak_keys(sample: Mapping[str, Any]) -> list[str]:
    input_fields = (
        "graph_t",
        "graph_history",
        "history_graph",
        "input_graph",
        "past_graph",
        "frames",
    )
    paths: list[str] = []
    for field in input_fields:
        if field not in sample:
            continue
        for path in _walk_keys(sample[field]):
            if any(part in LEAK_KEYS for part in path):
                paths.append(field + "." + ".".join(path))
    return sorted(set(paths))


def audit_records(records: list[Mapping[str, Any]], source: str) -> dict[str, Any]:
    relation_by_task: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {relation: Counter() for relation in RELATIONS}
    )
    event_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    holding_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    transition_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    horizon_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    change_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    task_records: Counter[str] = Counter()
    task_frames: Counter[str] = Counter()
    task_samples: Counter[str] = Counter()
    missing_action_window: Counter[str] = Counter()
    future_action_fields: Counter[str] = Counter()
    leakage_paths: Counter[str] = Counter()

    for index, record in enumerate(records):
        task = _task_key(record)
        task_records[task] += 1
        frames = _list_field(record, "frames", "graph_frames")
        samples = _list_field(record, "sample_index", "samples")
        events = _list_field(record, "events", "relation_events")
        task_frames[task] += len(frames)
        task_samples[task] += len(samples)

        for frame in frames:
            for relation, counts in _record_counts(frame).items():
                _merge_counts(relation_by_task[task][relation], counts)
                if relation == "holding":
                    for relation_name, label, _path in _relation_records(frame):
                        if relation_name == "holding":
                            evidence = _holding_evidence(label)
                            for key, value in evidence.items():
                                holding_by_task[task][key] += value
                            holding_by_task[task]["evidence_records"] += int(bool(evidence))

        for event in events:
            if not isinstance(event, Mapping):
                continue
            name = _event_name(event)
            if name:
                event_by_task[task][name] += 1
            for key in STATE_KEYS:
                value = event.get(key)
                if isinstance(value, str) and value:
                    transition_by_task[task][value] += 1
            if isinstance(event.get("relation"), str) and event.get("relation") in RELATIONS:
                change_by_task[task][str(event["relation"])] += 1

        for sample in samples:
            if not isinstance(sample, Mapping):
                continue
            if sample.get("horizon") is not None:
                horizon_by_task[task][str(sample["horizon"])] += 1
            has_window = any(
                isinstance(sample.get(key), list)
                for key in ("action_window", "past_action_window", "actions")
            )
            if not has_window:
                missing_action_window[task] += 1
            for key in sample:
                if "future" in str(key).lower() and "action" in str(key).lower():
                    future_action_fields[task] += 1
            for key, changed in (sample.get("relation_changes") or {}).items():
                if changed and "::" in str(key):
                    relation = str(key).rsplit("::", 1)[1]
                    if relation in RELATIONS:
                        change_by_task[task][relation] += int(changed)
            for path in _input_leak_keys(sample):
                leakage_paths[task + ":" + path] += 1

    coverage = {}
    for task in sorted(task_records):
        coverage[task] = {
            relation: dict(relation_by_task[task][relation])
            for relation in RELATIONS
            if relation_by_task[task][relation]
        }

    missing_relations = {
        task: [relation for relation in RELATIONS if not relation_by_task[task][relation].get("valid")]
        for task in sorted(task_records)
    }
    warnings: list[str] = []
    for task, relations in missing_relations.items():
        if relations:
            warnings.append(f"{task}: no valid labels for {', '.join(relations)}")
    if any(missing_action_window.values()):
        warnings.append("some samples do not expose an action window field")
    if future_action_fields:
        warnings.append("future-action fields were found in sample metadata")
    if leakage_paths:
        warnings.append("goal/task metadata appears inside a model-input graph field")

    return {
        "audit_version": "phase2d-relation-audit.v1",
        "source": source,
        "status": "pass_with_warnings" if warnings else "pass",
        "records": dict(task_records),
        "frames": dict(task_frames),
        "samples": dict(task_samples),
        "coverage": coverage,
        "missing_valid_relations_by_task": missing_relations,
        "events_by_task": {task: dict(counter) for task, counter in sorted(event_by_task.items())},
        "holding_evidence_by_task": {
            task: dict(counter) for task, counter in sorted(holding_by_task.items())
        },
        "holding_state_tokens_by_task": {
            task: dict(counter) for task, counter in sorted(transition_by_task.items())
        },
        "relation_changes_by_task": {
            task: dict(counter) for task, counter in sorted(change_by_task.items())
        },
        "horizons_by_task": {
            task: dict(counter) for task, counter in sorted(horizon_by_task.items())
        },
        "sample_contract": {
            "missing_action_window_by_task": dict(missing_action_window),
            "future_action_field_count_by_task": dict(future_action_fields),
        },
        "input_metadata_leakage_paths": dict(leakage_paths),
        "warnings": warnings,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    all_records: list[dict[str, Any]] = []
    for path in args.dataset:
        all_records.extend(load_records(path))
    report = audit_records(all_records, ",".join(str(path) for path in args.dataset))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
