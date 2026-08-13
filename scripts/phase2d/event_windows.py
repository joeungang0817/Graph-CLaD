"""Classify official-demo graph windows for event-centered sampling.

The classifier is deliberately based on labels already derived from exact
state replay.  It never creates a negative label from an unknown relation and
does not add any event metadata to the model input tensors.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


def _relation_record(
    graph: Mapping[str, Any], source: str, target: str, relation: str
) -> Mapping[str, Any] | None:
    for edge in graph.get("edges", []):
        if not isinstance(edge, Mapping):
            continue
        if str(edge.get("source")) != source or str(edge.get("target")) != target:
            continue
        relations = edge.get("relations") or edge.get("semantic_relation")
        if isinstance(relations, Mapping) and isinstance(relations.get(relation), Mapping):
            return relations[relation]
    return None


def _pair_values(graph: Mapping[str, Any], relation: str) -> list[tuple[str, bool | None, bool]]:
    values: list[tuple[str, bool | None, bool]] = []
    for edge in graph.get("edges", []):
        if not isinstance(edge, Mapping):
            continue
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source != "robot0" or target == "robot0":
            continue
        record = _relation_record(graph, source, target, relation)
        if not isinstance(record, Mapping):
            continue
        valid = bool(record.get("valid"))
        value = record.get("value") if valid else None
        values.append((target, bool(value) if value is not None else None, valid))
    return values


def _event_tags(events: list[Mapping[str, Any]], start_step: int, target_step: int) -> list[str]:
    tags: set[str] = set()
    for event in events:
        try:
            step = int(event.get("step"))
        except (TypeError, ValueError):
            continue
        if not start_step < step <= target_step:
            continue
        if event.get("event") != "holding_state_change":
            continue
        from_state = str(event.get("from_state", ""))
        to_state = str(event.get("to_state", ""))
        if to_state == "holding":
            tags.add("holding_onset")
        elif from_state == "holding" and to_state == "release":
            tags.add("holding_release")
        elif to_state == "contact_candidate":
            tags.add("contact_candidate")
    return sorted(tags)


def classify_sample(record: Mapping[str, Any], sample: Mapping[str, Any]) -> dict[str, Any]:
    """Return event metadata for one future-relation sample.

    Positive windows include holding onset, active holding, and release.  A
    contact-only window is a hard negative only when both current and future
    holding records are known and false.  Unknown holding records are marked
    ambiguous and are never silently converted to negatives.
    """

    start_step = int(sample.get("start_step", 0))
    target_step = int(sample.get("target_step", start_step))
    events = [event for event in record.get("events", []) if isinstance(event, Mapping)]
    tags = _event_tags(events, start_step, target_step)
    current_graph = sample.get("graph_t") or {}
    future_graph = sample.get("graph_target") or {}
    current_holding = _pair_values(current_graph, "holding")
    future_holding = _pair_values(future_graph, "holding")
    current_contact = _pair_values(current_graph, "contact")
    future_contact = _pair_values(future_graph, "contact")

    holding_by_object = {object_id: (value, valid) for object_id, value, valid in current_holding}
    future_holding_by_object = {object_id: (value, valid) for object_id, value, valid in future_holding}
    all_objects = sorted(set(holding_by_object) | set(future_holding_by_object))
    positive_objects = [
        object_id
        for object_id in all_objects
        if (
            holding_by_object.get(object_id, (None, False))[1]
            and future_holding_by_object.get(object_id, (None, False))[1]
            and (
                bool(holding_by_object[object_id][0])
                or bool(future_holding_by_object[object_id][0])
            )
        )
    ]
    holding_event_objects = [
        str(event.get("object_id"))
        for event in events
        if event.get("event") == "holding_state_change"
        and start_step < int(event.get("step", -1)) <= target_step
        and str(event.get("object_id")) not in {"None", ""}
        and (
            event.get("to_state") == "holding"
            or (event.get("from_state") == "holding" and event.get("to_state") == "release")
        )
    ]
    positive_objects = sorted(set(positive_objects) | set(holding_event_objects))

    contact_objects = {
        object_id
        for object_id, value, valid in current_contact + future_contact
        if valid and value
    }
    known_holding_objects = {
        object_id
        for object_id, _, valid in current_holding + future_holding
        if valid
    }
    hard_negative_objects = sorted(
        object_id
        for object_id in contact_objects & known_holding_objects
        if object_id not in positive_objects
        and not bool(holding_by_object.get(object_id, (False, True))[0])
        and not bool(future_holding_by_object.get(object_id, (False, True))[0])
    )

    if positive_objects:
        category = "positive_event"
        tags = sorted(set(tags) | ({"holding_active"} if not tags else set()))
        objects = positive_objects
    elif hard_negative_objects:
        category = "hard_negative"
        tags = sorted(set(tags) | {"contact_without_holding"})
        objects = hard_negative_objects
    elif any(not valid for _, _, valid in current_holding + future_holding):
        category = "ambiguous"
        tags = sorted(set(tags) | {"holding_unknown"})
        objects = []
    else:
        category = "background"
        objects = []

    return {
        "event_category": category,
        "event_tags": tags,
        "event_object_ids": objects,
        "event_source": "official_hdf5_state_replay",
        "event_window": {"start_step": start_step, "target_step": target_step},
    }


def decorate_record_samples(record: Mapping[str, Any]) -> tuple[dict[str, Any], Counter[str]]:
    output = dict(record)
    decorated: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for original in record.get("samples", []):
        if not isinstance(original, Mapping):
            continue
        sample = dict(original)
        metadata = classify_sample(record, sample)
        sample.update(metadata)
        decorated.append(sample)
        counts[metadata["event_category"]] += 1
    output["samples"] = decorated
    output["event_window_definition"] = "holding_event_windows_v1"
    return output, counts
