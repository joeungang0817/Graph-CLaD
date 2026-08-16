"""Pure Phase 3C contracts.

This module contains the parts of the Phase 3C protocol that must not depend
on torch, LIBERO, or a particular model.  Keeping the join, target, and
forbidden-input rules here makes it possible to test the causal boundary on a
CPU before a GPU or DecisionNCE installation is required.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


PHASE3C_SCHEMA_VERSION = "phase3c-joined-sample.v1"
PRIMARY_RELATIONS = (
    "left",
    "right",
    "front",
    "behind",
    "above",
    "below",
    "contact",
    "on",
)
TARGET_SOURCE_TYPES = frozenset({"object"})
TARGET_DESTINATION_TYPES = frozenset({"object", "fixture"})
ACTION_DIM = 7
TAU = 6

# These names are rejected only from the model-input view.  Target payloads
# are allowed to retain future graphs and labels for evaluation.
FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "graph_target",
        "future_graph",
        "future_action",
        "action_future",
        "action_window_future",
        "reward",
        "success",
        "is_object_of_interest",
        "task_semantics",
        "bddl",
        "goal",
        "goal_state",
        "relation_changes",
    }
)


def canonical_json(value: Any) -> bytes:
    """Serialize JSON deterministically for provenance hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_vector(value: Any, dimension: int) -> list[float] | None:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        return None
    if len(value) != dimension:
        return None
    result = [_finite_float(item) for item in value]
    if any(item is None for item in result):
        return None
    return [float(item) for item in result if item is not None]


def _node_map(graph: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    nodes = graph.get("nodes")
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
        raise ValueError("graph.nodes must be a sequence")
    result: dict[str, Mapping[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, Mapping) or "node_id" not in node:
            raise ValueError("graph.nodes contains an invalid node")
        node_id = str(node["node_id"])
        if node_id in result:
            raise ValueError(f"duplicate graph node_id: {node_id}")
        result[node_id] = node
    return result


def _edge_map(graph: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    edges = graph.get("edges")
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)):
        raise ValueError("graph.edges must be a sequence")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for edge in edges:
        if not isinstance(edge, Mapping):
            raise ValueError("graph.edges contains an invalid edge")
        if "source" not in edge or "target" not in edge:
            raise ValueError("graph edge is missing source/target")
        key = (str(edge["source"]), str(edge["target"]))
        if key in result:
            raise ValueError(f"duplicate graph edge: {key[0]}->{key[1]}")
        result[key] = edge
    return result


def validate_graph(graph: Any, *, name: str = "graph") -> None:
    if not isinstance(graph, Mapping):
        raise ValueError(f"{name} must be an object")
    _node_map(graph)
    _edge_map(graph)


def parse_action_window(value: Any, *, tau: int = TAU) -> list[list[float]]:
    """Normalize Phase 2D action rows and enforce the causal shape contract."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("action_window must be a sequence")
    if len(value) != tau:
        raise ValueError(f"action_window must contain exactly {tau} rows")
    rows: list[list[float]] = []
    for index, row in enumerate(value):
        if isinstance(row, Mapping):
            row = row.get("action")
        parsed = _finite_vector(row, ACTION_DIM)
        if parsed is None:
            raise ValueError(f"action_window[{index}] must be a finite {ACTION_DIM}-vector")
        rows.append(parsed)
    return rows


def _relation_record(edge: Mapping[str, Any], relation: str) -> tuple[bool, bool]:
    records = edge.get("relations")
    if not isinstance(records, Mapping):
        records = edge.get("semantic_relation")
    if not isinstance(records, Mapping):
        return False, False
    raw = records.get(relation)
    if not isinstance(raw, Mapping) or not bool(raw.get("valid")):
        return False, False
    value = raw.get("value")
    if not isinstance(value, bool):
        return False, False
    return value, True


def _position(node: Mapping[str, Any]) -> list[float] | None:
    features = node.get("features")
    if not isinstance(features, Mapping) or features.get("position_valid") != 1:
        return None
    return _finite_vector(features.get("position"), 3)


def _candidate_edges(
    current: Mapping[str, Any],
    future: Mapping[str, Any],
) -> list[tuple[tuple[str, str], Mapping[str, Any], Mapping[str, Any]]]:
    current_nodes = _node_map(current)
    future_nodes = _node_map(future)
    current_edges = _edge_map(current)
    future_edges = _edge_map(future)
    candidates: list[tuple[tuple[str, str], Mapping[str, Any], Mapping[str, Any]]] = []
    for key in sorted(set(current_edges) & set(future_edges)):
        source = current_nodes.get(key[0])
        target = current_nodes.get(key[1])
        future_source = future_nodes.get(key[0])
        future_target = future_nodes.get(key[1])
        if not all(isinstance(node, Mapping) for node in (source, target, future_source, future_target)):
            continue
        if str(source.get("node_type")) not in TARGET_SOURCE_TYPES:
            continue
        if str(target.get("node_type")) not in TARGET_DESTINATION_TYPES:
            continue
        candidates.append((key, current_edges[key], future_edges[key]))
    return candidates


def relation_targets(
    current: Mapping[str, Any],
    future: Mapping[str, Any],
    *,
    relations: Sequence[str] = PRIMARY_RELATIONS,
) -> dict[str, Any]:
    """Build masked sample-level any-change labels and edge diagnostics."""

    relation_names = tuple(str(name) for name in relations)
    unknown = sorted(set(relation_names) - set(PRIMARY_RELATIONS))
    if unknown:
        raise ValueError(f"unsupported Phase 3C relations: {unknown}")
    candidates = _candidate_edges(current, future)
    relation_valid: dict[str, int] = {}
    relation_any_change: dict[str, int] = {}
    edge_diagnostics: list[dict[str, Any]] = []
    for relation in relation_names:
        valid_values: list[bool] = []
        changed_values: list[bool] = []
        for key, current_edge, future_edge in candidates:
            current_value, current_valid = _relation_record(current_edge, relation)
            future_value, future_valid = _relation_record(future_edge, relation)
            valid = bool(current_valid and future_valid)
            changed = bool(valid and current_value != future_value)
            valid_values.append(valid)
            changed_values.append(changed)
            edge_diagnostics.append(
                {
                    "source": key[0],
                    "target": key[1],
                    "relation": relation,
                    "current_value": current_value if current_valid else None,
                    "future_value": future_value if future_valid else None,
                    "current_valid": int(current_valid),
                    "future_valid": int(future_valid),
                    "valid": int(valid),
                    "changed": int(changed),
                }
            )
        relation_valid[relation] = int(any(valid_values))
        relation_any_change[relation] = int(any(changed_values))
    return {
        "relation_valid": relation_valid,
        "relation_any_change": relation_any_change,
        "edge_diagnostics": edge_diagnostics,
        "candidate_edge_count": len(candidates),
    }


def scene_max_displacement(
    current: Mapping[str, Any], future: Mapping[str, Any]
) -> float:
    current_nodes = _node_map(current)
    future_nodes = _node_map(future)
    distances: list[float] = []
    for node_id in sorted(set(current_nodes) & set(future_nodes)):
        left = current_nodes[node_id]
        right = future_nodes[node_id]
        if str(left.get("node_type")) != "object":
            continue
        left_position = _position(left)
        right_position = _position(right)
        if left_position is None or right_position is None:
            continue
        distances.append(
            math.sqrt(sum((left_position[i] - right_position[i]) ** 2 for i in range(3)))
        )
    return max(distances, default=0.0)


def model_input_view(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return exactly the part permitted to enter a Phase 3C model."""

    return {
        "schema": record.get("schema"),
        "sample_id": record.get("sample_id"),
        "task_id": record.get("task_id"),
        "episode_id": record.get("episode_id"),
        "demo_key": record.get("demo_key"),
        "split": record.get("split"),
        "tau": record.get("tau"),
        "prev_step": record.get("prev_step"),
        "current_step": record.get("current_step"),
        "past_action_window": record.get("past_action_window"),
        "graph_prev": record.get("graph_prev"),
        "graph_t": record.get("graph_t"),
    }


def forbidden_keys(value: Any) -> list[str]:
    """Find forbidden keys recursively, including their dotted path."""

    found: list[str] = []

    def visit(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                key_text = str(key)
                child_path = f"{path}.{key_text}" if path else key_text
                if key_text in FORBIDDEN_INPUT_KEYS:
                    found.append(child_path)
                visit(child, child_path)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for index, child in enumerate(node):
                visit(child, f"{path}[{index}]")

    visit(value, "")
    return found


def assert_causal_input(record: Mapping[str, Any]) -> None:
    found = forbidden_keys(model_input_view(record))
    if found:
        raise ValueError(f"forbidden future/non-causal fields in model input: {found}")
    parse_action_window(record.get("past_action_window"), tau=int(record.get("tau", TAU)))
    validate_graph(record.get("graph_prev"), name="graph_prev")
    validate_graph(record.get("graph_t"), name="graph_t")


def support_report(
    records: Iterable[Mapping[str, Any]],
    *,
    min_train_positive: int = 20,
    min_train_negative: int = 20,
    min_validation_positive: int = 5,
    min_validation_negative: int = 5,
) -> dict[str, Any]:
    """Compute fold-independent support counts without inspecting test labels."""

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    total = 0
    for record in records:
        total += 1
        split = str(record.get("split"))
        targets = record.get("target", {})
        if not isinstance(targets, Mapping):
            raise ValueError("joined record target must be an object")
        labels = targets.get("relation_any_change", {})
        masks = targets.get("relation_valid", {})
        if not isinstance(labels, Mapping) or not isinstance(masks, Mapping):
            raise ValueError("target relation labels/masks must be objects")
        for relation in PRIMARY_RELATIONS:
            if not bool(masks.get(relation, 0)):
                continue
            counts[relation][f"{split}_valid"] += 1
            counts[relation][f"{split}_positive"] += int(bool(labels.get(relation, 0)))
            counts[relation][f"{split}_negative"] += int(not bool(labels.get(relation, 0)))
    eligibility = {}
    for relation in PRIMARY_RELATIONS:
        item = counts[relation]
        eligible = (
            item["train_positive"] >= min_train_positive
            and item["train_negative"] >= min_train_negative
            and item["validation_positive"] >= min_validation_positive
            and item["validation_negative"] >= min_validation_negative
        )
        eligibility[relation] = {
            "train_positive": item["train_positive"],
            "train_negative": item["train_negative"],
            "validation_positive": item["validation_positive"],
            "validation_negative": item["validation_negative"],
            "test_positive": item["test_positive"],
            "test_negative": item["test_negative"],
            "eligible_without_test": bool(eligible),
        }
    return {
        "contract": "phase3c-relation-support.v1",
        "total_records": total,
        "selection_source": ["train", "validation"],
        "test_counts_inspected_for_selection": False,
        "thresholds": {
            "train_positive": min_train_positive,
            "train_negative": min_train_negative,
            "validation_positive": min_validation_positive,
            "validation_negative": min_validation_negative,
        },
        "relations": eligibility,
    }


@dataclass(frozen=True)
class JoinCounters:
    source_records: int = 0
    source_samples: int = 0
    candidate_left_samples: int = 0
    joined_samples: int = 0
    boundary_drops: int = 0
    missing_right_samples: int = 0
    hash_mismatches: int = 0
    duplicate_left_keys: int = 0
    invalid_samples: int = 0
    emitted_future_action_fields: int = 0

