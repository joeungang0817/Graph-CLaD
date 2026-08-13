"""Remove task-derived relevance metadata from Phase 2D model inputs."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Mapping


ARTIFACT_VERSION = "phase2d-full-demo.v3+input-clean-v1"


def clean_graph(graph: Any) -> Any:
    if not isinstance(graph, Mapping):
        return graph
    output = dict(graph)
    nodes: list[Any] = []
    for original in graph.get("nodes", []):
        if not isinstance(original, Mapping):
            nodes.append(original)
            continue
        node = dict(original)
        features = dict(node.get("features") or {})
        features.pop("is_object_of_interest", None)
        node["features"] = features
        vector = node.get("feature_vector")
        if isinstance(vector, list) and len(vector) > 4:
            vector = list(vector)
            if abs(float(vector[4])) > 1e-9:
                raise ValueError(f"non-zero task flag at node {node.get('node_id')}")
            vector[4] = 0.0
            node["feature_vector"] = vector
        nodes.append(node)
    output["nodes"] = nodes
    return output


def clean_record(record: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(record)
    output["artifact_version"] = ARTIFACT_VERSION
    output["input_feature_schema"] = {
        "name": "node_feature_vector_v2_no_task_flag",
        "node_feature_dim": 24,
        "reserved_zero_slot": "feature_vector[4]",
        "removed_feature_key": "is_object_of_interest",
        "model_input_rule": "do not use task-derived relevance metadata",
    }
    samples: list[Any] = []
    for original in record.get("samples", []):
        if not isinstance(original, Mapping):
            samples.append(original)
            continue
        sample = dict(original)
        sample["graph_t"] = clean_graph(sample.get("graph_t", {}))
        sample["graph_target"] = clean_graph(sample.get("graph_target", {}))
        samples.append(sample)
    output["samples"] = samples
    return output


def clean_dataset(source: Path, output: Path, task_id: int | None = None) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    records = samples = node_occurrences = remaining_keys = nonzero_slots = 0
    temporary = output.with_suffix(output.suffix + ".tmp")
    with gzip.open(source, "rt", encoding="utf-8") as input_file, gzip.open(
        temporary, "wt", encoding="utf-8", compresslevel=1
    ) as output_file:
        for line in input_file:
            if not line.strip():
                continue
            cleaned = clean_record(json.loads(line))
            records += 1
            for sample in cleaned.get("samples", []):
                if not isinstance(sample, Mapping):
                    continue
                samples += 1
                for graph in (sample.get("graph_t", {}), sample.get("graph_target", {})):
                    for node in graph.get("nodes", []) if isinstance(graph, Mapping) else []:
                        if not isinstance(node, Mapping):
                            continue
                        node_occurrences += 1
                        remaining_keys += int("is_object_of_interest" in (node.get("features") or {}))
                        vector = node.get("feature_vector")
                        if isinstance(vector, list) and len(vector) > 4:
                            nonzero_slots += int(abs(float(vector[4])) > 1e-9)
            output_file.write(json.dumps(cleaned, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(output)
    return {
        "qa_version": "phase2d-input-clean.v1",
        "status": "pass" if remaining_keys == 0 and nonzero_slots == 0 else "fail",
        "task_id": task_id,
        "source": str(source),
        "output": str(output),
        "records": records,
        "samples": samples,
        "node_graph_occurrences": node_occurrences,
        "remaining_task_flag_keys": remaining_keys,
        "nonzero_reserved_task_flag_slots": nonzero_slots,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--qa-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = clean_dataset(args.input, args.output, args.task_id)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.qa_output:
        args.qa_output.parent.mkdir(parents=True, exist_ok=True)
        args.qa_output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
