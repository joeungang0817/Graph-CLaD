"""Compare the two arms of the controlled Stage 2 one-task pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from scripts.phase3c.io import load_json_config, write_json


def _path(value: Any) -> Path:
    raw = os.path.expandvars(str(value))
    if "$" in raw:
        raise ValueError(f"unresolved environment variable in path: {value}")
    return Path(raw).expanduser()


def _runtime(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping) or value.get("schema") != "phase5-policy-run.v1":
        raise ValueError(f"unsupported Stage 2 runtime manifest: {path}")
    if value.get("status") != "completed":
        raise ValueError(f"Stage 2 runtime is not completed: {path}")
    return value


def analyze(config: Mapping[str, Any]) -> dict[str, Any]:
    semantic = _runtime(_path(config["semantic_runtime"]))
    graph = _runtime(_path(config["graph_runtime"]))
    for key in ("policy_manifest_sha256", "base_checkpoint_sha256", "seed", "updates", "batch_size"):
        if semantic.get(key) != graph.get(key):
            raise ValueError(f"pilot arms are not matched on {key}")
    semantic_loss = float(semantic["best_validation_ddpm_loss"])
    graph_loss = float(graph["best_validation_ddpm_loss"])
    semantic_sensitivity = semantic.get("foresight_sensitivity") or {}
    graph_sensitivity = graph.get("foresight_sensitivity") or {}
    return {
        "schema": "phase5-policy-pilot-analysis.v1",
        "status": "completed",
        "claim_scope": "stage2-one-task-preliminary-pilot",
        "semantic_runtime": str(config["semantic_runtime"]),
        "graph_runtime": str(config["graph_runtime"]),
        "graph_arm": graph.get("structured_model_id"),
        "matched": {
            "policy_manifest_sha256": semantic["policy_manifest_sha256"],
            "base_checkpoint_sha256": semantic["base_checkpoint_sha256"],
            "seed": semantic["seed"],
            "updates": semantic["updates"],
            "batch_size": semantic["batch_size"],
        },
        "validation_ddpm_loss": {
            "semantic": semantic_loss,
            "graph": graph_loss,
            "difference_graph_minus_semantic": graph_loss - semantic_loss,
            "lower_is_better": True,
        },
        "foresight_sensitivity": {
            "semantic": semantic_sensitivity,
            "graph": graph_sensitivity,
            "zeroed_mean_abs_delta_graph_minus_semantic": float(
                graph_sensitivity.get("zeroed_mean_abs_delta", 0.0)
            ) - float(semantic_sensitivity.get("zeroed_mean_abs_delta", 0.0)),
            "shuffled_mean_abs_delta_graph_minus_semantic": float(
                graph_sensitivity.get("shuffled_mean_abs_delta", 0.0)
            ) - float(semantic_sensitivity.get("shuffled_mean_abs_delta", 0.0)),
        },
        "limitations": [
            "one task and one seed",
            "offline DDPM validation loss only",
            "no LIBERO rollout or success-rate claim",
            "not an official CLaD Stage 2 reproduction",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(load_json_config(args.config))
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
