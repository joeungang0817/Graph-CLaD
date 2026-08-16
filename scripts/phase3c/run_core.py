"""Run the configured Phase 3C model/fold/seed screen sequentially."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any

from .io import load_json_config, write_json
from .train_core import train


CORE_MODELS = (
    "C3-Sem-PastAct",
    "C3-SceneSet-PastAct",
    "C3-Pair-PastAct",
    "C3-GeomMPNN-PastAct",
    "C3-RelPool-PastAct",
    "C3-RelMPNN-PastAct",
)


def _path(value: Any) -> Path:
    raw = os.path.expandvars(str(value))
    if "$" in raw:
        raise ValueError(f"unresolved environment variable in path: {value}")
    return Path(raw).expanduser()


def run(config: dict[str, Any]) -> dict[str, Any]:
    models = tuple(str(value) for value in config.get("models", CORE_MODELS))
    folds = tuple(str(value) for value in config.get("folds", [config.get("split", "train")]))
    seeds = tuple(int(value) for value in config.get("seeds", [config.get("seed", 0)]))
    output_root = _path(config.get("output_root", "phase3c_runs"))
    base_checkpoints = config.get("base_checkpoints", config.get("base_checkpoint"))
    if base_checkpoints is None:
        raise ValueError("run_core requires base_checkpoint or base_checkpoints")
    results: list[dict[str, Any]] = []
    for model_id in models:
        if model_id == "C3-Sem-PastAct":
            # The semantic candidate is still trained through the same common
            # head; a zero structured branch is supplied by train_core later.
            # Keep it in the run manifest so the six-model screen is explicit.
            pass
        for fold in folds:
            for seed in seeds:
                if isinstance(base_checkpoints, dict):
                    base_value = base_checkpoints.get(fold, base_checkpoints.get(str(fold)))
                else:
                    base_value = base_checkpoints
                if base_value is None:
                    raise KeyError(f"no base checkpoint configured for fold={fold}")
                run_config = copy.deepcopy(config)
                run_config.update({
                    "model_id": model_id,
                    "split": str(config.get("train_split", "train")),
                    "seed": seed,
                    "base_checkpoint": base_value,
                    "output_root": str(output_root / model_id / fold / f"seed{seed}"),
                })
                match = re.fullmatch(r"test_task(\d+)", fold)
                if match:
                    run_config["held_out_task_id"] = int(match.group(1))
                result = train(run_config)
                results.append(result)
    summary = {
        "schema": "phase3c-core-screen.v1",
        "status": "completed",
        "models": list(models),
        "folds": list(folds),
        "seeds": list(seeds),
        "runs": results,
    }
    write_json(output_root / "screen_manifest.json", summary)
    return summary


def main() -> None:  # pragma: no cover - SSH GPU CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(load_json_config(args.config)), ensure_ascii=False))


if __name__ == "__main__":
    main()
