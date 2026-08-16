"""Train one frozen-base CLaD checkpoint per held-out task fold."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any

from .io import load_json_config, write_json
from .train_base_clad import train


def _path(value: Any) -> Path:
    raw = os.path.expandvars(str(value))
    if "$" in raw:
        raise ValueError(f"unresolved environment variable in path: {value}")
    return Path(raw).expanduser()


def run(config: dict[str, Any]) -> dict[str, Any]:
    folds = tuple(str(value) for value in config.get("folds", ["test_task0", "test_task1", "test_task2"]))
    seeds = tuple(int(value) for value in config.get("seeds", [config.get("seed", 0)]))
    output_root = _path(config.get("output_root", "phase3c_base_clad"))
    results: list[dict[str, Any]] = []
    for fold in folds:
        for seed in seeds:
            match = re.fullmatch(r"test_task(\d+)", fold)
            run_config = copy.deepcopy(config)
            run_config.update({
                "seed": seed,
                "output_root": str(output_root / fold / f"seed{seed}"),
            })
            if match:
                run_config["held_out_task_id"] = int(match.group(1))
            results.append(train(run_config))
    summary = {"schema": "phase3c-base-clad-screen.v1", "status": "completed", "folds": list(folds), "seeds": list(seeds), "runs": results}
    write_json(output_root / "screen_manifest.json", summary)
    return summary


def main() -> None:  # pragma: no cover - SSH GPU CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(load_json_config(args.config)), ensure_ascii=False))


if __name__ == "__main__":
    main()
