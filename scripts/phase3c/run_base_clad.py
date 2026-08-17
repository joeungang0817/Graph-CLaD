"""Train one frozen-base CLaD checkpoint per held-out task fold."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .io import load_json_config, set_run_state, write_json
from .contracts import canonical_sha256
from .train_base_clad import train


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value: Any) -> Path:
    raw = os.path.expandvars(str(value))
    if "$" in raw:
        raise ValueError(f"unresolved environment variable in path: {value}")
    return Path(raw).expanduser()


def _completed_runtime(
    path: Path, *, fold: str, seed: int, config_sha256: str
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("status") != "completed":
        return None
    if (
        str(value.get("fold")) != fold
        or int(value.get("seed", -1)) != seed
        or str(value.get("config_sha256")) != config_sha256
    ):
        raise ValueError(f"completed runtime seed mismatch: {path}")
    for path_key, hash_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("resume_checkpoint", "resume_checkpoint_sha256"),
    ):
        if not value.get(path_key) or not value.get(hash_key):
            raise ValueError(f"completed base runtime is missing {path_key}/{hash_key}")
        artifact = Path(str(value[path_key]))
        if not artifact.exists():
            raise FileNotFoundError(f"completed base artifact is missing: {artifact}")
        if _sha256_file(artifact) != str(value.get(hash_key, "")):
            raise ValueError(f"completed base artifact hash mismatch: {artifact}")
    return value


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
                "fold": fold,
                "output_root": str(output_root / fold / f"seed{seed}"),
            })
            if match:
                run_config["held_out_task_id"] = int(match.group(1))
            runtime_path = Path(run_config["output_root"]) / "runtime_manifest.json"
            run_root = Path(run_config["output_root"])
            identity = {"fold": fold, "seed": seed, "config_sha256": canonical_sha256(run_config)}
            result = _completed_runtime(
                runtime_path,
                fold=fold,
                seed=seed,
                config_sha256=identity["config_sha256"],
            )
            if result is None:
                write_json(run_root / "run_config.json", run_config)
                set_run_state(run_root, "RUNNING", identity)
                try:
                    result = train(run_config)
                except Exception as exc:
                    set_run_state(
                        run_root,
                        "FAILED",
                        {**identity, "error_type": type(exc).__name__, "error": str(exc)},
                    )
                    raise
            set_run_state(
                run_root,
                "COMPLETED",
                {**identity, "runtime_manifest": str(runtime_path)},
            )
            results.append(result)
    summary = {"schema": "phase3c-base-clad-screen.v3", "status": "completed", "folds": list(folds), "seeds": list(seeds), "runs": results}
    write_json(output_root / "screen_manifest.json", summary)
    return summary


def main() -> None:  # pragma: no cover - SSH GPU CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(load_json_config(args.config)), ensure_ascii=False))


if __name__ == "__main__":
    main()
