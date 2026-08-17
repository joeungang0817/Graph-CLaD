"""Run the configured Phase 3C model/fold/seed screen sequentially."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .io import load_json_config, write_json
from .contracts import canonical_sha256
from .models.adapters import Phase3CAdapter, SemanticPastActEncoder
from .models.structured import build_structured_model
from .parameter_match import select_width, trainable_parameter_count
from .train_core import train


CORE_MODELS = (
    "C3-Sem-PastAct",
    "C3-SceneSet-PastAct",
    "C3-Pair-PastAct",
    "C3-GeomMPNN-PastAct",
    "C3-RelPool-PastAct",
    "C3-RelMPNN-PastAct",
)


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
    path: Path, *, model_id: str, fold: str, seed: int, config_sha256: str
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("status") != "completed":
        return None
    if (
        str(value.get("model_id")) != model_id
        or str(value.get("fold")) != fold
        or int(value.get("seed", -1)) != seed
        or str(value.get("config_sha256")) != config_sha256
    ):
        raise ValueError(f"completed runtime identity mismatch: {path}")
    for path_key, hash_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("resume_checkpoint", "resume_checkpoint_sha256"),
        ("metrics", "metrics_sha256"),
        ("predictions", "predictions_sha256"),
    ):
        if not value.get(path_key) or not value.get(hash_key):
            raise ValueError(f"completed core runtime is missing {path_key}/{hash_key}")
        artifact = Path(str(value[path_key]))
        if not artifact.exists():
            raise FileNotFoundError(f"completed core artifact is missing: {artifact}")
        if _sha256_file(artifact) != str(value.get(hash_key, "")):
            raise ValueError(f"completed core artifact hash mismatch: {artifact}")
    return value


def run(config: dict[str, Any]) -> dict[str, Any]:
    models = tuple(str(value) for value in config.get("models", CORE_MODELS))
    unknown_models = sorted(set(models) - set(CORE_MODELS))
    if unknown_models:
        raise ValueError(f"unknown Phase 3C models: {unknown_models}")
    folds = tuple(str(value) for value in config.get("folds", [config.get("split", "train")]))
    seeds = tuple(int(value) for value in config.get("seeds", [config.get("seed", 0)]))
    output_root = _path(config.get("output_root", "phase3c_runs"))
    base_checkpoints = config.get("base_checkpoints", config.get("base_checkpoint"))
    if base_checkpoints is None:
        raise ValueError("run_core requires base_checkpoint or base_checkpoints")
    vl_dim = int(config.get("vl_dim", 1024))
    structured_dim = int(config.get("structured_dim", 256))
    reference_width = int(config.get("parameter_reference_width", 128))
    tolerance = float(config.get("parameter_tolerance", 0.05))
    width_candidates = tuple(
        int(value) for value in config.get("width_candidates", range(64, 385, 8))
    )

    def adapter_for(model_id: str, width: int) -> Phase3CAdapter:
        if model_id == "C3-Sem-PastAct":
            structured = SemanticPastActEncoder(
                structured_dim, hidden_dim=width
            )
        else:
            structured = build_structured_model(
                model_id, hidden_dim=width, output_dim=structured_dim
            )
        return Phase3CAdapter(
            structured,
            semantic_dim=2 * vl_dim,
            structured_dim=structured_dim,
        )

    reference_model = str(
        config.get("parameter_reference_model", "C3-RelMPNN-PastAct")
    )
    if reference_model not in CORE_MODELS:
        raise ValueError(f"unknown parameter reference model: {reference_model}")
    target_parameters = trainable_parameter_count(
        adapter_for(reference_model, reference_width)
    )
    parameter_match: dict[str, Any] = {}
    for model_id in models:
        report = select_width(
            lambda width, name=model_id: adapter_for(name, width),
            width_candidates,
            target_parameters,
            tolerance=tolerance,
        )
        if not report["within_tolerance"]:
            raise ValueError(
                f"parameter match failed for {model_id}: "
                f"relative_error={report['relative_error']:.4f}"
            )
        parameter_match[model_id] = report
    results: list[dict[str, Any]] = []
    for model_id in models:
        selected_width = int(parameter_match[model_id]["selected"]["width"])
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
                    "fold": fold,
                    "split": str(config.get("train_split", "train")),
                    "seed": seed,
                    "base_checkpoint": base_value,
                    "hidden_dim": selected_width,
                    "output_root": str(output_root / model_id / fold / f"seed{seed}"),
                })
                match = re.fullmatch(r"test_task(\d+)", fold)
                if match:
                    run_config["held_out_task_id"] = int(match.group(1))
                runtime_path = Path(run_config["output_root"]) / "runtime_manifest.json"
                result = _completed_runtime(
                    runtime_path,
                    model_id=model_id,
                    fold=fold,
                    seed=seed,
                    config_sha256=canonical_sha256(run_config),
                )
                if result is None:
                    result = train(run_config)
                results.append(result)
    summary = {
        "schema": "phase3c-core-screen.v3",
        "status": "completed",
        "models": list(models),
        "folds": list(folds),
        "seeds": list(seeds),
        "parameter_reference": {
            "model_id": reference_model,
            "width": reference_width,
            "trainable_parameters": target_parameters,
            "tolerance": tolerance,
        },
        "parameter_match": parameter_match,
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
