"""Train the controlled CLaD Stage 1 objective for one fold/seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset import (
    SemanticFeatureStore,
    collate_phase3c,
    fit_normalization,
    iter_filtered_records,
    iter_shuffled_batches,
)
from .contracts import canonical_sha256
from .io import load_json_config, write_json
from .models.semantic_clad import ControlledCLaD


def _get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in config:
        return config[key]
    for section in ("base_clad", "phase3c", "training"):
        nested = config.get(section)
        if isinstance(nested, dict) and key in nested:
            return nested[key]
    return default


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_sha256() -> str:
    repository = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__),
        repository / "scripts" / "phase3c" / "dataset.py",
        repository / "scripts" / "phase3c" / "models" / "semantic_clad.py",
        repository / "baseline_code" / "LatentDynamics.py",
    )
    return canonical_sha256(
        {str(path.relative_to(repository)): _sha256_file(path) for path in paths}
    )


def _path(value: Any) -> Path:
    raw = os.path.expandvars(str(value))
    if "$" in raw:
        raise ValueError(f"unresolved environment variable in path: {value}")
    return Path(raw).expanduser()


def _configure_determinism(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        warn_only = True
    except TypeError:  # older torch without warn_only
        torch.use_deterministic_algorithms(True)
        warn_only = False
    return {
        "seed": seed,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": True,
        "deterministic_warn_only": warn_only,
    }


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _torch_load(path: Path, *, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location=map_location)


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(value: dict[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if torch.cuda.is_available() and value.get("cuda") is not None:
        torch.cuda.set_rng_state_all(value["cuda"])


def _filtered_batches(
    path: Path,
    *,
    batch_size: int,
    split: str,
    exclude_task_id: int | None,
):
    pending: list[dict[str, Any]] = []
    for record in iter_filtered_records(
        path, split=split, exclude_task_id=exclude_task_id
    ):
        pending.append(record)
        if len(pending) == int(batch_size):
            yield pending
            pending = []
    if pending:
        yield pending


def _stage1_total(losses: dict[str, torch.Tensor], recon_weight: float) -> torch.Tensor:
    return losses["loss_p"] + losses["loss_s"] + float(recon_weight) * (
        losses["loss_p_recon"] + losses["loss_v_recon"]
    )


def _evaluate_stage1(
    *,
    model: ControlledCLaD,
    joined_path: Path,
    store: SemanticFeatureStore,
    normalization: Any,
    device: torch.device,
    batch_size: int,
    max_nodes: int | None,
    split: str,
    exclude_task_id: int | None,
    recon_weight: float,
) -> dict[str, Any]:
    totals = {
        "total": 0.0,
        "loss_p": 0.0,
        "loss_s": 0.0,
        "loss_p_recon": 0.0,
        "loss_v_recon": 0.0,
    }
    rows = 0
    training_states = [(module, bool(module.training)) for module in model.modules()]
    model.train()
    core = getattr(model, "core", None)
    if core is not None:
        # The unchanged CLaD core emits Stage-1 losses only while the root core
        # is in training mode. Keep that branch active but disable stochastic
        # dropout in every descendant for stable validation selection.
        for module in core.modules():
            if module is not core:
                module.training = False
    try:
        with torch.no_grad():
            for records in _filtered_batches(
                joined_path,
                batch_size=batch_size,
                split=split,
                exclude_task_id=exclude_task_id,
            ):
                batch = collate_phase3c(
                    records,
                    store,
                    max_nodes=max_nodes,
                    normalization=normalization,
                    device=device,
                )
                losses = model.training_loss(batch, action_mask_ratio=0.0)
                total = _stage1_total(losses, recon_weight)
                if not torch.isfinite(total):
                    raise FloatingPointError("non-finite validation Stage-1 loss")
                count = len(records)
                rows += count
                totals["total"] += float(total.detach().cpu()) * count
                for name in totals:
                    if name != "total":
                        totals[name] += float(losses[name].detach().cpu()) * count
    finally:
        for module, state in training_states:
            module.training = state
    if rows == 0:
        raise ValueError(
            f"joined manifest has no base CLaD validation records for split={split}"
        )
    return {
        "split": split,
        "rows": rows,
        **{name: value / rows for name, value in totals.items()},
        "action_mask_ratio": 0.0,
        "stochastic_submodules": "disabled",
    }


def train(config: dict[str, Any]) -> dict[str, Any]:
    joined_path = _path(_get(config, "joined_manifest"))
    store_root = _path(_get(config, "semantic_store"))
    output_root = _path(_get(config, "output_root"))
    split = str(_get(config, "split", "train"))
    fold = str(_get(config, "fold", "unspecified"))
    held_out_raw = _get(config, "held_out_task_id")
    held_out_task_id = int(held_out_raw) if held_out_raw is not None else None
    seed = int(_get(config, "seed", 0))
    updates = int(_get(config, "updates", 25_000))
    batch_size = int(_get(config, "batch_size", 128))
    shuffle_buffer = int(_get(config, "shuffle_buffer", max(2048, batch_size * 8)))
    hidden_dim = int(_get(config, "hidden_dim", 1024))
    vl_dim = int(_get(config, "vl_dim", hidden_dim))
    max_nodes = _get(config, "max_nodes")
    max_nodes = int(max_nodes) if max_nodes is not None else None
    learning_rate = float(_get(config, "learning_rate", 1e-4))
    weight_decay = float(_get(config, "weight_decay", 1e-4))
    recon_weight = float(_get(config, "recon_weight", 0.1))
    validation_split = str(_get(config, "validation_split", "validation"))
    validation_interval = int(_get(config, "validation_interval", 500))
    resume = bool(_get(config, "resume", True))
    device = torch.device(str(_get(config, "device", "cuda" if torch.cuda.is_available() else "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requested CUDA but CUDA is unavailable")
    if updates <= 0 or batch_size <= 0 or validation_interval <= 0:
        raise ValueError("updates, batch_size, and validation_interval must be positive")
    if shuffle_buffer < batch_size:
        raise ValueError("shuffle_buffer must be at least batch_size")
    if hidden_dim <= 0 or vl_dim <= 0 or (max_nodes is not None and max_nodes <= 0):
        raise ValueError("hidden_dim, vl_dim, and max_nodes must be positive")
    if learning_rate <= 0.0 or weight_decay < 0.0 or recon_weight < 0.0:
        raise ValueError("optimizer values must be non-negative and learning_rate positive")
    determinism = _configure_determinism(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if not joined_path.exists():
        raise FileNotFoundError(joined_path)
    joined_sha256 = _sha256_file(joined_path)
    normalization = fit_normalization(
        iter_filtered_records(
            joined_path, split=split, exclude_task_id=held_out_task_id
        )
    )
    store = SemanticFeatureStore(store_root)
    store_source_sha = str(
        (store.manifest.get("source") or {}).get("joined_manifest_sha256", "")
    )
    if store_source_sha != joined_sha256:
        raise ValueError("semantic store was not built from the configured joined manifest")
    if store.feature_dim != vl_dim:
        raise ValueError(f"semantic store feature_dim={store.feature_dim} does not match vl_dim={vl_dim}")
    model = ControlledCLaD(proprio_dim=16, vl_dim=vl_dim, hidden_dim=hidden_dim, action_dim=42).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    config_sha256 = canonical_sha256(config)
    store_manifest_sha256 = _sha256_file(store_root / "manifest.json")
    trainer_source_sha256 = _sha256_file(Path(__file__))
    code_sha256 = _code_sha256()
    checkpoint_root = output_root / "checkpoints"
    last_checkpoint = checkpoint_root / "last.pt"
    best_checkpoint = checkpoint_root / "best.pt"
    start_update = 0
    best_update = 0
    best_validation_loss = float("inf")
    validation_history: list[dict[str, Any]] = []
    losses: list[float] = []
    resume_rng: dict[str, Any] | None = None
    if resume and last_checkpoint.exists():
        resumed = _torch_load(last_checkpoint)
        expected = {
            "config_sha256": config_sha256,
            "source_joined_manifest_sha256": joined_sha256,
            "semantic_store_manifest_sha256": store_manifest_sha256,
            "trainer_source_sha256": trainer_source_sha256,
            "code_sha256": code_sha256,
            "fold": fold,
            "seed": seed,
        }
        for name, value in expected.items():
            if resumed.get(name) != value:
                raise ValueError(f"base resume checkpoint {name} mismatch")
        start_update = int(resumed.get("update", 0))
        if start_update < 0 or start_update > updates:
            raise ValueError("base resume checkpoint update is outside requested budget")
        model.load_state_dict(resumed["model_state"])
        model.restore_ema_initialization_state(resumed.get("ema_initialized"))
        optimizer.load_state_dict(resumed["optimizer_state"])
        best_update = int(resumed.get("best_update", 0))
        best_validation_loss = float(resumed.get("best_validation_loss", float("inf")))
        validation_history = list(resumed.get("validation_history", []))
        losses = [float(value) for value in resumed.get("recent_losses", [])]
        resume_rng = resumed.get("rng_state")
    model.train()
    start = time.time()
    training_batches = iter_shuffled_batches(
        joined_path,
        batch_size=batch_size,
        seed=seed,
        split=split,
        exclude_task_id=held_out_task_id,
        shuffle_buffer=shuffle_buffer,
    )
    for _ in range(start_update):
        next(training_batches)
    if resume_rng is not None:
        _restore_rng_state(resume_rng)
    for update in range(start_update + 1, updates + 1):
        batch_records = next(training_batches)
        batch = collate_phase3c(batch_records, store, max_nodes=max_nodes, normalization=normalization, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.training_loss(batch, action_mask_ratio=0.3)
        total = _stage1_total(loss_dict, recon_weight)
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite base CLaD loss at update {update}")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_ema_after_optimizer_step()
        losses.append(float(total.detach().cpu()))
        should_validate = update % validation_interval == 0 or update == updates
        if should_validate:
            training_rng = _capture_rng_state()
            validation = _evaluate_stage1(
                model=model,
                joined_path=joined_path,
                store=store,
                normalization=normalization,
                device=device,
                batch_size=batch_size,
                max_nodes=max_nodes,
                split=validation_split,
                exclude_task_id=held_out_task_id,
                recon_weight=recon_weight,
            )
            _restore_rng_state(training_rng)
            validation_history.append({"update": update, **validation})
            if float(validation["total"]) < best_validation_loss:
                best_validation_loss = float(validation["total"])
                best_update = update
                _atomic_torch_save(
                    best_checkpoint,
                    {
                        "schema": "phase3c-base-clad-checkpoint.v3",
                        "kind": "validation_best",
                        "model_state": model.state_dict(),
                        "selected_update": update,
                        "best_validation_stage1_loss": best_validation_loss,
                        "seed": seed,
                        "fold": fold,
                        "vl_dim": vl_dim,
                        "hidden_dim": hidden_dim,
                        "action_dim": 42,
                        "proprio_dim": 16,
                        "source_joined_manifest_sha256": joined_sha256,
                        "semantic_store_manifest_sha256": store_manifest_sha256,
                        "code_sha256": code_sha256,
                        "normalization": normalization.to_dict(),
                        "ema_initialized": model.ema_initialization_state(),
                    },
                )
            _atomic_torch_save(
                last_checkpoint,
                {
                    "schema": "phase3c-base-clad-resume.v3",
                    "kind": "resume_last",
                    "model_state": model.state_dict(),
                    "ema_initialized": model.ema_initialization_state(),
                    "optimizer_state": optimizer.state_dict(),
                    "update": update,
                    "best_update": best_update,
                    "best_validation_loss": best_validation_loss,
                    "validation_history": validation_history,
                    "recent_losses": losses[-100:],
                    "rng_state": _capture_rng_state(),
                    "seed": seed,
                    "fold": fold,
                    "config_sha256": config_sha256,
                    "trainer_source_sha256": trainer_source_sha256,
                    "code_sha256": code_sha256,
                    "source_joined_manifest_sha256": joined_sha256,
                    "semantic_store_manifest_sha256": store_manifest_sha256,
                },
            )
    if not best_checkpoint.exists() or best_update <= 0:
        raise RuntimeError("base CLaD completed without a validation-best checkpoint")
    elapsed_seconds = time.time() - start
    updates_this_process = max(0, updates - start_update)
    runtime = {
        "schema": "phase3c-base-clad-run.v3",
        "status": "completed",
        "config_sha256": config_sha256,
        "split": split,
        "fold": fold,
        "seed": seed,
        "updates": updates,
        "resumed_from_update": start_update,
        "selected_update": best_update,
        "best_validation_stage1_loss": best_validation_loss,
        "validation_split": validation_split,
        "validation_interval": validation_interval,
        "validation_history": validation_history,
        "batch_size": batch_size,
        "shuffle_buffer": shuffle_buffer,
        "device": str(device),
        "elapsed_seconds": elapsed_seconds,
        "training_samples_this_process": updates_this_process * batch_size,
        "training_samples_per_second": (
            updates_this_process * batch_size / elapsed_seconds if elapsed_seconds > 0 else None
        ),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "runtime_environment": {
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "determinism": determinism,
            "amp_mode": "disabled",
        },
        "mean_last_100_loss": float(np.mean(losses[-100:])),
        "checkpoint": str(best_checkpoint),
        "checkpoint_sha256": _sha256_file(best_checkpoint),
        "resume_checkpoint": str(last_checkpoint),
        "resume_checkpoint_sha256": _sha256_file(last_checkpoint),
        "joined_manifest": str(joined_path),
        "joined_manifest_sha256": joined_sha256,
        "semantic_store": str(store_root),
        "semantic_store_manifest_sha256": store_manifest_sha256,
        "trainer_source_sha256": trainer_source_sha256,
        "code_sha256": code_sha256,
    }
    write_json(output_root / "runtime_manifest.json", runtime)
    store.close()
    return runtime


def main() -> None:  # pragma: no cover - SSH GPU CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = train(load_json_config(args.config))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
