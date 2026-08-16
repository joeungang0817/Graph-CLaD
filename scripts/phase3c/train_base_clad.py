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

from .dataset import SemanticFeatureStore, collate_phase3c, fit_normalization
from .io import iter_json_objects, load_json_config, write_json
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


def _path(value: Any) -> Path:
    raw = os.path.expandvars(str(value))
    if "$" in raw:
        raise ValueError(f"unresolved environment variable in path: {value}")
    return Path(raw).expanduser()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _records(path: Path, split: str, *, held_out_task_id: int | None = None) -> list[dict[str, Any]]:
    records = []
    for item in iter_json_objects(path):
        if str(item.get("split")) == str(split) and (
            held_out_task_id is None or int(item.get("task_id", -1)) != held_out_task_id
        ):
            records.append(dict(item))
    if not records:
        raise ValueError(f"joined manifest has no records for split={split}")
    return records


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


def train(config: dict[str, Any]) -> dict[str, Any]:
    joined_path = _path(_get(config, "joined_manifest"))
    store_root = _path(_get(config, "semantic_store"))
    output_root = _path(_get(config, "output_root"))
    split = str(_get(config, "split", "train"))
    held_out_raw = _get(config, "held_out_task_id")
    held_out_task_id = int(held_out_raw) if held_out_raw is not None else None
    seed = int(_get(config, "seed", 0))
    updates = int(_get(config, "updates", 25_000))
    batch_size = int(_get(config, "batch_size", 128))
    hidden_dim = int(_get(config, "hidden_dim", 1024))
    vl_dim = int(_get(config, "vl_dim", hidden_dim))
    max_nodes = _get(config, "max_nodes")
    max_nodes = int(max_nodes) if max_nodes is not None else None
    learning_rate = float(_get(config, "learning_rate", 1e-4))
    weight_decay = float(_get(config, "weight_decay", 1e-4))
    recon_weight = float(_get(config, "recon_weight", 0.1))
    device = torch.device(str(_get(config, "device", "cuda" if torch.cuda.is_available() else "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requested CUDA but CUDA is unavailable")
    if updates <= 0 or batch_size <= 0:
        raise ValueError("updates and batch_size must be positive")
    _seed(seed)
    records = _records(joined_path, split, held_out_task_id=held_out_task_id)
    normalization = fit_normalization(records)
    store = SemanticFeatureStore(store_root)
    if store.feature_dim != vl_dim:
        raise ValueError(f"semantic store feature_dim={store.feature_dim} does not match vl_dim={vl_dim}")
    model = ControlledCLaD(proprio_dim=16, vl_dim=vl_dim, hidden_dim=hidden_dim, action_dim=42).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()
    start = time.time()
    losses: list[float] = []
    for update in range(1, updates + 1):
        indices = [(update - 1) * batch_size + offset for offset in range(batch_size)]
        batch_records = [records[index % len(records)] for index in indices]
        batch = collate_phase3c(batch_records, store, max_nodes=max_nodes, normalization=normalization, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.training_loss(batch)
        total = loss_dict["loss_p"] + loss_dict["loss_s"] + recon_weight * (loss_dict["loss_p_recon"] + loss_dict["loss_v_recon"])
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite base CLaD loss at update {update}")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_ema_after_optimizer_step()
        losses.append(float(total.detach().cpu()))
    checkpoint = output_root / "checkpoints" / "last.pt"
    payload = {
        "schema": "phase3c-base-clad-checkpoint.v1",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "update": updates,
        "seed": seed,
        "vl_dim": vl_dim,
        "hidden_dim": hidden_dim,
        "action_dim": 42,
        "proprio_dim": 16,
        "source_joined_manifest_sha256": _sha256_file(joined_path),
        "normalization": normalization.to_dict(),
    }
    _atomic_torch_save(checkpoint, payload)
    runtime = {
        "schema": "phase3c-base-clad-run.v1",
        "status": "completed",
        "split": split,
        "seed": seed,
        "updates": updates,
        "batch_size": batch_size,
        "device": str(device),
        "elapsed_seconds": time.time() - start,
        "mean_last_100_loss": float(np.mean(losses[-100:])),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "joined_manifest": str(joined_path),
        "semantic_store": str(store_root),
    }
    write_json(output_root / "runtime_manifest.json", runtime)
    return runtime


def main() -> None:  # pragma: no cover - SSH GPU CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = train(load_json_config(args.config))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
