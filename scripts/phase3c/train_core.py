"""Train one Phase 3C structured candidate against a frozen CLaD base."""

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

from .dataset import NormalizationStats, SemanticFeatureStore, collate_phase3c, fit_normalization
from .io import atomic_jsonl, iter_json_objects, load_json_config, write_json, write_json_line
from .losses import relation_motion_loss
from .metrics import evaluate_motion, evaluate_relation_predictions
from .models.adapters import Phase3CAdapter, ZeroStructuredEncoder
from .models.semantic_clad import ControlledCLaD
from .models.structured import build_structured_model


def _get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in config:
        return config[key]
    for section in ("core", "phase3c", "training"):
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
    records = [
        dict(item) for item in iter_json_objects(path)
        if str(item.get("split")) == split
        and (held_out_task_id is None or int(item.get("task_id", -1)) != held_out_task_id)
    ]
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
    base_checkpoint = _path(_get(config, "base_checkpoint"))
    output_root = _path(_get(config, "output_root"))
    model_id = str(_get(config, "model_id", "C3-RelMPNN-PastAct"))
    train_split = str(_get(config, "split", "train"))
    held_out_raw = _get(config, "held_out_task_id")
    held_out_task_id = int(held_out_raw) if held_out_raw is not None else None
    seed = int(_get(config, "seed", 0))
    updates = int(_get(config, "updates", 3_000))
    batch_size = int(_get(config, "batch_size", 64))
    hidden_dim = int(_get(config, "hidden_dim", 128))
    structured_dim = int(_get(config, "structured_dim", 256))
    vl_dim = int(_get(config, "vl_dim", 1024))
    relation_weight = float(_get(config, "relation_weight", 1.0))
    motion_weight = float(_get(config, "motion_weight", 0.1))
    learning_rate = float(_get(config, "learning_rate", 1e-4))
    weight_decay = float(_get(config, "weight_decay", 1e-4))
    max_nodes_raw = _get(config, "max_nodes")
    max_nodes = int(max_nodes_raw) if max_nodes_raw is not None else None
    device = torch.device(str(_get(config, "device", "cuda" if torch.cuda.is_available() else "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requested CUDA but CUDA is unavailable")
    if not base_checkpoint.exists():
        raise FileNotFoundError(base_checkpoint)
    _seed(seed)
    records = _records(joined_path, train_split, held_out_task_id=held_out_task_id)
    store = SemanticFeatureStore(store_root)
    if store.feature_dim != vl_dim:
        raise ValueError(f"semantic store feature_dim={store.feature_dim} does not match vl_dim={vl_dim}")
    checkpoint = torch.load(base_checkpoint, map_location="cpu")
    normalization = NormalizationStats.from_dict(checkpoint["normalization"]) if checkpoint.get("normalization") else fit_normalization(records)
    checkpoint_vl = int(checkpoint.get("vl_dim", vl_dim))
    checkpoint_hidden = int(checkpoint.get("hidden_dim", vl_dim))
    clad = ControlledCLaD(proprio_dim=16, vl_dim=checkpoint_vl, hidden_dim=checkpoint_hidden, action_dim=42).to(device)
    clad.load_state_dict(checkpoint["model_state"])
    clad.eval()
    for parameter in clad.parameters():
        parameter.requires_grad_(False)
    if model_id == "C3-Sem-PastAct":
        structured = ZeroStructuredEncoder(structured_dim).to(device)
    else:
        structured = build_structured_model(model_id, hidden_dim=hidden_dim, output_dim=structured_dim).to(device)
    adapter = Phase3CAdapter(structured, semantic_dim=2 * checkpoint_hidden, structured_dim=structured_dim).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate, weight_decay=weight_decay)
    start = time.time()
    losses: list[float] = []
    adapter.train()
    for update in range(1, updates + 1):
        batch_records = [records[((update - 1) * batch_size + offset) % len(records)] for offset in range(batch_size)]
        batch = collate_phase3c(batch_records, store, max_nodes=max_nodes, normalization=normalization, device=device)
        with torch.no_grad():
            semantic = clad.encode_foresight(batch)
        optimizer.zero_grad(set_to_none=True)
        prediction = adapter(batch, semantic)
        losses_dict = relation_motion_loss(prediction, batch, relation_weight=relation_weight, motion_weight=motion_weight)
        total = losses_dict["loss"]
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite core loss at update {update}")
        total.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        losses.append(float(total.detach().cpu()))
    checkpoint_path = output_root / "checkpoints" / "last.pt"
    _atomic_torch_save(
        checkpoint_path,
        {
            "schema": "phase3c-core-checkpoint.v1",
            "model_id": model_id,
            "model_state": adapter.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "update": updates,
            "seed": seed,
            "vl_dim": checkpoint_vl,
            "hidden_dim": checkpoint_hidden,
            "structured_hidden_dim": hidden_dim,
            "structured_dim": structured_dim,
            "base_checkpoint_sha256": _sha256_file(base_checkpoint),
        },
    )
    prediction_path = output_root / "predictions" / "evaluation.jsonl.gz"
    metrics_path = output_root / "metrics.json"
    evaluation_split = str(_get(config, "evaluation_split", "test"))
    evaluation_records: list[dict[str, Any]] = []
    for item in iter_json_objects(joined_path):
        if str(item.get("split")) == evaluation_split and (
            held_out_task_id is None or int(item.get("task_id", -1)) == held_out_task_id
        ):
            evaluation_records.append(dict(item))
    metrics: dict[str, Any] = {"status": "skipped", "split": evaluation_split}
    if evaluation_records:
        adapter.eval()
        relation_logits: list[np.ndarray] = []
        relation_targets: list[np.ndarray] = []
        relation_masks: list[np.ndarray] = []
        motions: list[float] = []
        target_motions: list[float] = []
        with atomic_jsonl(prediction_path) as handle:
            for start_index in range(0, len(evaluation_records), batch_size):
                batch_records = evaluation_records[start_index : start_index + batch_size]
                batch = collate_phase3c(batch_records, store, max_nodes=max_nodes, normalization=normalization, device=device)
                with torch.no_grad():
                    semantic = clad.encode_foresight(batch)
                    prediction = adapter(batch, semantic)
                logits = prediction["relation_logits"].detach().cpu().numpy()
                predicted_motion = prediction["scene_motion"].detach().cpu().numpy().reshape(-1)
                target_relation = batch.target_relation_change.detach().cpu().numpy()
                relation_mask = batch.target_relation_mask.detach().cpu().numpy()
                target_motion = batch.target_scene_motion.detach().cpu().numpy().reshape(-1)
                relation_logits.append(logits)
                relation_targets.append(target_relation)
                relation_masks.append(relation_mask)
                motions.extend(predicted_motion.tolist())
                target_motions.extend(target_motion.tolist())
                for row, record in enumerate(batch_records):
                    write_json_line(handle, {
                        "sample_id": str(record["sample_id"]), "task_id": int(record["task_id"]),
                        "episode_id": str(record["episode_id"]), "relation_logits": logits[row].tolist(),
                        "scene_motion": float(predicted_motion[row]), "target_relation_change": target_relation[row].tolist(),
                        "target_relation_mask": relation_mask[row].tolist(), "target_scene_motion": float(target_motion[row]),
                    })
        metrics = {
            "status": "completed", "split": evaluation_split,
            "relation": evaluate_relation_predictions(np.concatenate(relation_logits), np.concatenate(relation_targets), np.concatenate(relation_masks)),
            "motion": evaluate_motion(motions, target_motions),
            "prediction_path": str(prediction_path),
        }
        write_json(metrics_path, metrics)
    runtime = {
        "schema": "phase3c-core-run.v1",
        "status": "completed",
        "model_id": model_id,
        "split": train_split,
        "seed": seed,
        "updates": updates,
        "batch_size": batch_size,
        "device": str(device),
        "elapsed_seconds": time.time() - start,
        "mean_last_100_loss": float(np.mean(losses[-100:])),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "base_checkpoint": str(base_checkpoint),
            "base_checkpoint_sha256": _sha256_file(base_checkpoint),
            "normalization": normalization.to_dict(),
        "evaluation": metrics,
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
