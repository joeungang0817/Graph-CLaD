"""Train one Phase 3C structured candidate against a frozen CLaD base."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from contextlib import nullcontext
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset import (
    NormalizationStats,
    SemanticFeatureStore,
    collate_phase3c,
    fit_normalization,
    iter_filtered_records,
    iter_shuffled_batches,
)
from .contracts import PRIMARY_RELATIONS, canonical_sha256
from .io import atomic_jsonl, load_json_config, write_json, write_json_line
from .losses import relation_motion_loss
from .metrics import evaluate_motion, evaluate_relation_predictions, no_change_fpr
from .models.adapters import Phase3CAdapter, SemanticPastActEncoder
from .models.semantic_clad import ControlledCLaD
from .models.structured import build_structured_model
from .parameter_match import trainable_parameter_count
from .provenance import runtime_provenance


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


def _code_sha256() -> str:
    repository = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__),
        repository / "scripts" / "phase3c" / "contracts.py",
        repository / "scripts" / "phase3c" / "dataset.py",
        repository / "scripts" / "phase3c" / "io.py",
        repository / "scripts" / "phase3c" / "losses.py",
        repository / "scripts" / "phase3c" / "metrics.py",
        repository / "scripts" / "phase3c" / "models" / "adapters.py",
        repository / "scripts" / "phase3c" / "models" / "semantic_clad.py",
        repository / "scripts" / "phase3c" / "models" / "structured.py",
        repository / "scripts" / "phase3c" / "parameter_match.py",
        repository / "scripts" / "phase3c" / "provenance.py",
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


def _amp_setup(device: torch.device, requested: bool):
    if not requested or device.type != "cuda":
        return "disabled", None
    if bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
        return "bf16", None
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    return "fp16", scaler


def _autocast_context(device: torch.device, amp_mode: str):
    if amp_mode == "disabled":
        return nullcontext()
    dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


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
    exclude_task_id: int | None = None,
    include_task_id: int | None = None,
):
    pending: list[dict[str, Any]] = []
    for record in iter_filtered_records(
        path,
        split=split,
        exclude_task_id=exclude_task_id,
        include_task_id=include_task_id,
    ):
        pending.append(record)
        if len(pending) == int(batch_size):
            yield pending
            pending = []
    if pending:
        yield pending


def _relation_pos_weights(
    path: Path,
    *,
    split: str,
    exclude_task_id: int | None,
    cap: float,
) -> tuple[list[float], dict[str, dict[str, int]], float]:
    positive = {name: 0 for name in PRIMARY_RELATIONS}
    negative = {name: 0 for name in PRIMARY_RELATIONS}
    motion_squared_sum = 0.0
    motion_count = 0
    for record in iter_filtered_records(
        path, split=split, exclude_task_id=exclude_task_id
    ):
        target = record.get("target")
        if not isinstance(target, dict):
            raise ValueError("joined record target must be an object")
        labels = target.get("relation_any_change")
        valid = target.get("relation_valid")
        if not isinstance(labels, dict) or not isinstance(valid, dict):
            raise ValueError("joined relation labels/masks must be objects")
        for name in PRIMARY_RELATIONS:
            if not bool(valid.get(name, 0)):
                continue
            if bool(labels.get(name, 0)):
                positive[name] += 1
            else:
                negative[name] += 1
        motion = float(target.get("scene_max_displacement_m"))
        if not np.isfinite(motion) or motion < 0.0:
            raise ValueError("training motion target must be finite and non-negative")
        motion_squared_sum += motion * motion
        motion_count += 1
    weights: list[float] = []
    counts: dict[str, dict[str, int]] = {}
    for name in PRIMARY_RELATIONS:
        counts[name] = {"positive": positive[name], "negative": negative[name]}
        weights.append(
            min(float(cap), negative[name] / positive[name])
            if positive[name] > 0 and negative[name] > 0
            else 1.0
        )
    if motion_count == 0:
        raise ValueError("training split has no motion targets")
    motion_scale = max(float(np.sqrt(motion_squared_sum / motion_count)), 1e-6)
    return weights, counts, motion_scale


def _relation_counts(
    path: Path, *, split: str, exclude_task_id: int | None
) -> dict[str, dict[str, int]]:
    counts = {
        name: {"positive": 0, "negative": 0} for name in PRIMARY_RELATIONS
    }
    for record in iter_filtered_records(
        path, split=split, exclude_task_id=exclude_task_id
    ):
        target = record.get("target")
        if not isinstance(target, dict):
            raise ValueError("joined record target must be an object")
        labels = target.get("relation_any_change")
        valid = target.get("relation_valid")
        if not isinstance(labels, dict) or not isinstance(valid, dict):
            raise ValueError("joined relation labels/masks must be objects")
        for name in PRIMARY_RELATIONS:
            if bool(valid.get(name, 0)):
                key = "positive" if bool(labels.get(name, 0)) else "negative"
                counts[name][key] += 1
    return counts


def _evaluate_adapter(
    *,
    adapter: Phase3CAdapter,
    clad: ControlledCLaD,
    joined_path: Path,
    store: SemanticFeatureStore,
    normalization: NormalizationStats,
    device: torch.device,
    batch_size: int,
    max_nodes: int | None,
    split: str,
    exclude_task_id: int | None = None,
    include_task_id: int | None = None,
    fixed_thresholds: Sequence[float] | None = None,
    prediction_path: Path | None = None,
    prediction_metadata: dict[str, Any] | None = None,
    motion_scale: float = 1.0,
    eligible_relations: Sequence[bool] | None = None,
) -> dict[str, Any]:
    if prediction_path is not None and fixed_thresholds is None:
        raise ValueError("persisted test predictions require validation-fixed thresholds")
    adapter.eval()
    relation_logits: list[np.ndarray] = []
    relation_targets: list[np.ndarray] = []
    relation_masks: list[np.ndarray] = []
    motions: list[float] = []
    target_motions: list[float] = []
    evaluated_rows = 0
    context = atomic_jsonl(prediction_path) if prediction_path is not None else nullcontext(None)
    with context as handle:
        for batch_records in _filtered_batches(
            joined_path,
            batch_size=batch_size,
            split=split,
            exclude_task_id=exclude_task_id,
            include_task_id=include_task_id,
        ):
            batch = collate_phase3c(
                batch_records,
                store,
                max_nodes=max_nodes,
                normalization=normalization,
                device=device,
            )
            with torch.no_grad():
                semantic = clad.encode_foresight(batch)
                prediction = adapter(batch, semantic)
            logits = prediction["relation_logits"].detach().cpu().numpy()
            predicted_motion = (
                prediction["scene_motion"] * float(motion_scale)
            ).detach().cpu().numpy().reshape(-1)
            target_relation = batch.target_relation_change.detach().cpu().numpy()
            raw_relation_mask = batch.target_relation_mask.detach().cpu().numpy()
            relation_mask = raw_relation_mask.copy()
            if eligible_relations is not None:
                relation_mask = relation_mask * np.asarray(
                    eligible_relations, dtype=np.float32
                )[None, :]
            target_motion = batch.target_scene_motion.detach().cpu().numpy().reshape(-1)
            relation_logits.append(logits)
            relation_targets.append(target_relation)
            relation_masks.append(relation_mask)
            motions.extend(predicted_motion.tolist())
            target_motions.extend(target_motion.tolist())
            evaluated_rows += len(batch_records)
            if handle is not None:
                for row, record in enumerate(batch_records):
                    for relation_index, relation in enumerate(PRIMARY_RELATIONS):
                        threshold = float(fixed_thresholds[relation_index])
                        logit = float(logits[row, relation_index])
                        probability = float(1.0 / (1.0 + np.exp(-np.clip(logit, -700.0, 700.0))))
                        eligible = (
                            True
                            if eligible_relations is None
                            else bool(eligible_relations[relation_index])
                        )
                        valid = bool(raw_relation_mask[row, relation_index])
                        payload = {
                            "schema": "phase3c-core-prediction.v2",
                            **(prediction_metadata or {}),
                            "sample_id": str(record["sample_id"]),
                            "task_id": int(record["task_id"]),
                            "episode_id": str(record["episode_id"]),
                            "prev_step": int(record["prev_step"]),
                            "current_step": int(record["current_step"]),
                            "target_step": int(record["target_step"]),
                            "tau": int(record.get("tau", 6)),
                            "relation": relation,
                            "eligible": eligible,
                            "valid": valid,
                            "evaluated": bool(eligible and valid),
                            "target": int(target_relation[row, relation_index]) if valid else None,
                            "logit": logit,
                            "probability": probability,
                            "threshold": threshold,
                            "prediction": int(probability >= threshold),
                            "scene_motion": float(predicted_motion[row]),
                            "target_scene_motion": float(target_motion[row]),
                        }
                        prevalence = (prediction_metadata or {}).get("train_prevalence")
                        if isinstance(prevalence, dict) and relation in prevalence:
                            payload["train_prevalence"] = float(prevalence[relation])
                        write_json_line(handle, payload)
    if not evaluated_rows:
        raise ValueError(
            f"joined manifest has no evaluation records for split={split}, "
            f"exclude_task_id={exclude_task_id}, include_task_id={include_task_id}"
        )
    logits_array = np.concatenate(relation_logits)
    targets_array = np.concatenate(relation_targets)
    masks_array = np.concatenate(relation_masks)
    relation_metrics = evaluate_relation_predictions(
        logits_array,
        targets_array,
        masks_array,
        fixed_thresholds=fixed_thresholds,
    )
    selected_thresholds = (
        list(fixed_thresholds)
        if fixed_thresholds is not None
        else [
            0.5
            if relation_metrics["per_relation"][name]["threshold"] is None
            else float(relation_metrics["per_relation"][name]["threshold"])
            for name in PRIMARY_RELATIONS
        ]
    )
    return {
        "status": "completed",
        "split": split,
        "rows": evaluated_rows,
        "relation": relation_metrics,
        "no_change_fpr": no_change_fpr(
            logits_array, targets_array, masks_array, selected_thresholds
        ),
        "motion": evaluate_motion(motions, target_motions),
        "prediction_path": str(prediction_path) if prediction_path is not None else None,
    }


def train(config: dict[str, Any]) -> dict[str, Any]:
    total_start = time.time()
    joined_path = _path(_get(config, "joined_manifest"))
    store_root = _path(_get(config, "semantic_store"))
    base_checkpoint = _path(_get(config, "base_checkpoint"))
    output_root = _path(_get(config, "output_root"))
    model_id = str(_get(config, "model_id", "C3-RelMPNN-PastAct"))
    fold = str(_get(config, "fold", "unspecified"))
    train_split = str(_get(config, "split", "train"))
    held_out_raw = _get(config, "held_out_task_id")
    held_out_task_id = int(held_out_raw) if held_out_raw is not None else None
    seed = int(_get(config, "seed", 0))
    updates = int(_get(config, "updates", 10_000))
    batch_size = int(_get(config, "batch_size", 64))
    shuffle_buffer = int(_get(config, "shuffle_buffer", max(2048, batch_size * 8)))
    hidden_dim = int(_get(config, "hidden_dim", 128))
    structured_dim = int(_get(config, "structured_dim", 256))
    vl_dim = int(_get(config, "vl_dim", 1024))
    relation_weight = float(_get(config, "relation_weight", 1.0))
    motion_weight = float(_get(config, "motion_weight", 0.1))
    pos_weight_cap = float(_get(config, "pos_weight_cap", 20.0))
    min_train_positive = int(_get(config, "min_train_positive", 20))
    min_train_negative = int(_get(config, "min_train_negative", 20))
    min_validation_positive = int(_get(config, "min_validation_positive", 5))
    min_validation_negative = int(_get(config, "min_validation_negative", 5))
    learning_rate = float(_get(config, "learning_rate", 3e-4))
    weight_decay = float(_get(config, "weight_decay", 1e-4))
    max_nodes_raw = _get(config, "max_nodes")
    max_nodes = int(max_nodes_raw) if max_nodes_raw is not None else None
    validation_interval = int(_get(config, "validation_interval", 500))
    early_stopping_patience = int(_get(config, "early_stopping_patience", 5))
    minimum_updates = int(_get(config, "minimum_updates", min(3_000, updates)))
    resume = bool(_get(config, "resume", True))
    amp_requested = bool(_get(config, "amp", True))
    device = torch.device(str(_get(config, "device", "cuda" if torch.cuda.is_available() else "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requested CUDA but CUDA is unavailable")
    if not base_checkpoint.exists():
        raise FileNotFoundError(base_checkpoint)
    determinism = _configure_determinism(seed)
    amp_mode, grad_scaler = _amp_setup(device, amp_requested)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if updates <= 0 or batch_size <= 0 or validation_interval <= 0:
        raise ValueError("updates, batch_size, and validation_interval must be positive")
    if shuffle_buffer < batch_size:
        raise ValueError("shuffle_buffer must be at least batch_size")
    if early_stopping_patience <= 0 or minimum_updates <= 0 or minimum_updates > updates:
        raise ValueError("invalid early-stopping patience/minimum_updates")
    if hidden_dim <= 0 or structured_dim <= 0 or (max_nodes is not None and max_nodes <= 0):
        raise ValueError("hidden_dim, structured_dim, and max_nodes must be positive")
    if relation_weight < 0.0 or motion_weight < 0.0 or pos_weight_cap <= 0.0:
        raise ValueError("loss weights and pos_weight_cap must be non-negative/positive")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if min(min_train_positive, min_train_negative, min_validation_positive, min_validation_negative) < 1:
        raise ValueError("relation support thresholds must be positive")
    provenance_start = runtime_provenance()
    if bool(_get(config, "require_clean_git", False)) and provenance_start.get(
        "git_dirty"
    ) is not False:
        raise RuntimeError(
            "formal Phase3C training requires a clean git checkout with readable provenance"
        )
    if not joined_path.exists():
        raise FileNotFoundError(joined_path)
    joined_sha256 = _sha256_file(joined_path)
    store = SemanticFeatureStore(store_root)
    semantic_store_integrity = store.verify_integrity(
        require_orientation_attestation=bool(
            _get(config, "require_orientation_attestation", True)
        )
    )
    semantic_store_integrity_sha256 = canonical_sha256(semantic_store_integrity)
    store_manifest_sha256 = _sha256_file(store_root / "manifest.json")
    store_source_sha = str(
        (store.manifest.get("source") or {}).get("joined_manifest_sha256", "")
    )
    if store_source_sha != joined_sha256:
        raise ValueError("semantic store was not built from the configured joined manifest")
    if store.feature_dim != vl_dim:
        raise ValueError(f"semantic store feature_dim={store.feature_dim} does not match vl_dim={vl_dim}")
    checkpoint = _torch_load(base_checkpoint, map_location="cpu")
    if (
        checkpoint.get("schema") != "phase3c-base-clad-checkpoint.v4"
        or checkpoint.get("kind") != "validation_best"
    ):
        raise ValueError(
            "core training requires a Phase3C v4 validation-best base checkpoint"
        )
    if str(checkpoint.get("source_joined_manifest_sha256", "")) != joined_sha256:
        raise ValueError("base checkpoint was not trained from the configured joined manifest")
    if str(checkpoint.get("semantic_store_manifest_sha256", "")) != store_manifest_sha256:
        raise ValueError("base checkpoint was not trained from the configured semantic store")
    if (
        str(checkpoint.get("semantic_store_integrity_sha256", ""))
        != semantic_store_integrity_sha256
    ):
        raise ValueError(
            "base checkpoint semantic-store integrity/QA attestation does not match"
        )
    normalization = (
        NormalizationStats.from_dict(checkpoint["normalization"])
        if checkpoint.get("normalization")
        else fit_normalization(
            iter_filtered_records(
                joined_path, split=train_split, exclude_task_id=held_out_task_id
            )
        )
    )
    checkpoint_vl = int(checkpoint.get("vl_dim", vl_dim))
    checkpoint_hidden = int(checkpoint.get("hidden_dim", vl_dim))
    if checkpoint_vl != vl_dim or checkpoint_vl != store.feature_dim:
        raise ValueError(
            f"base checkpoint vl_dim={checkpoint_vl}, config vl_dim={vl_dim}, "
            f"store feature_dim={store.feature_dim} must match"
        )
    clad = ControlledCLaD(proprio_dim=16, vl_dim=checkpoint_vl, hidden_dim=checkpoint_hidden, action_dim=42).to(device)
    clad.load_state_dict(checkpoint["model_state"])
    clad.eval()
    for parameter in clad.parameters():
        parameter.requires_grad_(False)
    if model_id == "C3-Sem-PastAct":
        structured = SemanticPastActEncoder(
            structured_dim, hidden_dim=hidden_dim
        ).to(device)
    else:
        structured = build_structured_model(model_id, hidden_dim=hidden_dim, output_dim=structured_dim).to(device)
    adapter = Phase3CAdapter(structured, semantic_dim=2 * checkpoint_hidden, structured_dim=structured_dim).to(device)
    trainable_parameters = trainable_parameter_count(adapter)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate, weight_decay=weight_decay)
    config_sha256 = canonical_sha256(config)
    trainer_source_sha256 = _sha256_file(Path(__file__))
    code_sha256 = _code_sha256()
    base_checkpoint_sha256 = _sha256_file(base_checkpoint)
    pos_weight_values, train_relation_support, motion_scale = _relation_pos_weights(
        joined_path,
        split=train_split,
        exclude_task_id=held_out_task_id,
        cap=pos_weight_cap,
    )
    validation_split = str(_get(config, "validation_split", "validation"))
    validation_relation_support = _relation_counts(
        joined_path,
        split=validation_split,
        exclude_task_id=held_out_task_id,
    )
    eligible_relations = [
        train_relation_support[name]["positive"] >= min_train_positive
        and train_relation_support[name]["negative"] >= min_train_negative
        and validation_relation_support[name]["positive"] >= min_validation_positive
        and validation_relation_support[name]["negative"] >= min_validation_negative
        for name in PRIMARY_RELATIONS
    ]
    if sum(eligible_relations) < 2:
        raise ValueError(
            "fewer than two relations satisfy the frozen train/validation support gate"
        )
    pos_weight = torch.tensor(pos_weight_values, dtype=torch.float32, device=device)
    relation_eligibility = torch.tensor(
        eligible_relations, dtype=torch.bool, device=device
    )
    start = time.time()
    losses: list[float] = []
    validation_history: list[dict[str, Any]] = []
    best_validation_pr_auc = -float("inf")
    best_update = 0
    stale_validations = 0
    completed_updates = 0
    start_update = 0
    resume_rng: dict[str, Any] | None = None
    checkpoint_root = output_root / "checkpoints"
    checkpoint_path = checkpoint_root / "best.pt"
    last_checkpoint_path = checkpoint_root / "last.pt"
    if resume and last_checkpoint_path.exists():
        resumed = _torch_load(last_checkpoint_path, map_location="cpu")
        expected = {
            "config_sha256": config_sha256,
            "trainer_source_sha256": trainer_source_sha256,
            "code_sha256": code_sha256,
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "source_joined_manifest_sha256": joined_sha256,
            "semantic_store_manifest_sha256": store_manifest_sha256,
            "semantic_store_integrity_sha256": semantic_store_integrity_sha256,
            "model_id": model_id,
            "fold": fold,
            "seed": seed,
        }
        for name, value in expected.items():
            if resumed.get(name) != value:
                raise ValueError(f"core resume checkpoint {name} mismatch")
        if resumed.get("amp_mode") != amp_mode:
            raise ValueError("core resume checkpoint amp_mode mismatch")
        start_update = int(resumed.get("update", 0))
        if start_update < 0 or start_update > updates:
            raise ValueError("core resume checkpoint update is outside requested budget")
        adapter.load_state_dict(resumed["model_state"])
        optimizer.load_state_dict(resumed["optimizer_state"])
        if grad_scaler is not None:
            scaler_state = resumed.get("grad_scaler_state")
            if not isinstance(scaler_state, dict):
                raise ValueError("fp16 resume checkpoint is missing grad_scaler_state")
            grad_scaler.load_state_dict(scaler_state)
        best_validation_pr_auc = float(
            resumed.get("best_validation_macro_pr_auc", -float("inf"))
        )
        best_update = int(resumed.get("best_update", 0))
        stale_validations = int(resumed.get("stale_validations", 0))
        validation_history = list(resumed.get("validation_history", []))
        losses = [float(value) for value in resumed.get("recent_losses", [])]
        resume_rng = resumed.get("rng_state")
        if best_update > 0 and not checkpoint_path.exists():
            raise FileNotFoundError(
                "core resume checkpoint references a missing validation-best checkpoint"
            )
        completed_updates = start_update
    adapter.train()
    training_batches = iter_shuffled_batches(
        joined_path,
        batch_size=batch_size,
        seed=seed,
        split=train_split,
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
        with _autocast_context(device, amp_mode):
            with torch.no_grad():
                semantic = clad.encode_foresight(batch)
            prediction = adapter(batch, semantic)
            losses_dict = relation_motion_loss(
                prediction,
                batch,
                relation_weight=relation_weight,
                motion_weight=motion_weight,
                motion_scale=motion_scale,
                pos_weight=pos_weight,
                relation_eligibility=relation_eligibility,
            )
        total = losses_dict["loss"]
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite core loss at update {update}")
        if grad_scaler is None:
            total.backward()
        else:
            grad_scaler.scale(total).backward()
            grad_scaler.unscale_(optimizer)
        if update == start_update + 1:
            missing_gradients = [
                name
                for name, parameter in adapter.named_parameters()
                if parameter.requires_grad and parameter.grad is None
            ]
            if missing_gradients:
                raise RuntimeError(
                    "core adapter contains trainable parameters unused by the loss: "
                    + ", ".join(missing_gradients[:20])
                )
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        if grad_scaler is None:
            optimizer.step()
        else:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        losses.append(float(total.detach().cpu()))
        completed_updates = update
        should_validate = update % validation_interval == 0 or update == updates
        if should_validate:
            training_rng = _capture_rng_state()
            validation_snapshot = _evaluate_adapter(
                adapter=adapter,
                clad=clad,
                joined_path=joined_path,
                store=store,
                normalization=normalization,
                device=device,
                batch_size=batch_size,
                max_nodes=max_nodes,
                split=validation_split,
                exclude_task_id=held_out_task_id,
                motion_scale=motion_scale,
                eligible_relations=eligible_relations,
            )
            _restore_rng_state(training_rng)
            score = validation_snapshot["relation"]["macro_pr_auc"]
            if score is None:
                raise ValueError("validation macro PR-AUC is unavailable after support gating")
            validation_history.append({
                "update": update,
                "macro_pr_auc": float(score),
                "family_macro_pr_auc": validation_snapshot["relation"]["family_macro_pr_auc"],
                "macro_f1": validation_snapshot["relation"]["macro_f1"],
                "family_macro_f1": validation_snapshot["relation"]["family_macro_f1"],
                "motion_mae": validation_snapshot["motion"]["mae"],
            })
            if float(score) > best_validation_pr_auc:
                best_validation_pr_auc = float(score)
                best_update = update
                stale_validations = 0
                _atomic_torch_save(
                    checkpoint_path,
                    {
                        "schema": "phase3c-core-checkpoint.v4",
                        "kind": "validation_best",
                        "model_id": model_id,
                        "fold": fold,
                        "model_state": adapter.state_dict(),
                        "requested_updates": updates,
                        "selected_update": best_update,
                        "best_validation_macro_pr_auc": best_validation_pr_auc,
                        "seed": seed,
                        "vl_dim": checkpoint_vl,
                        "hidden_dim": checkpoint_hidden,
                        "structured_hidden_dim": hidden_dim,
                        "structured_dim": structured_dim,
                        "base_checkpoint_sha256": base_checkpoint_sha256,
                        "source_joined_manifest_sha256": joined_sha256,
                        "semantic_store_manifest_sha256": store_manifest_sha256,
                        "semantic_store_integrity_sha256": semantic_store_integrity_sha256,
                        "code_sha256": code_sha256,
                        "trainable_parameters": trainable_parameters,
                        "motion_scale_m": motion_scale,
                        "eligible_relations": {
                            name: bool(value)
                            for name, value in zip(PRIMARY_RELATIONS, eligible_relations)
                        },
                        "relation_pos_weight": dict(zip(PRIMARY_RELATIONS, pos_weight_values)),
                    },
                )
            else:
                stale_validations += 1
            _atomic_torch_save(
                last_checkpoint_path,
                {
                    "schema": "phase3c-core-resume.v4",
                    "kind": "resume_last",
                    "model_id": model_id,
                    "fold": fold,
                    "seed": seed,
                    "model_state": adapter.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "amp_mode": amp_mode,
                    "grad_scaler_state": grad_scaler.state_dict() if grad_scaler is not None else None,
                    "update": update,
                    "best_update": best_update,
                    "best_validation_macro_pr_auc": best_validation_pr_auc,
                    "stale_validations": stale_validations,
                    "validation_history": validation_history,
                    "recent_losses": losses[-100:],
                    "rng_state": _capture_rng_state(),
                    "config_sha256": config_sha256,
                    "trainer_source_sha256": trainer_source_sha256,
                    "code_sha256": code_sha256,
                    "base_checkpoint_sha256": base_checkpoint_sha256,
                    "source_joined_manifest_sha256": joined_sha256,
                    "semantic_store_manifest_sha256": store_manifest_sha256,
                    "semantic_store_integrity_sha256": semantic_store_integrity_sha256,
                },
            )
            adapter.train()
            if update >= minimum_updates and stale_validations >= early_stopping_patience:
                break
    if not checkpoint_path.exists() or best_update <= 0:
        raise RuntimeError("core training completed without a validation checkpoint")
    best_checkpoint = _torch_load(checkpoint_path, map_location="cpu")
    adapter.load_state_dict(best_checkpoint["model_state"])
    prediction_path = output_root / "predictions" / "evaluation.jsonl.gz"
    metrics_path = output_root / "metrics.json"
    evaluation_split = str(_get(config, "evaluation_split", "test"))
    validation_metrics = _evaluate_adapter(
        adapter=adapter,
        clad=clad,
        joined_path=joined_path,
        store=store,
        normalization=normalization,
        device=device,
        batch_size=batch_size,
        max_nodes=max_nodes,
        split=validation_split,
        exclude_task_id=held_out_task_id,
        motion_scale=motion_scale,
        eligible_relations=eligible_relations,
    )
    validation_thresholds = []
    for relation in PRIMARY_RELATIONS:
        value = validation_metrics["relation"]["per_relation"][relation]["threshold"]
        validation_thresholds.append(0.5 if value is None else float(value))
    train_prevalence = {
        name: (
            train_relation_support[name]["positive"]
            / max(
                1,
                train_relation_support[name]["positive"]
                + train_relation_support[name]["negative"],
            )
        )
        for name in PRIMARY_RELATIONS
    }
    metrics = _evaluate_adapter(
        adapter=adapter,
        clad=clad,
        joined_path=joined_path,
        store=store,
        normalization=normalization,
        device=device,
        batch_size=batch_size,
        max_nodes=max_nodes,
        split=evaluation_split,
        include_task_id=held_out_task_id,
        fixed_thresholds=validation_thresholds,
        prediction_path=prediction_path,
        prediction_metadata={
            "model_id": model_id,
            "fold": fold,
            "seed": seed,
            "train_prevalence": train_prevalence,
        },
        motion_scale=motion_scale,
        eligible_relations=eligible_relations,
    )
    metrics["threshold_selection"] = {
        "split": validation_split,
        "excluded_task_id": held_out_task_id,
        "thresholds": dict(zip(PRIMARY_RELATIONS, validation_thresholds)),
    }
    write_json(metrics_path, metrics)
    elapsed_seconds = time.time() - start
    total_elapsed_seconds = time.time() - total_start
    updates_this_process = max(0, completed_updates - start_update)
    runtime = {
        "schema": "phase3c-core-run.v4",
        "status": "completed",
        "protocol": config.get("protocol"),
        "claim_scope": config.get("claim_scope"),
        "config_sha256": config_sha256,
        "model_id": model_id,
        "fold": fold,
        "split": train_split,
        "seed": seed,
        "updates": updates,
        "completed_updates": completed_updates,
        "resumed_from_update": start_update,
        "selected_update": best_update,
        "best_validation_macro_pr_auc": best_validation_pr_auc,
        "validation_interval": validation_interval,
        "early_stopping_patience": early_stopping_patience,
        "minimum_updates": minimum_updates,
        "validation_history": validation_history,
        "batch_size": batch_size,
        "trainable_parameters": trainable_parameters,
        "shuffle_buffer": shuffle_buffer,
        "device": str(device),
        "elapsed_seconds": elapsed_seconds,
        "total_elapsed_seconds": total_elapsed_seconds,
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
            "amp_requested": amp_requested,
            "amp_mode": amp_mode,
            "provenance_start": provenance_start,
            "provenance_end": runtime_provenance(),
        },
        "mean_last_100_loss": float(np.mean(losses[-100:])),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "resume_checkpoint": str(last_checkpoint_path),
        "resume_checkpoint_sha256": _sha256_file(last_checkpoint_path),
        "metrics": str(metrics_path),
        "metrics_sha256": _sha256_file(metrics_path),
        "predictions": str(prediction_path),
        "predictions_sha256": _sha256_file(prediction_path),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "joined_manifest_sha256": joined_sha256,
        "semantic_store_manifest_sha256": store_manifest_sha256,
        "semantic_store_integrity": semantic_store_integrity,
        "semantic_store_integrity_sha256": semantic_store_integrity_sha256,
        "normalization": normalization.to_dict(),
        "train_relation_support": train_relation_support,
        "validation_relation_support": validation_relation_support,
        "eligible_relations": {
            name: bool(value) for name, value in zip(PRIMARY_RELATIONS, eligible_relations)
        },
        "relation_pos_weight": dict(zip(PRIMARY_RELATIONS, pos_weight_values)),
        "motion_scale_m": motion_scale,
        "validation": validation_metrics,
        "evaluation": metrics,
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
