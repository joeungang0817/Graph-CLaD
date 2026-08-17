"""Train a small controlled DDPM policy for the Stage 2 one-task pilot.

This is a controlled reimplementation for a preliminary smoke/pilot.  It is
not the official CLaD Stage 2 trainer and it does not perform LIBERO rollout.
The semantic and graph arms share the same policy and diffusion schedule; the
graph arm only changes the frozen/learned foresight adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from scripts.phase3c.contracts import parse_action_window
from scripts.phase3c.dataset import (
    GraphBatch,
    NormalizationStats,
    SemanticFeatureStore,
    graph_tensors,
)
from scripts.phase3c.io import iter_json_objects, load_json_config, write_json
from scripts.phase3c.models.adapters import Phase3CAdapter
from scripts.phase3c.models.semantic_clad import ControlledCLaD
from scripts.phase3c.models.structured import build_structured_model
from scripts.phase4.models.foresight_adapter import (
    ResidualGraphForesightAdapter,
    SemanticForesightInterface,
)


ACTION_DIM = 42
OBS_DIM = 1024 + 1024 + 16
FORESIGHT_DIM = 2048


def _path(value: Any) -> Path:
    raw = os.path.expandvars(str(value))
    if "$" in raw:
        raise ValueError(f"unresolved environment variable in path: {value}")
    return Path(raw).expanduser()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_rows(path: Path, split: str, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in iter_json_objects(path):
        if value.get("schema") != "phase5-policy-sample.v1":
            raise ValueError(f"unsupported Stage 2 policy row schema: {value.get('schema')}")
        if str(value.get("split")) != str(split):
            continue
        rows.append(dict(value))
        if limit is not None and len(rows) >= int(limit):
            break
    if not rows:
        raise ValueError(f"policy manifest contains no rows for split={split}")
    return rows


def _stack_graph(graphs: Sequence[GraphBatch]) -> GraphBatch:
    if not graphs:
        raise ValueError("cannot stack an empty graph batch")
    return GraphBatch(
        *(torch.stack([getattr(graph, name) for graph in graphs]) for name in GraphBatch.__dataclass_fields__)
    )


def _move_graph(graph: GraphBatch, device: torch.device) -> GraphBatch:
    """Move every tensor in the frozen graph dataclass explicitly.

    ``GraphBatch`` is intentionally a small immutable dataclass and does not
    expose a PyTorch-style ``.to`` method.  Keeping the move explicit prevents
    a future metadata field from being silently dropped.
    """

    return GraphBatch(
        *(getattr(graph, name).to(device) for name in GraphBatch.__dataclass_fields__)
    )


def _causal_batch(
    rows: Sequence[Mapping[str, Any]],
    store: SemanticFeatureStore,
    normalization: NormalizationStats,
    *,
    max_nodes: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    graph_prev: list[GraphBatch] = []
    graph_current: list[GraphBatch] = []
    v_history: list[torch.Tensor] = []
    p_history: list[torch.Tensor] = []
    past_actions: list[torch.Tensor] = []
    language: list[torch.Tensor] = []
    action_targets: list[torch.Tensor] = []
    for row in rows:
        tau = int(row.get("tau", 6))
        if tau != 6:
            raise ValueError(f"Stage 2 pilot requires tau=6, got {tau}")
        previous, current, prev_p, current_p = graph_tensors(
            row["graph_prev"], row["graph_t"], max_nodes=max_nodes
        )
        graph_prev.append(normalization.apply_graph(previous))
        graph_current.append(normalization.apply_graph(current))
        task_id = int(row["task_id"])
        demo_key = str(row["demo_key"])
        prev_step = int(row["prev_step"])
        current_step = int(row["current_step"])
        views = []
        for step in (prev_step, current_step):
            views.append(
                torch.stack(
                    [
                        torch.from_numpy(store.image(task_id, demo_key, step, view))
                        for view in (0, 1)
                    ]
                )
            )
        v_history.append(torch.stack(views))
        p_history.append(
            normalization.apply_proprio(
                torch.tensor([prev_p, current_p], dtype=torch.float32)
            )
        )
        past_actions.append(
            torch.tensor(
                parse_action_window(row["past_action_window"], tau=6),
                dtype=torch.float32,
            )
        )
        language.append(torch.from_numpy(store.language(task_id, demo_key)))
        action_targets.append(
            torch.tensor(
                parse_action_window(row["action_target_window"], tau=6),
                dtype=torch.float32,
            ).flatten()
        )
    return {
        "v_history": torch.stack(v_history).to(device),
        "p_history": torch.stack(p_history).to(device),
        "past_action": torch.stack(past_actions).to(device),
        "language": torch.stack(language).to(device),
        "graph_prev": _move_graph(_stack_graph(graph_prev), device),
        "graph_current": _move_graph(_stack_graph(graph_current), device),
        "action_target": torch.stack(action_targets).to(device),
    }


def _load_base(path: Path, device: torch.device) -> tuple[ControlledCLaD, NormalizationStats, dict[str, Any]]:
    payload = _torch_load(path)
    clad = ControlledCLaD(
        proprio_dim=16,
        vl_dim=int(payload.get("vl_dim", 1024)),
        hidden_dim=int(payload.get("hidden_dim", 1024)),
        action_dim=42,
    )
    clad.load_state_dict(payload["model_state"])
    clad.to(device).eval()
    for parameter in clad.parameters():
        parameter.requires_grad_(False)
    normalization = NormalizationStats.from_dict(payload["normalization"])
    return clad, normalization, payload


def _load_graph_adapter(
    path: Path,
    *,
    model_id: str,
    device: torch.device,
) -> ResidualGraphForesightAdapter:
    payload = _torch_load(path)
    if str(payload.get("model_id")) != str(model_id):
        raise ValueError(
            f"structured checkpoint model mismatch: expected {model_id}, got {payload.get('model_id')}"
        )
    structured_dim = int(payload.get("structured_dim", 256))
    width = int(payload.get("structured_hidden_dim", 128))
    vl_dim = int(payload.get("vl_dim", 1024))
    adapter = Phase3CAdapter(
        build_structured_model(
            model_id, hidden_dim=width, output_dim=structured_dim
        ),
        semantic_dim=2 * vl_dim,
        structured_dim=structured_dim,
    )
    adapter.load_state_dict(payload["model_state"])
    residual = ResidualGraphForesightAdapter(
        adapter.structured_encoder,
        structured_dim=structured_dim,
        hidden_dim=vl_dim,
    ).to(device)
    for parameter in residual.structured_encoder.parameters():
        parameter.requires_grad_(False)
    return residual


def _timestep_embedding(timestep: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    scale = math.log(10000.0) / max(half - 1, 1)
    values = torch.exp(torch.arange(half, device=timestep.device) * -scale)
    values = timestep.float().unsqueeze(1) * values.unsqueeze(0)
    embedding = torch.cat([torch.sin(values), torch.cos(values)], dim=1)
    if dimension % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


class TinyFiLMDiffusionPolicy(nn.Module):
    """Small, explicit epsilon-prediction policy used only for the pilot."""

    def __init__(self, *, hidden_dim: int = 256, diffusion_steps: int = 1000):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.diffusion_steps = int(diffusion_steps)
        self.observation = nn.Sequential(nn.Linear(OBS_DIM, hidden_dim), nn.GELU())
        self.foresight = nn.Sequential(nn.Linear(FORESIGHT_DIM, hidden_dim), nn.GELU())
        self.time = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.action = nn.Linear(ACTION_DIM, hidden_dim)
        self.film = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, ACTION_DIM),
        )
        betas = torch.linspace(1e-4, 2e-2, diffusion_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        observation: torch.Tensor,
        foresight: torch.Tensor,
    ) -> torch.Tensor:
        condition = (
            self.observation(observation)
            + self.foresight(foresight)
            + self.time(_timestep_embedding(timestep, self.hidden_dim))
        )
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        hidden = self.action(noisy_action)
        hidden = hidden * (1.0 + gamma) + beta
        return self.head(hidden)

    def noisy_action(
        self,
        action: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self.alphas_cumprod[timestep].unsqueeze(1)
        return alpha.sqrt() * action + (1.0 - alpha).sqrt() * noise


def _condition_inputs(
    batch: Mapping[str, torch.Tensor],
    clad: ControlledCLaD,
    foresight_adapter: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        semantic_foresight = clad.encode_foresight(batch)
    if isinstance(foresight_adapter, SemanticForesightInterface):
        adapted = foresight_adapter(semantic_foresight)
    else:
        adapted = foresight_adapter(batch, semantic_foresight)
    observation = torch.cat(
        [
            batch["v_history"][:, 1].mean(dim=1),
            batch["language"],
            batch["p_history"][:, 1],
        ],
        dim=1,
    ).float()
    foresight = adapted["foresight"].float()
    if observation.shape[1] != OBS_DIM or foresight.shape[1] != FORESIGHT_DIM:
        raise ValueError("Stage 2 condition dimensions do not match the policy contract")
    return observation, foresight


def _action_stats(rows: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.tensor(
        [
            parse_action_window(row["action_target_window"], tau=6)
            for row in rows
        ],
        dtype=torch.float32,
    ).flatten(start_dim=1)
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False).clamp_min(1e-3)
    return mean, std


def _evaluate(
    policy: TinyFiLMDiffusionPolicy,
    rows: Sequence[Mapping[str, Any]],
    store: SemanticFeatureStore,
    normalization: NormalizationStats,
    clad: ControlledCLaD,
    adapter: nn.Module,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    *,
    batch_size: int,
    max_nodes: int,
    device: torch.device,
    seed: int,
) -> float:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    policy.eval()
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = _causal_batch(
                rows[start : start + batch_size], store, normalization,
                max_nodes=max_nodes, device=device,
            )
            observation, foresight = _condition_inputs(batch, clad, adapter)
            action = (batch["action_target"] - action_mean) / action_std
            timestep = torch.randint(
                0, policy.diffusion_steps, (action.shape[0],), device=device,
                generator=generator,
            )
            noise = torch.randn(action.shape, device=device, generator=generator)
            prediction = policy(
                policy.noisy_action(action, timestep, noise),
                timestep, observation, foresight,
            )
            losses.append(float(torch.mean((prediction - noise) ** 2).cpu()))
    if not losses:
        raise ValueError("validation produced no batches")
    return float(np.mean(losses))


def _sensitivity(
    policy: TinyFiLMDiffusionPolicy,
    batch: Mapping[str, torch.Tensor],
    clad: ControlledCLaD,
    adapter: nn.Module,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, float]:
    policy.eval()
    with torch.no_grad():
        observation, foresight = _condition_inputs(batch, clad, adapter)
        action = (batch["action_target"] - action_mean) / action_std
        timestep = torch.full(
            (action.shape[0],), policy.diffusion_steps // 2,
            dtype=torch.long, device=device,
        )
        noise = torch.zeros_like(action)
        noisy = policy.noisy_action(action, timestep, noise)
        actual = policy(noisy, timestep, observation, foresight)
        zeroed = policy(noisy, timestep, observation, torch.zeros_like(foresight))
        shuffled = policy(noisy, timestep, observation, foresight.flip(0))
    return {
        "zeroed_mean_abs_delta": float(torch.mean(torch.abs(actual - zeroed)).cpu()),
        "shuffled_mean_abs_delta": float(torch.mean(torch.abs(actual - shuffled)).cpu()),
    }


def train(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("stage2_policy", config)
    if not isinstance(section, Mapping):
        raise ValueError("stage2_policy config must be an object")
    seed = int(section.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(str(section.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Stage 2 pilot requested CUDA but CUDA is unavailable")
    manifest_path = _path(section["policy_manifest"])
    output_root = _path(section["output_root"])
    if (output_root / "runtime_manifest.json").exists():
        raise FileExistsError("Stage 2 pilot output is immutable; use a new versioned output_root")
    output_root.mkdir(parents=True, exist_ok=True)
    train_rows = _load_rows(
        manifest_path, str(section.get("train_split", "train")),
        section.get("max_train_rows"),
    )
    validation_rows = _load_rows(
        manifest_path, str(section.get("validation_split", "validation")),
        section.get("max_validation_rows"),
    )
    base_path = _path(section["base_checkpoint"])
    clad, normalization, _ = _load_base(base_path, device)
    arm = str(section.get("arm", "semantic"))
    model_id = str(section.get("structured_model_id", "C3-RelMPNN-PastAct"))
    if arm == "semantic":
        foresight_adapter: nn.Module = SemanticForesightInterface(hidden_dim=clad.hidden_dim).to(device)
    elif arm == "graph":
        checkpoint = section.get("structured_checkpoint")
        if not checkpoint:
            raise ValueError("graph Stage 2 arm requires structured_checkpoint")
        foresight_adapter = _load_graph_adapter(
            _path(checkpoint), model_id=model_id, device=device
        )
    else:
        raise ValueError("arm must be semantic or graph")
    # The semantic arm is entirely frozen.  The graph arm trains only the
    # residual adapter (delta layers and alpha gates); its Phase 3C encoder is
    # frozen so the comparison isolates the Stage-2 conditioning change.
    for parameter in foresight_adapter.parameters():
        parameter.requires_grad_(False)
    if arm == "graph":
        structured_encoder = getattr(foresight_adapter, "structured_encoder", None)
        frozen_ids = (
            {id(parameter) for parameter in structured_encoder.parameters()}
            if structured_encoder is not None
            else set()
        )
        for parameter in foresight_adapter.parameters():
            if id(parameter) not in frozen_ids:
                parameter.requires_grad_(True)
    action_mean, action_std = _action_stats(train_rows)
    action_mean = action_mean.to(device)
    action_std = action_std.to(device)
    policy = TinyFiLMDiffusionPolicy(
        hidden_dim=int(section.get("policy_hidden_dim", 256)),
        diffusion_steps=int(section.get("diffusion_steps", 1000)),
    ).to(device)
    trainable = list(policy.parameters())
    if arm == "graph":
        trainable.extend(
            parameter for parameter in foresight_adapter.parameters()
            if parameter.requires_grad
        )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(section.get("learning_rate", 3e-4)),
        weight_decay=float(section.get("weight_decay", 1e-4)),
    )
    updates = int(section.get("updates", 1000))
    batch_size = int(section.get("batch_size", 64))
    validation_interval = int(section.get("validation_interval", max(1, updates // 4)))
    max_nodes = int(section.get("max_nodes", 32))
    if updates <= 0 or batch_size <= 0:
        raise ValueError("updates and batch_size must be positive")
    start_time = time.time()
    best_loss = float("inf")
    best_update = 0
    losses: list[float] = []
    integrity: dict[str, Any] | None = None
    sensitivity: dict[str, float] | None = None
    generator = np.random.default_rng(seed)
    with SemanticFeatureStore(_path(section["semantic_store"])) as store:
        integrity = store.verify_integrity()
        for update in range(1, updates + 1):
            indices = generator.integers(0, len(train_rows), size=batch_size)
            rows = [train_rows[int(index)] for index in indices]
            batch = _causal_batch(rows, store, normalization, max_nodes=max_nodes, device=device)
            observation, foresight = _condition_inputs(batch, clad, foresight_adapter)
            action = (batch["action_target"] - action_mean) / action_std
            timestep = torch.randint(0, policy.diffusion_steps, (action.shape[0],), device=device)
            noise = torch.randn_like(action)
            prediction = policy(
                policy.noisy_action(action, timestep, noise),
                timestep, observation, foresight,
            )
            loss = torch.mean((prediction - noise) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stage 2 DDPM loss at update {update}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if update % validation_interval == 0 or update == updates:
                validation_loss = _evaluate(
                    policy, validation_rows, store, normalization, clad,
                    foresight_adapter, action_mean, action_std,
                    batch_size=batch_size, max_nodes=max_nodes, device=device,
                    seed=seed + update,
                )
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_update = update
                    torch.save(
                        {
                            "schema": "phase5-policy-checkpoint.v1",
                            "arm": arm,
                            "model_id": model_id if arm == "graph" else None,
                            "selected_update": update,
                            "validation_ddpm_loss": best_loss,
                            "policy_state": policy.state_dict(),
                            "foresight_adapter_state": foresight_adapter.state_dict(),
                            "action_mean": action_mean.detach().cpu(),
                            "action_std": action_std.detach().cpu(),
                            "base_checkpoint_sha256": _sha256_file(base_path),
                            "manifest_sha256": _sha256_file(manifest_path),
                        },
                        output_root / "best.pt",
                    )
        best_payload = _torch_load(output_root / "best.pt")
        policy.load_state_dict(best_payload["policy_state"])
        foresight_adapter.load_state_dict(best_payload["foresight_adapter_state"])
        sensitivity_batch = _causal_batch(
            validation_rows[: min(batch_size, len(validation_rows))],
            store,
            normalization,
            max_nodes=max_nodes,
            device=device,
        )
        sensitivity = _sensitivity(
            policy,
            sensitivity_batch,
            clad,
            foresight_adapter,
            action_mean,
            action_std,
            device=device,
        )
    if best_update <= 0:
        raise RuntimeError("Stage 2 pilot completed without a validation checkpoint")
    runtime = {
        "schema": "phase5-policy-run.v1",
        "status": "completed",
        "claim_scope": "stage2-one-task-preliminary-pilot",
        "arm": arm,
        "structured_model_id": model_id if arm == "graph" else None,
        "seed": seed,
        "updates": updates,
        "batch_size": batch_size,
        "selected_update": best_update,
        "best_validation_ddpm_loss": best_loss,
        "best_checkpoint": str(output_root / "best.pt"),
        "mean_last_100_loss": float(np.mean(losses[-100:])),
        "elapsed_seconds": time.time() - start_time,
        "training_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": _sha256_file(base_path),
        "policy_manifest": str(manifest_path),
        "policy_manifest_sha256": _sha256_file(manifest_path),
        "stage2_limitations": [
            "controlled DDPM reimplementation, not official CLaD Stage 2",
            "one task and one seed",
            "offline policy loss smoke; no LIBERO rollout success claim",
        ],
        "causal_input_contract": {
            "uses_future_action_as_input": False,
            "uses_future_graph_as_input": False,
            "uses_target_metadata_as_input": False,
            "target": "action_target_window",
        },
        "semantic_store_integrity": integrity,
        "foresight_sensitivity": sensitivity,
    }
    write_json(output_root / "runtime_manifest.json", runtime)
    write_json(
        output_root / "metrics.json",
        {
            "schema": "phase5-policy-metrics.v1",
            "status": "completed",
            "runtime": runtime,
        },
    )
    return runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(train(load_json_config(args.config)), ensure_ascii=False))


if __name__ == "__main__":
    main()
