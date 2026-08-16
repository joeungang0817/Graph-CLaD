"""Controlled wrapper for the original CLaD Stage 1 latent dynamics model."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CLaDBatch:
    """Minimum tensor contract consumed by :class:`ControlledCLaD`.

    Graph tensors and metadata are deliberately absent.  Structured adapters
    receive the same batch plus graph fields later, while this wrapper only
    sees the replayed semantic/proprio/action path.
    """

    v_history: torch.Tensor
    p_history: torch.Tensor
    past_action: torch.Tensor
    language: torch.Tensor
    target_p: torch.Tensor | None = None
    target_v: torch.Tensor | None = None


def _get(batch: Any, name: str) -> Any:
    if isinstance(batch, Mapping):
        if name not in batch:
            raise KeyError(f"CLaD batch is missing {name}")
        return batch[name]
    if not hasattr(batch, name):
        raise AttributeError(f"CLaD batch is missing {name}")
    return getattr(batch, name)


def _finite_tensor(value: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(value):
        value = value.float()
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


@contextmanager
def _temporary_eval(module: nn.Module):
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


class ControlledCLaD(nn.Module):
    """Use the unchanged original CLaD Stage 1 model under a strict contract.

    The wrapper exposes only two paths:

    * ``training_loss`` supplies the target tensors to the original objective;
    * ``encode_foresight`` supplies no target/future input and returns the
      concatenated semantic/proprio foresight `[B, 2*hidden_dim]`.

    A later graph adapter can consume that representation without changing the
    original CLaD code or accidentally forwarding future action.
    """

    def __init__(
        self,
        *,
        proprio_dim: int = 16,
        vl_dim: int = 1024,
        hidden_dim: int = 1024,
        action_dim: int = 42,
        m_ema: float = 0.995,
        core: nn.Module | None = None,
    ):
        super().__init__()
        if int(vl_dim) != int(hidden_dim):
            raise ValueError("Phase 3C ControlledCLaD requires vl_dim == hidden_dim")
        if int(proprio_dim) != 16:
            raise ValueError("Phase 3C proprio contract is fixed at 16 dimensions")
        if int(action_dim) != 42:
            raise ValueError("Phase 3C action contract is fixed at tau*action_dim = 6*7 = 42")
        if not 0.0 < float(m_ema) < 1.0:
            raise ValueError("m_ema must be in (0, 1)")
        if core is None:
            from baseline_code.LatentDynamics import LatentDynamics

            core = LatentDynamics(proprio_dim, vl_dim, hidden_dim, action_dim)
        self.core = core
        if hasattr(self.core, "m_ema"):
            self.core.m_ema = float(m_ema)
        self.proprio_dim = int(proprio_dim)
        self.vl_dim = int(vl_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.tau = 6
        self.views = 2

    def _inputs(self, batch: Any) -> tuple[torch.Tensor, ...]:
        v_history = _finite_tensor(_get(batch, "v_history"), "v_history")
        p_history = _finite_tensor(_get(batch, "p_history"), "p_history")
        past_action = _finite_tensor(_get(batch, "past_action"), "past_action")
        language = _finite_tensor(_get(batch, "language"), "language")
        if v_history.ndim != 4 or tuple(v_history.shape[1:3]) != (2, self.views):
            raise ValueError(f"v_history must have shape [B,2,2,{self.vl_dim}], got {tuple(v_history.shape)}")
        if v_history.shape[-1] != self.vl_dim:
            raise ValueError("v_history feature dimension does not match vl_dim")
        if p_history.ndim != 3 or tuple(p_history.shape[1:]) != (2, self.proprio_dim):
            raise ValueError(f"p_history must have shape [B,2,{self.proprio_dim}], got {tuple(p_history.shape)}")
        if past_action.ndim == 3:
            if tuple(past_action.shape[1:]) != (self.tau, 7):
                raise ValueError(f"past_action must have shape [B,6,7], got {tuple(past_action.shape)}")
            past_action = past_action.flatten(start_dim=1)
        elif past_action.ndim == 2 and past_action.shape[1] == self.action_dim:
            pass
        else:
            raise ValueError(f"past_action must have shape [B,6,7] or [B,42], got {tuple(past_action.shape)}")
        if language.ndim != 2 or language.shape[1] != self.vl_dim:
            raise ValueError(f"language must have shape [B,{self.vl_dim}], got {tuple(language.shape)}")
        batch_size = v_history.shape[0]
        if any(value.shape[0] != batch_size for value in (p_history, past_action, language)):
            raise ValueError("CLaD batch tensors have inconsistent batch dimensions")
        return v_history, p_history, past_action, language

    def training_loss(self, batch: Any) -> dict[str, torch.Tensor]:
        v_history, p_history, past_action, language = self._inputs(batch)
        target_p = _finite_tensor(_get(batch, "target_p"), "target_p")
        target_v = _finite_tensor(_get(batch, "target_v"), "target_v")
        if target_p.ndim != 2 or target_p.shape[1] != self.proprio_dim:
            raise ValueError(f"target_p must have shape [B,{self.proprio_dim}], got {tuple(target_p.shape)}")
        if target_v.ndim != 3 or tuple(target_v.shape[1:]) != (self.views, self.vl_dim):
            raise ValueError(f"target_v must have shape [B,2,{self.vl_dim}], got {tuple(target_v.shape)}")
        if target_p.shape[0] != v_history.shape[0] or target_v.shape[0] != v_history.shape[0]:
            raise ValueError("CLaD targets have inconsistent batch dimensions")
        outputs = self.core(
            v_history,
            p_history,
            past_action,
            language,
            p_next=target_p,
            v_next=target_v,
            action_mask_ratio=0.3,
        )
        if not isinstance(outputs, Mapping):
            raise RuntimeError("CLaD core returned evaluation output while training")
        losses = {str(key): _finite_tensor(value, f"losses[{key}]") for key, value in outputs.items()}
        required = {"loss_p", "loss_s", "loss_p_recon", "loss_v_recon"}
        missing = sorted(required - set(losses))
        if missing:
            raise RuntimeError(f"CLaD core loss dictionary is missing {missing}")
        return losses

    @torch.no_grad()
    def encode_foresight(self, batch: Any) -> torch.Tensor:
        v_history, p_history, past_action, language = self._inputs(batch)
        # Eval mode prevents the original action mask from injecting randomness
        # and, importantly, the target tensors are never read on this path.
        with _temporary_eval(self.core):
            output = self.core(v_history, p_history, past_action, language, action_mask_ratio=0.0)
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise RuntimeError("CLaD core did not return (pred_p_emb, pred_s_emb) in eval mode")
        pred_p, pred_s = (_finite_tensor(output[0], "pred_p_emb"), _finite_tensor(output[1], "pred_s_emb"))
        if pred_p.ndim != 2 or pred_s.ndim != 2 or pred_p.shape != pred_s.shape:
            raise ValueError("CLaD foresight embeddings must be matching rank-2 tensors")
        if pred_p.shape[1] != self.hidden_dim:
            raise ValueError("CLaD foresight dimension does not match hidden_dim")
        return torch.cat([pred_p, pred_s], dim=-1)

    @torch.no_grad()
    def update_ema_after_optimizer_step(self) -> None:
        update = getattr(self.core, "update_ema", None)
        if not callable(update):
            raise RuntimeError("CLaD core does not expose update_ema")
        update()

    @property
    def foresight_dim(self) -> int:
        return 2 * self.hidden_dim
