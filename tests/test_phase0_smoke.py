"""Phase 0 synthetic smoke test for the supplied CLaD Stage 1 code.

This test intentionally uses a compact configuration so it can run on CPU.
The production-dimension assumptions are recorded separately in
``configs/phase0_synthetic.json`` and are not silently treated as verified.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "Phase 0 requires PyTorch. Install a compatible torch build before "
        "running tests/test_phase0_smoke.py."
    ) from exc

from baseline_code.Attentions import CrossAttnBlock
from baseline_code.LatentDynamics import LatentDynamics


SMOKE_CONFIG = {
    "batch_size": 2,
    "history_length": 2,
    "num_views": 2,
    "proprio_dim": 6,
    "visual_dim_per_view": 64,
    "vl_dim": 64,
    "hidden_dim": 64,
    "action_dim": 5,
}


def _cpu_safe_attention(self, x, y, return_attn_weights=False, prefer_flash=True):
    """Disable CUDA-only attention preferences for CPU smoke tests.

    The supplied attention module already exposes ``prefer_flash``.  The
    baseline LatentDynamics calls its blocks with the default value, so the
    test overrides only that default on CPU without modifying baseline code.
    """

    return _ORIGINAL_CROSS_ATTN_FORWARD(
        self,
        x,
        y,
        return_attn_weights=return_attn_weights,
        prefer_flash=prefer_flash if torch.cuda.is_available() else False,
    )


_ORIGINAL_CROSS_ATTN_FORWARD = CrossAttnBlock.forward


class Phase0SmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cfg = SMOKE_CONFIG
        cls.model = LatentDynamics(
            proprio_dim=cfg["proprio_dim"],
            vl_dim=cfg["vl_dim"],
            hidden_dim=cfg["hidden_dim"],
            action_dim=cfg["action_dim"],
        )
        cls.inputs = {
            "v_history": torch.randn(
                cfg["batch_size"],
                cfg["history_length"],
                cfg["num_views"],
                cfg["visual_dim_per_view"],
            ),
            "p_history": torch.randn(
                cfg["batch_size"], cfg["history_length"], cfg["proprio_dim"]
            ),
            "prev_action": torch.randn(cfg["batch_size"], cfg["action_dim"]),
            "lang": torch.randn(cfg["batch_size"], cfg["vl_dim"]),
            "p_next": torch.randn(cfg["batch_size"], cfg["proprio_dim"]),
            "v_next": torch.randn(
                cfg["batch_size"],
                cfg["num_views"],
                cfg["visual_dim_per_view"],
            ),
        }

    def setUp(self):
        torch.manual_seed(1)
        self.model.zero_grad(set_to_none=True)
        self.model.train()

    def test_input_contract(self):
        cfg = SMOKE_CONFIG
        self.assertEqual(
            cfg["num_views"] * cfg["visual_dim_per_view"],
            2 * cfg["vl_dim"],
            "s_backbone input must match flattened view features",
        )
        self.assertEqual(
            cfg["vl_dim"],
            cfg["hidden_dim"],
            "semantic features must match the query embedding dimension",
        )

    def test_train_forward_loss_shapes_and_backward(self):
        with patch.object(CrossAttnBlock, "forward", _cpu_safe_attention):
            losses = self.model(**self.inputs)

        expected_keys = {"loss_p", "loss_s", "loss_p_recon", "loss_v_recon"}
        self.assertEqual(set(losses), expected_keys)
        for name, loss in losses.items():
            self.assertEqual(loss.ndim, 0, f"{name} must be scalar")
            self.assertTrue(torch.isfinite(loss).item(), f"{name} is not finite")

        total_loss = sum(losses.values())
        total_loss.backward()

        gradients = [
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients, "no trainable parameter received a gradient")
        self.assertTrue(
            all(torch.isfinite(gradient).all().item() for gradient in gradients),
            "a trainable gradient contains NaN or Inf",
        )

    def test_eval_output_interface(self):
        self.model.eval()
        with torch.no_grad(), patch.object(CrossAttnBlock, "forward", _cpu_safe_attention):
            pred_p_emb, pred_s_emb = self.model(
                self.inputs["v_history"],
                self.inputs["p_history"],
                self.inputs["prev_action"],
                self.inputs["lang"],
            )

        expected_shape = (SMOKE_CONFIG["batch_size"], SMOKE_CONFIG["hidden_dim"])
        self.assertEqual(tuple(pred_p_emb.shape), expected_shape)
        self.assertEqual(tuple(pred_s_emb.shape), expected_shape)
        self.assertTrue(torch.isfinite(pred_p_emb).all().item())
        self.assertTrue(torch.isfinite(pred_s_emb).all().item())

    def test_ema_initialization_and_momentum_update(self):
        online = next(self.model.p_backbone.parameters())
        target = next(self.model.p_backbone_target.parameters())

        with torch.no_grad():
            online_before = online.detach().clone()
            self.model.update_ema()
            target_after_init = target.detach().clone()

        self.assertTrue(torch.allclose(target_after_init, online_before))

        with torch.no_grad():
            online.add_(0.25)
            target_before_momentum = target.detach().clone()
            online_after = online.detach().clone()
            self.model.update_ema()
            target_after_momentum = target.detach().clone()

        expected = (
            self.model.m_ema * target_before_momentum
            + (1.0 - self.model.m_ema) * online_after
        )
        self.assertTrue(torch.allclose(target_after_momentum, expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
