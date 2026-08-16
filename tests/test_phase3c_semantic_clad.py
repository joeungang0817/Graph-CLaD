from __future__ import annotations

import unittest


class Phase3CSemanticCLaDTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            cls.torch = None
            return
        cls.torch = torch

    def test_contract_rejects_wrong_dimensions_without_torch(self):
        if self.torch is None:
            self.skipTest("torch is not installed in the local CPU environment")
        from scripts.phase3c.models.semantic_clad import ControlledCLaD

        with self.assertRaises(ValueError):
            ControlledCLaD(proprio_dim=15, vl_dim=32, hidden_dim=32, action_dim=42)
        with self.assertRaises(ValueError):
            ControlledCLaD(proprio_dim=16, vl_dim=32, hidden_dim=64, action_dim=42)

    def test_synthetic_shape_and_ema_path(self):
        if self.torch is None:
            self.skipTest("torch is not installed in the local CPU environment")
        import torch
        import torch.nn as nn

        from scripts.phase3c.models.semantic_clad import ControlledCLaD

        class FakeCore(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))
                self.m_ema = 0.995
                self.update_calls = 0

            def forward(self, v_history, p_history, action, language, p_next=None, v_next=None, action_mask_ratio=0.3):
                latent = v_history[:, -1].mean(dim=1) * self.weight
                if self.training:
                    return {
                        "loss_p": latent.mean() * 0 + self.weight.square(),
                        "loss_s": latent.mean() * 0 + self.weight.square(),
                        "loss_p_recon": latent.mean() * 0 + self.weight.square(),
                        "loss_v_recon": latent.mean() * 0 + self.weight.square(),
                    }
                return latent, latent

            def update_ema(self):
                self.update_calls += 1

        model = ControlledCLaD(proprio_dim=16, vl_dim=8, hidden_dim=8, action_dim=42, core=FakeCore())
        batch = {
            "v_history": torch.randn(2, 2, 2, 8),
            "p_history": torch.randn(2, 2, 16),
            "past_action": torch.randn(2, 6, 7),
            "language": torch.randn(2, 8),
            "target_p": torch.randn(2, 16),
            "target_v": torch.randn(2, 2, 8),
        }
        losses = model.training_loss(batch)
        self.assertEqual(set(losses), {"loss_p", "loss_s", "loss_p_recon", "loss_v_recon"})
        features = model.encode_foresight(batch)
        self.assertEqual(tuple(features.shape), (2, 16))
        model.update_ema_after_optimizer_step()
        self.assertEqual(model.core.update_calls, 1)

    def test_future_targets_are_not_required_for_foresight(self):
        if self.torch is None:
            self.skipTest("torch is not installed in the local CPU environment")
        import torch
        import torch.nn as nn

        from scripts.phase3c.models.semantic_clad import ControlledCLaD

        class EvalOnlyCore(nn.Module):
            def forward(self, v_history, p_history, action, language, **kwargs):
                self.last_kwargs = kwargs
                value = v_history[:, -1, 0]
                return value, value

            def update_ema(self):
                pass

        model = ControlledCLaD(proprio_dim=16, vl_dim=4, hidden_dim=4, action_dim=42, core=EvalOnlyCore())
        batch = {
            "v_history": torch.zeros(1, 2, 2, 4),
            "p_history": torch.zeros(1, 2, 16),
            "past_action": torch.zeros(1, 6, 7),
            "language": torch.zeros(1, 4),
        }
        self.assertEqual(tuple(model.encode_foresight(batch).shape), (1, 8))
        self.assertEqual(model.core.last_kwargs, {"action_mask_ratio": 0.0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
