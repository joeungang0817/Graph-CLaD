from __future__ import annotations

import unittest


class Phase4ForesightAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            cls.torch = None
            return
        cls.torch = torch

    def _batch(self):
        from scripts.phase3c.models.structured import GraphBatch, StructuredBatch

        torch = self.torch
        node = torch.zeros(3, 2, 8)
        node[:, 0, 0] = 1.0
        node[:, 1, 1] = 1.0
        node[..., 7] = 1.0
        mask = torch.ones(3, 2, dtype=torch.bool)
        edge_mask = torch.ones(3, 2, 2, dtype=torch.bool)
        edge_mask[:, torch.arange(2), torch.arange(2)] = False

        def graph():
            return GraphBatch(
                node.clone(),
                mask.clone(),
                torch.zeros(3, 2, 2, 5),
                torch.zeros(3, 2, 2, 2),
                torch.zeros(3, 2, 2, 14),
                edge_mask.clone(),
            )

        return StructuredBatch(graph(), graph(), torch.randn(3, 6, 7))

    def test_zero_alpha_matches_semantic_interface_exactly(self):
        if self.torch is None:
            self.skipTest("torch is unavailable")
        import torch
        from scripts.phase3c.models.adapters import SemanticPastActEncoder
        from scripts.phase4.models.foresight_adapter import (
            ResidualGraphForesightAdapter,
            SemanticForesightInterface,
        )

        torch.manual_seed(0)
        batch = self._batch()
        semantic = torch.randn(3, 16)
        baseline = SemanticForesightInterface(hidden_dim=8)(semantic)
        graph = ResidualGraphForesightAdapter(
            SemanticPastActEncoder(output_dim=4, hidden_dim=6),
            structured_dim=4,
            hidden_dim=8,
        )(batch, semantic)
        for key in ("pred_p", "pred_s", "foresight"):
            self.assertTrue(torch.equal(baseline[key], graph[key]), key)

    def test_adapter_off_skips_encoder_and_future_metadata(self):
        if self.torch is None:
            self.skipTest("torch is unavailable")
        import torch
        import torch.nn as nn
        from scripts.phase4.models.foresight_adapter import ResidualGraphForesightAdapter

        class ForbiddenEncoder(nn.Module):
            def forward(self, batch):
                raise AssertionError("adapter-off must not execute structured encoder")

        class FullBatch:
            pass

        source = self._batch()
        batch = FullBatch()
        batch.graph_prev = source.graph_prev
        batch.graph_current = source.graph_current
        batch.past_action = source.past_action
        batch.target_v = torch.full((3, 2, 8), float("nan"))
        batch.future_action = torch.full((3, 6, 7), float("nan"))
        model = ResidualGraphForesightAdapter(
            ForbiddenEncoder(), structured_dim=4, hidden_dim=8
        )
        result = model(batch, torch.randn(3, 16), adapter_enabled=False)
        self.assertEqual(tuple(result["foresight"].shape), (3, 16))
        self.assertTrue(torch.isfinite(result["foresight"]).all())

    def test_nonzero_residual_responds_to_causal_structured_input(self):
        if self.torch is None:
            self.skipTest("torch is unavailable")
        import torch
        from scripts.phase3c.models.adapters import SemanticPastActEncoder
        from scripts.phase4.models.foresight_adapter import ResidualGraphForesightAdapter

        batch = self._batch()
        model = ResidualGraphForesightAdapter(
            SemanticPastActEncoder(output_dim=4, hidden_dim=6),
            structured_dim=4,
            hidden_dim=8,
        ).eval()
        model.alpha.data.fill_(1.0)
        semantic = torch.randn(3, 16)
        first = model(batch, semantic)["foresight"]
        batch.past_action.add_(1.0)
        second = model(batch, semantic)["foresight"]
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), tuple(second.shape))


if __name__ == "__main__":
    unittest.main(verbosity=2)
