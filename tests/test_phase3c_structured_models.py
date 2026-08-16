from __future__ import annotations

import unittest


class Phase3CStructuredModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            cls.torch = None
            return
        cls.torch = torch

    def _batch(self):
        torch = self.torch
        from scripts.phase3c.models.structured import GraphBatch, StructuredBatch

        node = torch.randn(2, 5, 24)
        mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
        edge_mask = mask[:, :, None] & mask[:, None, :]
        # Remove self loops, matching the graph extractor contract.
        diagonal = torch.eye(5, dtype=torch.bool).unsqueeze(0)
        edge_mask &= ~diagonal
        def graph():
            return GraphBatch(
                node_features=node.clone(), node_mask=mask.clone(),
                edge_geometry=torch.randn(2, 5, 5, 5), edge_contact=torch.randn(2, 5, 5, 4),
                edge_relations=torch.randn(2, 5, 5, 28), edge_mask=edge_mask.clone(),
            )
        return StructuredBatch(graph_prev=graph(), graph_current=graph(), past_action=torch.randn(2, 6, 7))

    def test_all_core_models_have_same_output_shape(self):
        if self.torch is None:
            self.skipTest("torch is not installed in the local CPU environment")
        from scripts.phase3c.models.structured import build_structured_model

        batch = self._batch()
        for name in ("C3-SceneSet-PastAct", "C3-Pair-PastAct", "C3-GeomMPNN-PastAct", "C3-RelPool-PastAct", "C3-RelMPNN-PastAct"):
            model = build_structured_model(name, hidden_dim=16, output_dim=32)
            result = model(batch)
            self.assertEqual(tuple(result.shape), (2, 32), name)
            self.assertTrue(self.torch.isfinite(result).all(), name)

    def test_relpool_is_permutation_invariant(self):
        if self.torch is None:
            self.skipTest("torch is not installed in the local CPU environment")
        import torch
        from scripts.phase3c.models.structured import build_structured_model

        batch = self._batch()
        model = build_structured_model("C3-RelPool-PastAct", hidden_dim=16, output_dim=32).eval()
        first = model(batch)
        permutation = torch.tensor([0, 2, 1, 3, 4])
        def permute(graph):
            node = graph.node_features[:, permutation]
            node_mask = graph.node_mask[:, permutation]
            edge_geometry = graph.edge_geometry[:, permutation][:, :, permutation]
            edge_contact = graph.edge_contact[:, permutation][:, :, permutation]
            edge_relations = graph.edge_relations[:, permutation][:, :, permutation]
            edge_mask = graph.edge_mask[:, permutation][:, :, permutation]
            from scripts.phase3c.models.structured import GraphBatch
            return GraphBatch(node, node_mask, edge_geometry, edge_contact, edge_relations, edge_mask)
        from scripts.phase3c.models.structured import StructuredBatch
        second = model(StructuredBatch(permute(batch.graph_prev), permute(batch.graph_current), batch.past_action))
        self.assertTrue(torch.allclose(first, second, atol=1e-5, rtol=1e-5))

    def test_geom_model_does_not_read_relation_values(self):
        if self.torch is None:
            self.skipTest("torch is not installed in the local CPU environment")
        import torch
        from scripts.phase3c.models.structured import build_structured_model, StructuredBatch, GraphBatch

        batch = self._batch()
        model = build_structured_model("C3-GeomMPNN-PastAct", hidden_dim=16, output_dim=32).eval()
        first = model(batch)
        changed = []
        for graph in (batch.graph_prev, batch.graph_current):
            changed.append(GraphBatch(graph.node_features, graph.node_mask, graph.edge_geometry, graph.edge_contact, torch.randn_like(graph.edge_relations), graph.edge_mask))
        second = model(StructuredBatch(changed[0], changed[1], batch.past_action))
        self.assertTrue(torch.allclose(first, second, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
