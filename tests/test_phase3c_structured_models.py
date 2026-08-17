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

        mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
        node = torch.zeros(2, 5, 8)
        node[..., 4:7] = torch.randn(2, 5, 3)
        node[..., 7] = mask.float()
        node[:, 0, 0] = 1.0
        node[:, 1:, 1] = mask[:, 1:].float()
        edge_mask = mask[:, :, None] & mask[:, None, :]
        # Remove self loops, matching the graph extractor contract.
        diagonal = torch.eye(5, dtype=torch.bool).unsqueeze(0)
        edge_mask &= ~diagonal
        def graph():
            return GraphBatch(
                node_features=node.clone(), node_mask=mask.clone(),
                edge_geometry=torch.randn(2, 5, 5, 5), edge_contact=torch.randn(2, 5, 5, 2),
                edge_relations=torch.randn(2, 5, 5, 14), edge_mask=edge_mask.clone(),
            )
        return StructuredBatch(graph_prev=graph(), graph_current=graph(), past_action=torch.randn(2, 6, 7))

    def _permute(self, batch, permutation):
        from scripts.phase3c.models.structured import GraphBatch, StructuredBatch

        def graph(value):
            return GraphBatch(
                value.node_features[:, permutation],
                value.node_mask[:, permutation],
                value.edge_geometry[:, permutation][:, :, permutation],
                value.edge_contact[:, permutation][:, :, permutation],
                value.edge_relations[:, permutation][:, :, permutation],
                value.edge_mask[:, permutation][:, :, permutation],
            )

        return StructuredBatch(
            graph(batch.graph_prev), graph(batch.graph_current), batch.past_action
        )

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

    def test_all_set_and_graph_models_obey_node_permutation_contract(self):
        if self.torch is None:
            self.skipTest("torch is not installed in the local CPU environment")
        import torch
        from scripts.phase3c.models.structured import build_structured_model

        batch = self._batch()
        permutations = {
            "C3-SceneSet-PastAct": torch.tensor([2, 0, 3, 1, 4]),
            # PairPastAct fixes the unique robot in slot zero; only entity
            # ordering is exchangeable.
            "C3-Pair-PastAct": torch.tensor([0, 3, 1, 2, 4]),
            "C3-GeomMPNN-PastAct": torch.tensor([2, 0, 3, 1, 4]),
            "C3-RelPool-PastAct": torch.tensor([2, 0, 3, 1, 4]),
            "C3-RelMPNN-PastAct": torch.tensor([2, 0, 3, 1, 4]),
        }
        for name, permutation in permutations.items():
            model = build_structured_model(name, hidden_dim=16, output_dim=32).eval()
            first = model(batch)
            second = model(self._permute(batch, permutation))
            self.assertTrue(
                torch.allclose(first, second, atol=2e-5, rtol=2e-5), name
            )

    def test_relpool_and_relmpnn_share_identical_raw_edge_token_contract(self):
        if self.torch is None:
            self.skipTest("torch is not installed in the local CPU environment")
        import torch
        from scripts.phase3c.models.structured import build_structured_model

        batch = self._batch()
        pool = build_structured_model(
            "C3-RelPool-PastAct", hidden_dim=16, output_dim=32
        ).eval()
        mpnn = build_structured_model(
            "C3-RelMPNN-PastAct", hidden_dim=16, output_dim=32
        ).eval()
        for name in ("node_encoder", "action_film", "edge_encoder"):
            getattr(mpnn, name).load_state_dict(getattr(pool, name).state_dict())
        pool_prepared = pool._prepare(batch)
        mpnn_prepared = mpnn._prepare(batch)
        pool_tokens, pool_mask = pool._edge_tokens(
            pool_prepared[0], pool_prepared[1], pool_prepared[2], pool_prepared[3]
        )
        mpnn_tokens, mpnn_mask = mpnn._edge_tokens(
            mpnn_prepared[0], mpnn_prepared[1], mpnn_prepared[2], mpnn_prepared[3]
        )
        self.assertTrue(torch.equal(pool_mask, mpnn_mask))
        self.assertTrue(torch.equal(pool_tokens, mpnn_tokens))

    def test_adapter_off_is_independent_of_structured_representation(self):
        if self.torch is None:
            self.skipTest("torch is not installed in the local CPU environment")
        import torch
        from scripts.phase3c.models.adapters import CommonRelationHead

        head = CommonRelationHead(
            semantic_dim=8, structured_dim=4, latent_dim=6, relation_dim=3
        ).eval()
        semantic = torch.randn(2, 8)
        first = head(semantic, torch.randn(2, 4), adapter_enabled=False)
        second = head(semantic, torch.randn(2, 4) * 1e6, adapter_enabled=False)
        for key in ("latent", "relation_logits", "scene_motion"):
            self.assertTrue(torch.equal(first[key], second[key]), key)

    def test_tiny_adapter_can_overfit_a_fixed_batch(self):
        if self.torch is None:
            self.skipTest("torch is not installed in the local CPU environment")
        import torch
        import torch.nn.functional as functional
        from scripts.phase3c.models.adapters import Phase3CAdapter

        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.project = torch.nn.Linear(42, 4)

            def forward(self, batch):
                return self.project(batch.past_action.flatten(start_dim=1))

        torch.manual_seed(11)
        batch = self._batch()
        model = Phase3CAdapter(
            TinyEncoder(), semantic_dim=8, structured_dim=4, relation_dim=3
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        semantic = torch.randn(2, 8)
        target = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        model.train()
        with torch.no_grad():
            initial = functional.binary_cross_entropy_with_logits(
                model(batch, semantic)["relation_logits"], target
            ).item()
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            loss = functional.binary_cross_entropy_with_logits(
                model(batch, semantic)["relation_logits"], target
            )
            loss.backward()
            optimizer.step()
        model.eval()
        final = functional.binary_cross_entropy_with_logits(
            model(batch, semantic)["relation_logits"], target
        ).item()
        self.assertLess(final, min(0.05, initial * 0.2))

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

    def test_scene_model_keeps_valid_isolated_nodes(self):
        if self.torch is None:
            self.skipTest("torch is not installed in the local CPU environment")
        from scripts.phase3c.models.structured import build_structured_model

        batch = self._batch()
        batch.graph_prev.edge_mask.zero_()
        batch.graph_current.edge_mask.zero_()
        model = build_structured_model("C3-SceneSet-PastAct", hidden_dim=16, output_dim=32)
        _, _, _, _, prepared_mask = model._prepare(batch)
        self.assertTrue(self.torch.equal(prepared_mask, batch.graph_prev.node_mask & batch.graph_current.node_mask))

    def test_six_adapters_can_match_relmpnn_budget(self):
        if self.torch is None:
            self.skipTest("torch is not installed in the local CPU environment")
        from scripts.phase3c.models.adapters import Phase3CAdapter, SemanticPastActEncoder
        from scripts.phase3c.models.structured import build_structured_model
        from scripts.phase3c.parameter_match import select_width, trainable_parameter_count

        names = (
            "C3-Sem-PastAct", "C3-SceneSet-PastAct", "C3-Pair-PastAct",
            "C3-GeomMPNN-PastAct", "C3-RelPool-PastAct", "C3-RelMPNN-PastAct",
        )

        def adapter(name, width):
            structured = (
                SemanticPastActEncoder(256, hidden_dim=width)
                if name == "C3-Sem-PastAct"
                else build_structured_model(name, hidden_dim=width, output_dim=256)
            )
            return Phase3CAdapter(structured, semantic_dim=2048, structured_dim=256)

        target = trainable_parameter_count(adapter("C3-RelMPNN-PastAct", 128))
        for name in names:
            report = select_width(
                lambda width, model_name=name: adapter(model_name, width),
                range(64, 385, 8),
                target,
                tolerance=0.05,
            )
            self.assertTrue(report["within_tolerance"], (name, report))

    def test_structured_models_have_no_dead_trainable_parameters(self):
        if self.torch is None:
            self.skipTest("torch is not installed in the local CPU environment")
        names = (
            "C3-SceneSet-PastAct",
            "C3-Pair-PastAct",
            "C3-GeomMPNN-PastAct",
            "C3-RelPool-PastAct",
            "C3-RelMPNN-PastAct",
        )
        from scripts.phase3c.models.structured import build_structured_model

        batch = self._batch()
        for name in names:
            model = build_structured_model(name, hidden_dim=16, output_dim=32)
            model(batch).sum().backward()
            missing = [
                parameter_name
                for parameter_name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.grad is None
            ]
            self.assertEqual(missing, [], (name, missing))

    def test_adapter_physically_removes_future_targets_and_metadata(self):
        if self.torch is None:
            self.skipTest("torch is not installed in the local CPU environment")
        import torch
        import torch.nn as nn
        from scripts.phase3c.models.adapters import Phase3CAdapter

        class InspectEncoder(nn.Module):
            def forward(self, value):
                self.seen = value
                return value.past_action.flatten(start_dim=1)[:, :4]

        class FullBatch:
            pass

        source = self._batch()
        batch = FullBatch()
        batch.graph_prev = source.graph_prev
        batch.graph_current = source.graph_current
        batch.past_action = source.past_action
        batch.task_ids = torch.tensor([0, 1])
        batch.target_v = torch.full((2, 2, 4), 1e9)
        batch.target_p = torch.full((2, 16), -1e9)
        batch.target_relation_change = torch.ones(2, 8)
        encoder = InspectEncoder()
        adapter = Phase3CAdapter(encoder, semantic_dim=8, structured_dim=4).eval()
        first = adapter(batch, torch.zeros(2, 8))["relation_logits"]
        batch.target_v.fill_(-1e9)
        batch.target_p.fill_(1e9)
        batch.target_relation_change.zero_()
        second = adapter(batch, torch.zeros(2, 8))["relation_logits"]
        self.assertFalse(hasattr(encoder.seen, "target_v"))
        self.assertFalse(hasattr(encoder.seen, "task_ids"))
        self.assertTrue(torch.allclose(first, second, atol=0.0, rtol=0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
