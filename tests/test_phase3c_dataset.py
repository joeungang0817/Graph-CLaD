from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class Phase3CDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            cls.torch = None
            return
        cls.torch = torch

    def test_semantic_store_rejects_wrong_schema(self):
        if self.torch is None:
            self.skipTest("torch is not installed in the local CPU environment")
        from scripts.phase3c.dataset import SemanticFeatureStore
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"schema": "bad"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                SemanticFeatureStore(root)

    def test_graph_tensors_reserve_task_slot_and_build_masks(self):
        if self.torch is None:
            self.skipTest("torch is not installed in the local CPU environment")
        from scripts.phase3c.dataset import graph_tensors
        def node(node_id, node_type, value):
            return {"node_id": node_id, "node_type": node_type, "feature_vector": [0.0] * 24, "features": {"joint_pos": [0.0] * 7, "joint_vel": [0.0] * 7, "gripper_qpos": [0.0] * 2}}
        previous = {"nodes": [node("robot0", "robot", 0), node("obj", "object", 0)], "edges": []}
        current = {"nodes": [node("robot0", "robot", 0), node("obj", "object", 0)], "edges": []}
        previous["nodes"][0]["feature_vector"][4] = 1.0
        with self.assertRaises(ValueError):
            graph_tensors(previous, current)
        previous["nodes"][0]["feature_vector"][4] = 0.0
        prev_graph, curr_graph, prev_p, curr_p = graph_tensors(previous, current)
        self.assertEqual(tuple(prev_graph.node_features.shape), (2, 24))
        self.assertEqual(len(prev_p), 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
