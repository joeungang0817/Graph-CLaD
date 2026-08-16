from __future__ import annotations

import json
import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np


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
            return {
                "node_id": node_id,
                "node_type": node_type,
                "feature_vector": [0.0] * 24,
                "features": {
                    "position": [float(value), 0.0, 0.0], "position_valid": 1,
                    "joint_pos": [0.0] * 7, "joint_pos_valid": 1,
                    "joint_vel": [0.0] * 7, "joint_vel_valid": 1,
                    "gripper_qpos": [0.0] * 2, "gripper_qpos_valid": 1,
                },
            }
        previous = {"nodes": [node("robot0", "robot", 0), node("obj", "object", 0)], "edges": []}
        current = {"nodes": [node("robot0", "robot", 0), node("obj", "object", 0)], "edges": []}
        previous["nodes"][0]["feature_vector"][4] = 1.0
        with self.assertRaises(ValueError):
            graph_tensors(previous, current)
        previous["nodes"][0]["feature_vector"][4] = 0.0
        prev_graph, curr_graph, prev_p, curr_p = graph_tensors(previous, current)
        self.assertEqual(tuple(prev_graph.node_features.shape), (2, 8))
        self.assertEqual(tuple(prev_graph.edge_contact.shape), (2, 2, 2))
        self.assertEqual(tuple(prev_graph.edge_relations.shape), (2, 2, 14))
        self.assertEqual(len(prev_p), 16)

    def test_streaming_shuffle_is_seeded_and_bounded(self):
        if self.torch is None:
            self.skipTest("torch is not installed in the local CPU environment")
        from scripts.phase3c.dataset import iter_shuffled_batches

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "joined.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for index in range(20):
                    handle.write(json.dumps({"sample_id": str(index), "split": "train", "task_id": 0}) + "\n")
            first = next(iter_shuffled_batches(path, batch_size=5, seed=1, split="train", shuffle_buffer=10))
            repeated = next(iter_shuffled_batches(path, batch_size=5, seed=1, split="train", shuffle_buffer=10))
            different = next(iter_shuffled_batches(path, batch_size=5, seed=2, split="train", shuffle_buffer=10))
        self.assertEqual([row["sample_id"] for row in first], [row["sample_id"] for row in repeated])
        self.assertNotEqual([row["sample_id"] for row in first], [row["sample_id"] for row in different])

    def test_collate_converts_numpy_semantic_features(self):
        if self.torch is None:
            self.skipTest("torch is not installed in the local CPU environment")
        from scripts.phase3c.contracts import PRIMARY_RELATIONS
        from scripts.phase3c.dataset import collate_phase3c

        def node(node_id, node_type, position):
            return {
                "node_id": node_id,
                "node_type": node_type,
                "feature_vector": [0.0] * 24,
                "features": {
                    "position": list(position), "position_valid": 1,
                    "joint_pos": [0.0] * 7, "joint_pos_valid": 1,
                    "joint_vel": [0.0] * 7, "joint_vel_valid": 1,
                    "gripper_qpos": [0.0] * 2, "gripper_qpos_valid": 1,
                },
            }

        def graph(step):
            return {
                "step": step,
                "nodes": [node("robot0", "robot", (0, 0, 0)), node("obj", "object", (0.1, 0, 0))],
                "edges": [],
            }

        class Store:
            def image(self, task_id, demo_key, step, view):
                return np.full(4, step + view, dtype=np.float32)

            def language(self, task_id, demo_key):
                return np.ones(4, dtype=np.float32)

        record = {
            "sample_id": "s0", "task_id": 0, "episode_id": "e0", "demo_key": "demo_0",
            "prev_step": 0, "current_step": 6, "target_step": 12,
            "graph_prev": graph(0), "graph_t": graph(6),
            "past_action_window": [[0.0] * 7 for _ in range(6)],
            "target": {
                "graph": graph(12),
                "relation_any_change": {name: 0 for name in PRIMARY_RELATIONS},
                "relation_valid": {name: 1 for name in PRIMARY_RELATIONS},
                "scene_max_displacement_m": 0.0,
            },
        }
        batch = collate_phase3c([record], Store())
        self.assertEqual(tuple(batch.v_history.shape), (1, 2, 2, 4))
        self.assertEqual(tuple(batch.target_v.shape), (1, 2, 4))
        self.assertEqual(tuple(batch.graph_prev.node_features.shape), (1, 2, 8))


if __name__ == "__main__":
    unittest.main(verbosity=2)
