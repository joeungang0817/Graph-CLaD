from __future__ import annotations

import json
import gzip
import copy
import tempfile
import unittest
import hashlib
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

    def test_semantic_store_verifies_shards_and_orientation_attestation(self):
        if self.torch is None:
            self.skipTest("torch is not installed in the local CPU environment")
        from scripts.phase3c.dataset import SemanticFeatureStore

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "0" / "demo_0.npz"
            shard.parent.mkdir(parents=True)
            np.savez(
                shard,
                steps=np.asarray([0], dtype=np.int32),
                view0=np.zeros((1, 4), dtype=np.float32),
                view1=np.zeros((1, 4), dtype=np.float32),
                language=np.zeros(4, dtype=np.float32),
            )
            shard_sha = hashlib.sha256(shard.read_bytes()).hexdigest()
            qa = root / "qa" / "determinism.json"
            qa.parent.mkdir(parents=True)
            qa.write_text("{}\n", encoding="utf-8")
            sheet = root / "qa" / "orientation_contact_sheet.png"
            sheet.write_bytes(b"reviewed image")
            attestation = {
                "schema": "phase3c-camera-orientation-human-attestation.v1",
                "status": "pass",
                "accepted_existing_semantic_store": True,
                "source_qa": "determinism.json",
                "source_qa_sha256": hashlib.sha256(qa.read_bytes()).hexdigest(),
                "source_contact_sheet": "orientation_contact_sheet.png",
                "source_contact_sheet_sha256": hashlib.sha256(
                    sheet.read_bytes()
                ).hexdigest(),
            }
            (root / "qa" / "orientation_human_attestation.json").write_text(
                json.dumps(attestation), encoding="utf-8"
            )
            manifest = {
                "schema": "phase3c-semantic-feature-store.v2",
                "decisionnce": {"feature_dim": 4},
                "index": {},
                "shards": 1,
                "shard_sha256": {"0/demo_0.npz": shard_sha},
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            store = SemanticFeatureStore(root)
            self.assertEqual(store.verify_integrity()["verified_shards"], 1)
            store.close()
            shard.write_bytes(shard.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                SemanticFeatureStore(root).verify_integrity()

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

    def test_normalization_ignores_invalid_position_and_geometry_fill_values(self):
        if self.torch is None:
            self.skipTest("torch is not installed in the local CPU environment")
        from scripts.phase3c.dataset import fit_normalization

        def node(node_id, node_type, position, valid):
            return {
                "node_id": node_id,
                "node_type": node_type,
                "feature_vector": [0.0] * 24,
                "features": {
                    "position": list(position), "position_valid": int(valid),
                    "joint_pos": [0.0] * 7, "joint_pos_valid": 1,
                    "joint_vel": [0.0] * 7, "joint_vel_valid": 1,
                    "gripper_qpos": [0.0] * 2, "gripper_qpos_valid": 1,
                },
            }

        def graph(fill):
            return {
                "nodes": [
                    node("robot0", "robot", (1.0, 2.0, 3.0), True),
                    node("obj", "object", (fill, fill, fill), False),
                ],
                "edges": [{
                    "source": "robot0", "target": "obj",
                    "features": {
                        "relative_position": [fill, fill, fill],
                        "distance": fill,
                        "distance_valid": 0,
                    },
                }],
            }

        first = {"graph_prev": graph(0.0), "graph_t": graph(0.0)}
        second = copy.deepcopy(first)
        second["graph_prev"] = graph(1e9)
        second["graph_t"] = graph(-1e9)
        stats_first = fit_normalization([first])
        stats_second = fit_normalization([second])
        self.assertEqual(stats_first.node_mean, stats_second.node_mean)
        self.assertEqual(stats_first.edge_geometry_mean, stats_second.edge_geometry_mean)


if __name__ == "__main__":
    unittest.main(verbosity=2)
