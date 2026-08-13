"""Pure tests for the Phase 2 graph boundary."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.graph_extractor import (  # noqa: E402
    FEATURE_DIM,
    build_graph_sequence,
    extract_graph_snapshot,
)


def snapshot_fixture():
    return {
        "step": 4,
        "robot_state": {
            "eef_pos": [0.0, 0.0, 1.0],
            "eef_quat": [1.0, 0.0, 0.0, 0.0],
            "gripper_qpos": [0.1, -0.1],
            "joint_pos": [0.0] * 7,
            "joint_vel": [0.0] * 7,
            "available_keys": ["robot0_eef_pos"],
        },
        "object_states": [
            {
                "logical_id": "red_cube",
                "node_type": "object",
                "is_object_of_interest": True,
                "body_id": 3,
                "pose": {"position": [0.3, 0.0, 0.9], "quaternion": [1, 0, 0, 0]},
                "joint_state": {"error": "object has no joint"},
                "is_open": {"error": "unsupported"},
                "is_close": None,
            },
            {
                "logical_id": "target_zone",
                "node_type": "fixture",
                "is_object_of_interest": False,
                "body_id": 8,
                "pose": {"position": [0.6, 0.0, 0.9], "quaternion": [1, 0, 0, 0]},
                "joint_state": None,
                "is_open": False,
                "is_close": True,
            },
        ],
    }


class Phase2GraphExtractorTest(unittest.TestCase):
    def test_graph_is_directed_and_body_id_is_not_identity(self):
        graph = extract_graph_snapshot(snapshot_fixture())
        self.assertEqual(graph["audit"]["node_count"], 3)
        self.assertEqual(graph["audit"]["edge_count"], 6)
        self.assertFalse(graph["audit"]["body_id_used_as_identity"])
        self.assertEqual(graph["nodes"][1]["node_id"], "red_cube")
        self.assertEqual(graph["nodes"][1]["runtime_body_id"], 3)
        robot_to_cube = next(
            edge for edge in graph["edges"]
            if edge["source"] == "robot0" and edge["target"] == "red_cube"
        )
        for actual, expected in zip(
            robot_to_cube["features"]["relative_position"], [0.3, 0.0, -0.1]
        ):
            self.assertAlmostEqual(actual, expected)

    def test_predicate_errors_are_unknown_not_false(self):
        graph = extract_graph_snapshot(snapshot_fixture())
        cube = next(node for node in graph["nodes"] if node["node_id"] == "red_cube")
        self.assertIsNone(cube["predicate_state"]["is_open"]["value"])
        self.assertEqual(cube["predicate_state"]["is_open"]["valid"], 0)
        self.assertEqual(cube["predicate_state"]["is_open"]["status"], "error")
        self.assertEqual(graph["audit"]["predicate_error_count"], 2)

    def test_missing_position_keeps_node_but_removes_its_edges(self):
        payload = snapshot_fixture()
        payload["object_states"][1]["pose"] = None
        graph = extract_graph_snapshot(payload)
        target = next(node for node in graph["nodes"] if node["node_id"] == "target_zone")
        self.assertEqual(target["features"]["position_valid"], 0)
        self.assertIn("target_zone", graph["audit"]["missing_position_node_ids"])
        self.assertTrue(all(
            "target_zone" not in (edge["source"], edge["target"])
            for edge in graph["edges"]
        ))

    def test_sequence_preserves_time_order_and_feature_shape(self):
        payload = {"suite": "fixture", "snapshots": [snapshot_fixture()]}
        spec = {"spec_version": "phase2.v1"}
        result = build_graph_sequence(payload, spec)
        self.assertEqual(result["temporal_audit"]["steps"], [4])
        self.assertFalse(result["temporal_audit"]["future_fields_included"])
        self.assertTrue(all(
            len(node["feature_vector"]) == FEATURE_DIM
            for node in result["graphs"][0]["nodes"]
        ))
        json.dumps(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
