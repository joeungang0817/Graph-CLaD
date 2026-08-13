from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.phase2d.input_clean import clean_graph
from scripts.phase2d.split_manifest import (
    build_manifest,
    repair_null_splits,
    split_lookup,
    validate_manifest,
)
from scripts.phase2d.temporal_holding import HoldingPolicy, annotate_temporal_holding


def _snapshot(step: int, contact: bool, closed: bool) -> dict:
    eef_x = 0.01 * step
    object_x = 0.20 + 0.01 * step
    return {
        "step": step,
        "robot_state": {
            "eef_pos": [eef_x, 0.0, 0.3],
            "gripper_qpos": [0.01, -0.01] if closed else [0.04, -0.04],
        },
        "object_states": [
            {"logical_id": "bowl", "pose": {"position": [object_x, 0.0, 0.3]}}
        ],
        "robot_contact_pairs": [["bowl", "robot0"]] if contact else [],
        "semantic_relation_records": {},
    }


class Phase2DRecoveredPipelineTest(unittest.TestCase):
    def test_temporal_holding_requires_motion_history_and_emits_release(self):
        snapshots = [
            _snapshot(0, False, False),
            _snapshot(1, True, True),
            _snapshot(2, True, True),
            _snapshot(3, True, True),
            _snapshot(4, False, False),
        ]
        events, intervals, traces = annotate_temporal_holding(
            snapshots, HoldingPolicy(history_frames=3)
        )
        self.assertEqual(traces["bowl"][1]["state"], "contact_candidate")
        self.assertEqual(traces["bowl"][2]["state"], "holding")
        self.assertEqual(traces["bowl"][4]["state"], "release")
        self.assertEqual(intervals, [{"object_id": "bowl", "start": 2, "end": 3, "length": 2}])
        self.assertTrue(any(event["to_state"] == "holding" for event in events))
        self.assertTrue(any(event["to_state"] == "release" for event in events))

    def test_input_clean_removes_task_relevance_metadata(self):
        graph = {
            "nodes": [
                {
                    "node_id": "bowl",
                    "features": {"is_object_of_interest": 0, "position_valid": 1},
                    "feature_vector": [0.0, 0.0, 0.0, 1.0, 0.0],
                }
            ]
        }
        cleaned = clean_graph(graph)
        self.assertNotIn("is_object_of_interest", cleaned["nodes"][0]["features"])
        self.assertEqual(cleaned["nodes"][0]["feature_vector"][4], 0.0)

    def test_manifest_is_episode_level_and_has_expected_control_counts(self):
        manifest = build_manifest(seed=0)
        report = validate_manifest(manifest, expected_count=150)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["in_task_counts"], {"train": 113, "validation": 19, "test": 18})
        self.assertEqual(report["task_generalization_counts"], {"train": 100, "test": 50})
        self.assertEqual(len(split_lookup(manifest)), 150)

    def test_legacy_null_split_repair_uses_demo_key_contract(self):
        manifest = {
            "episodes": [
                {
                    "task_id": 0,
                    "demo_key": "demo_0",
                    "demo_id": 0,
                    "in_task_split": "train",
                    "task_generalization_split": "train",
                }
            ]
        }
        record = {
            "task_id": 0,
            "demo_id": 0,
            "split": {"in_task": None, "task_generalization": None},
            "samples": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl.gz"
            output = Path(directory) / "output.jsonl.gz"
            with gzip.open(source, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            report = repair_null_splits(source, output, manifest)
            self.assertEqual(report["status"], "pass")
            with gzip.open(output, "rt", encoding="utf-8") as handle:
                repaired = json.loads(handle.readline())
            self.assertEqual(repaired["split"]["in_task"], "train")
            self.assertEqual(repaired["split"]["task_generalization"], "train")


if __name__ == "__main__":
    unittest.main(verbosity=2)
