from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.phase5.build_policy_manifest import build_policy_manifest


def _actions(value: float) -> list[list[float]]:
    return [[value] * 7 for _ in range(6)]


class Phase5PolicyManifestTest(unittest.TestCase):
    def test_future_action_is_label_and_target_graph_is_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            joined = root / "joined.jsonl.gz"
            source = root / "source.jsonl.gz"
            output = root / "policy.jsonl.gz"
            qa = root / "policy.qa.json"
            row = {
                "schema": "phase3c-joined-sample.v1",
                "sample_id": "task0_ep_prev0_cur6_next12_tau6",
                "task_id": 0,
                "episode_id": "ep",
                "demo_key": "demo0",
                "split": "train",
                "tau": 6,
                "prev_step": 0,
                "current_step": 6,
                "target_step": 12,
                "past_action_window": _actions(0.0),
                "graph_prev": {"nodes": [], "edges": []},
                "graph_t": {"nodes": [], "edges": []},
                "target": {"graph": {"future": "must-not-be-copied"}},
            }
            with gzip.open(joined, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
                # Keep an excluded row last to catch stale loop-variable use in
                # the emitted split counters.
                excluded = dict(row)
                excluded["sample_id"] = "task0_ep_prev1_cur7_next13_tau6"
                excluded["split"] = "test"
                excluded["prev_step"] = 1
                excluded["current_step"] = 7
                excluded["target_step"] = 13
                handle.write(json.dumps(excluded) + "\n")
            source_row = {
                "split": "train",
                "task_id": 0,
                "episode_id": "ep",
                "start_step": 6,
                "tau": 6,
                "action_window": _actions(1.0),
            }
            with gzip.open(source, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(source_row) + "\n")

            report = build_policy_manifest(
                {
                    "stage2_policy_manifest": {
                        "joined_manifest": str(joined),
                        "output": str(output),
                        "qa_output": str(qa),
                        "task_id": 0,
                        "phase2d_by_task": {"0": str(source)},
                    }
                }
            )
            self.assertEqual(report["counters"]["selected_rows"], 1)
            self.assertEqual(report["counters"]["train_rows"], 1)
            self.assertEqual(report["counters"]["validation_rows"], 0)
            self.assertEqual(report["counters"]["test_rows"], 0)
            self.assertFalse(report["future_action_in_model_input"])
            with gzip.open(output, "rt", encoding="utf-8") as handle:
                emitted = json.loads(handle.readline())
            self.assertIn("action_target_window", emitted)
            self.assertNotIn("target", emitted)
            self.assertNotIn("future_action", emitted)
            self.assertNotIn("target_graph", emitted)


if __name__ == "__main__":
    unittest.main()
