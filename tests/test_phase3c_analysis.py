from __future__ import annotations

import unittest
import gzip
import json
import tempfile
from pathlib import Path

from scripts.phase3c.analyze_core import (
    hierarchical_bootstrap_difference,
    load_predictions,
    score_rows,
)
from scripts.phase3c.contracts import PRIMARY_RELATIONS


def _row(sample_id: str, task_id: int, strength: float, *, flip: bool) -> dict:
    target = [(index + int(flip)) % 2 for index, _ in enumerate(PRIMARY_RELATIONS)]
    logits = [strength if value else -strength for value in target]
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "episode_id": f"task{task_id}_demo{ord(sample_id[0]) % 2}",
        "relation_logits": logits,
        "target_relation_change": target,
        "target_relation_mask": [1] * len(PRIMARY_RELATIONS),
        "scene_motion": 0.1,
        "target_scene_motion": 0.1,
    }


class Phase3CAnalysisTest(unittest.TestCase):
    def test_family_metric_flows_through_analysis_and_bootstrap(self):
        candidate = [
            _row("a", 0, 4.0, flip=False),
            _row("b", 0, 4.0, flip=True),
            _row("c", 1, 4.0, flip=False),
            _row("d", 1, 4.0, flip=True),
        ]
        baseline = [
            _row("a", 0, 0.1, flip=False),
            _row("b", 0, 0.1, flip=True),
            _row("c", 1, 0.1, flip=False),
            _row("d", 1, 0.1, flip=True),
        ]
        scored = score_rows(candidate)
        self.assertIn("family_macro_pr_auc", scored["relation"])
        difference = hierarchical_bootstrap_difference(
            candidate,
            baseline,
            metric="family_macro_pr_auc",
            replicates=10,
            seed=0,
        )
        self.assertEqual(difference["metric"], "family_macro_pr_auc")
        self.assertEqual(difference["common_rows"], 4)
        self.assertEqual(difference["resampling_units"], ["task", "episode"])

    def test_bootstrap_rejects_unpaired_sample_sets(self):
        candidate = [_row("a", 0, 4.0, flip=False), _row("b", 0, 4.0, flip=True)]
        baseline = [_row("a", 0, 4.0, flip=False)]
        with self.assertRaisesRegex(ValueError, "sample_id sets differ"):
            hierarchical_bootstrap_difference(candidate, baseline, replicates=2)

    def test_per_relation_prediction_artifact_pivots_with_fixed_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for index, relation in enumerate(PRIMARY_RELATIONS):
                    handle.write(json.dumps({
                        "schema": "phase3c-core-prediction.v2",
                        "sample_id": "s0",
                        "task_id": 0,
                        "episode_id": "e0",
                        "relation": relation,
                        "logit": float(index),
                        "target": index % 2,
                        "evaluated": True,
                        "threshold": 0.75,
                        "scene_motion": 0.1,
                        "target_scene_motion": 0.2,
                    }) + "\n")
            rows = load_predictions(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["fixed_thresholds"], [0.75] * len(PRIMARY_RELATIONS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
