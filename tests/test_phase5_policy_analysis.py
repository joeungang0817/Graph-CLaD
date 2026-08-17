from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.phase5.analyze_policy_pilot import analyze


def _runtime(path: Path, *, arm: str, loss: float) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "phase5-policy-run.v1",
                "status": "completed",
                "arm": arm,
                "structured_model_id": "C3-RelMPNN-PastAct" if arm == "graph" else None,
                "policy_manifest_sha256": "m",
                "base_checkpoint_sha256": "b",
                "seed": 0,
                "updates": 20,
                "batch_size": 64,
                "best_validation_ddpm_loss": loss,
                "foresight_sensitivity": {
                    "zeroed_mean_abs_delta": loss,
                    "shuffled_mean_abs_delta": loss * 2,
                },
            }
        ),
        encoding="utf-8",
    )


class Phase5PolicyAnalysisTest(unittest.TestCase):
    def test_matched_arm_difference_is_serialized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            semantic = root / "semantic.json"
            graph = root / "graph.json"
            _runtime(semantic, arm="semantic", loss=2.0)
            _runtime(graph, arm="graph", loss=1.5)
            result = analyze({"semantic_runtime": str(semantic), "graph_runtime": str(graph)})
            self.assertEqual(result["validation_ddpm_loss"]["difference_graph_minus_semantic"], -0.5)
            self.assertEqual(result["graph_arm"], "C3-RelMPNN-PastAct")


if __name__ == "__main__":
    unittest.main()
