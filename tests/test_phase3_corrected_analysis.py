import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.phase3.analyze_corrected_predictions import analyze


class CorrectedAnalysisTest(unittest.TestCase):
    def test_clustered_bootstrap_reads_paired_prediction_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {}
            for model, positive_score in (("G1-late-action", 0.9), ("B1-v2", 0.6)):
                path = root / f"{model}.jsonl.gz"
                with gzip.open(path, "wt", encoding="utf-8") as handle:
                    for index in range(8):
                        target = int(index % 2 == 0)
                        score = positive_score if target else 0.1
                        row = {
                            "sample_id": f"sample{index}",
                            "episode_id": f"episode{index // 4}",
                            "event_cluster_id": f"event{index}",
                            "event_cluster_source": "explicit_event_id",
                            "edge_source": "robot0",
                            "edge_target": "object0",
                            "change_target": target,
                            "conditional_oracle_current_change_probability": score,
                            "conditional_oracle_current_change_prediction": int(score >= 0.5),
                            "current_target": int(index % 3 == 0),
                            "release_target": int(index % 6 == 0),
                            "future_prediction": int(index % 6 != 0),
                            "hard_negative": int(index in {1, 5}),
                        }
                        handle.write(json.dumps(row))
                        handle.write("\n")
                artifacts[model] = path
            result = {
                "results": [
                    {
                        "fold": "test_task1",
                        "seed": 0,
                        "comparison_id": model,
                        "prediction_artifacts": {
                            "natural_test.correct": {"path": str(path)}
                        },
                    }
                    for model, path in artifacts.items()
                ]
            }
            result_path = root / "result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")

            report = analyze(result_path, replicates=25, seed=7)

            self.assertEqual(
                report["comparison"], "G1-late-action_minus_B1-v2"
            )
            self.assertEqual(
                report["hierarchical_bootstrap"]["event_pr_auc"]["replicates"],
                25,
            )
            self.assertEqual(
                report["resampling_hierarchy"],
                ["task_fold", "episode", "event_cluster"],
            )


if __name__ == "__main__":
    unittest.main()
