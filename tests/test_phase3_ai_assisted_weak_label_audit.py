import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.phase3.analyze_ai_assisted_weak_labels import analyze


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "phase3_ai_assisted_weak_label_audit_v1.json"


def _point(step, source_x, target_x, contact, closed):
    return {
        "step": step,
        "source_position": [source_x, 0.0, 0.0],
        "target_position": [target_x, 0.0, 0.0],
        "contact": contact,
        "contact_valid": True,
        "gripper_qpos": [0.01, -0.01] if closed else [0.04, -0.04],
        "gripper_qpos_valid": 1,
        # These masked fields intentionally contradict some trajectories.
        "holding": False,
        "holding_state": "masked",
        "holding_confidence": "masked",
        "holding_evidence": {"object_followed_eef": False},
    }


def _row(audit_id, event_type, pattern):
    trajectory = []
    for step in range(7):
        source = step * 0.01
        if pattern == "onset":
            late = step >= 4
            target = source + 0.02 if late else 0.12
            trajectory.append(_point(step, source, target, late, late))
        elif pattern == "release":
            early = step <= 2
            target = source + 0.02 if early else 0.05
            trajectory.append(_point(step, source, target, early, early))
        elif pattern == "hard_negative":
            source = step * 0.03
            trajectory.append(_point(step, source, 0.15, True, False))
        else:
            trajectory.append(_point(step, source, source + 0.02, True, True))
    return {
        "audit_id": audit_id,
        "task_id": 0,
        "event_type": event_type,
        "episode_id": f"episode_{audit_id}",
        "sample_id": f"sample_{audit_id}",
        "event_cluster_id": f"cluster_{audit_id}",
        "start_step": 0,
        "target_step": 6,
        "edge": {"source": "robot0", "target": "object0"},
        "trajectory": trajectory,
        "trajectory_qa": {"status": "pass"},
    }


class AiAssistedWeakLabelAuditTest(unittest.TestCase):
    def test_masked_continuous_rubric_and_artifacts(self):
        rows = [
            _row("A1", "onset", "onset"),
            _row("A2", "release", "release"),
            _row("A3", "hard_negative", "hard_negative"),
            _row("A4", "hard_negative", "held"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence.jsonl.gz"
            with gzip.open(evidence, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            output = root / "analysis"
            summary = analyze(evidence, CONFIG, output)

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["clusters"], 4)
            self.assertEqual(summary["decision_counts"]["likely_pass"], 3)
            self.assertEqual(summary["decision_counts"]["likely_label_error"], 1)
            self.assertTrue((output / "ai_assisted_cluster_reviews_v1.csv").exists())
            self.assertTrue((output / "ai_assisted_sensitivity_groups_v1.json").exists())
            groups_payload = json.loads(
                (output / "ai_assisted_sensitivity_groups_v1.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                groups_payload["group_unit"],
                "sample_id|edge_source|edge_target",
            )
            groups = groups_payload["groups"]
            self.assertEqual(
                groups["likely_label_error"],
                ["sample_A4|robot0|object0"],
            )
            all_keys = [
                key
                for decision_keys in groups.values()
                for key in decision_keys
            ]
            self.assertEqual(len(all_keys), len(set(all_keys)))


if __name__ == "__main__":
    unittest.main()
