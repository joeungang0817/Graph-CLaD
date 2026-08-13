import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.phase3.weak_label_audit_viewer import review_summary, save_review_bundle


class WeakLabelAuditViewerTest(unittest.TestCase):
    def test_summary_and_atomic_review_save(self):
        rows = [
            {
                "audit_id": "HQA-001",
                "task_id": "0",
                "event_type": "onset",
                "sample_id": "s1",
                "episode_id": "e1",
                "start_step": "0",
                "target_step": "6",
                "edge_source": "robot0",
                "edge_target": "object0",
                "reviewer_decision": "pass",
                "label_error_type": "",
                "reviewer_notes": "looks consistent",
            },
            {
                "audit_id": "HQA-002",
                "task_id": "0",
                "event_type": "release",
                "sample_id": "s2",
                "episode_id": "e2",
                "start_step": "1",
                "target_step": "7",
                "edge_source": "robot0",
                "edge_target": "object0",
                "reviewer_decision": "label_error",
                "label_error_type": "false_release",
                "reviewer_notes": "contact never ended",
            },
        ]
        summary = review_summary(rows)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["reviewed"], 2)
        self.assertEqual(summary["pass_rate"], 0.5)
        self.assertEqual(summary["error_type_counts"], {"false_release": 1})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"review_summary": {"status": "pending"}}
            (root / "holding_weak_label_audit_manifest_v1.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            saved = save_review_bundle(root, rows)
            self.assertEqual(saved["status"], "complete")
            with (root / "holding_weak_label_audit_review_v1.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                stored = list(csv.DictReader(handle))
            self.assertEqual(stored[1]["label_error_type"], "false_release")
            updated = json.loads(
                (root / "holding_weak_label_audit_manifest_v1.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(updated["review_summary"]["reviewed"], 2)


if __name__ == "__main__":
    unittest.main()
