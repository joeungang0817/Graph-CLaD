import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.phase3.build_weak_label_audit import build_audit


def _relation(value, evidence=None):
    return {"value": value, "valid": True, "evidence": evidence or {}}


def _graph(holding, contact, step):
    return {
        "nodes": [
            {
                "node_id": "robot0",
                "node_type": "robot",
                "features": {
                    "position": [step * 0.1, 0, 0],
                    "position_valid": 1,
                    "gripper_qpos": [0.01, -0.01],
                    "gripper_qpos_valid": 1,
                },
            },
            {
                "node_id": "object0",
                "node_type": "object",
                "features": {"position": [step * 0.1 + 0.02, 0, 0], "position_valid": 1},
            },
        ],
        "edges": [
            {
                "source": "robot0",
                "target": "object0",
                "node_type_pair": ["robot", "object"],
                "features": {"relative_position": [0.02, 0, 0], "distance": 0.02},
                "relations": {
                    "holding": _relation(holding, {"relative_pose_stable": True}),
                    "contact": _relation(contact),
                },
            }
        ],
    }


def _sample(task_id, episode, event_type):
    if event_type == "onset":
        current_holding, future_holding = False, True
        current_contact, future_contact = True, True
    elif event_type == "release":
        current_holding, future_holding = True, False
        current_contact, future_contact = True, False
    else:
        current_holding, future_holding = False, False
        current_contact, future_contact = True, True

    return {
        "suite": "libero_spatial",
        "task_id": task_id,
        "episode_id": episode,
        "split": "test",
        "start_step": 0,
        "target_step": 6,
        "tau": 6,
        "graph_t": _graph(current_holding, current_contact, 0),
        "graph_target": _graph(future_holding, future_contact, 6),
        "action_window": [{"action": [0.0] * 7} for _ in range(6)],
    }


def _trajectory_samples(task_id, episode, event_type):
    def state(step):
        if event_type == "onset":
            return step == 6, True
        if event_type == "release":
            return step < 6, step < 6
        return False, True

    rows = []
    for step in range(6):
        holding_t, contact_t = state(step)
        holding_f, contact_f = state(step + 1)
        rows.append(
            {
                "suite": "libero_spatial",
                "task_id": task_id,
                "episode_id": episode,
                "split": "test",
                "start_step": step,
                "target_step": step + 1,
                "tau": 1,
                "graph_t": _graph(holding_t, contact_t, step),
                "graph_target": _graph(holding_f, contact_f, step + 1),
                "action_window": [{"action": [0.0] * 7}],
            }
        )
    return rows


class WeakLabelAuditTest(unittest.TestCase):
    def test_builds_task_event_balanced_review_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            output = Path(directory) / "audit"
            for task_id in (0, 1, 2):
                task_root = root / f"task{task_id}"
                task_root.mkdir(parents=True)
                samples = []
                for event_type in ("onset", "release", "hard_negative"):
                    for index in range(2):
                        episode = f"task{task_id}_{event_type}_{index}"
                        samples.append(_sample(task_id, episode, event_type))
                        samples.extend(
                            _trajectory_samples(task_id, episode, event_type)
                        )
                path = task_root / f"phase2d_task{task_id}_graph_dataset.jsonl.gz"
                with gzip.open(path, "wt", encoding="utf-8") as handle:
                    handle.write(json.dumps({"samples": samples}))
                    handle.write("\n")

            manifest = build_audit(root, output, per_event=2)

            self.assertEqual(manifest["status"], "ready_for_manual_review")
            self.assertEqual(manifest["selected_items"], 18)
            self.assertTrue(Path(manifest["evidence"]).exists())
            self.assertTrue(Path(manifest["review_csv"]).exists())
            self.assertEqual(manifest["review_summary"]["status"], "pending")
            self.assertEqual(manifest["trajectory_qa"]["status"], "pass")
            with gzip.open(manifest["evidence"], "rt", encoding="utf-8") as handle:
                first = json.loads(next(handle))
            self.assertEqual(first["trajectory_qa"]["status"], "pass")
            self.assertEqual(len(first["trajectory"]), 7)

            full_output = Path(directory) / "audit_all_clusters"
            full_manifest = build_audit(root, full_output, per_event=None)
            self.assertTrue(full_manifest["all_event_clusters"])
            self.assertEqual(full_manifest["selected_items"], 18)
            self.assertEqual(full_manifest["expected_items"], 18)
            self.assertEqual(full_manifest["trajectory_qa"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
