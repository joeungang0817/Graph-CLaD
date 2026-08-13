import sys
import types
import unittest

import numpy as np


try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_module = types.ModuleType("torch")
    nn_module = types.ModuleType("torch.nn")

    class Module:
        pass

    nn_module.Module = Module
    torch_module.nn = nn_module
    sys.modules["torch"] = torch_module
    sys.modules["torch.nn"] = nn_module


from scripts.phase3.offline_probe import (
    _checkpoint_selection,
    _holding_metrics,
    attach_training_action_donors,
)


class CorrectedProtocolPureTest(unittest.TestCase):
    def test_holding_metrics_store_calibration_and_both_event_contracts(self):
        current = np.asarray([[[1.0, 0.0], [1.0, 1.0]]], dtype=np.float32)
        future = np.asarray([[[1.0, 1.0], [1.0, 0.0]]], dtype=np.float32)
        future_score = np.asarray([[[0.9, 0.8], [0.9, 0.2]]], dtype=np.float32)
        current_score = np.asarray([[[0.9, 0.1], [0.9, 0.9]]], dtype=np.float32)
        valid = np.ones_like(current)
        outputs = {
            "current_true": current,
            "current_score": current_score,
            "future_true": future,
            "future_score": future_score,
            "current_valid": valid,
            "future_valid": valid,
            "changed": (current != future).astype(np.float32),
        }

        metrics = _holding_metrics(outputs, ("contact", "holding"), 0.5, 0.5)

        self.assertEqual(metrics["change_event"]["f1"], 1.0)
        self.assertEqual(metrics["end_to_end_change_event"]["f1"], 1.0)
        self.assertIsNotNone(metrics["change_event"]["brier_score"])
        self.assertIsNotNone(metrics["change_event"]["ece"])

    def test_corrected_checkpoint_criterion_is_event_pr_auc(self):
        score, name = _checkpoint_selection(
            {
                "holding": {
                    "conditional_oracle_current_change_event": {
                        "pr_auc": 0.73
                    }
                }
            },
            criterion="holding_event_pr_auc",
        )
        self.assertEqual(score, 0.73)
        self.assertEqual(name, "holding_event_pr_auc")

    def test_matched_shuffle_preserves_task_marginal_and_changes_episode(self):
        records = []
        for index, episode in enumerate(("a", "a", "b", "b", "c", "c", "d", "d")):
            records.append(
                {
                    "sample_id": f"s{index}",
                    "suite": "libero_spatial",
                    "task_id": 1,
                    "episode_id": episode,
                    "actions": [float(index), 0.0, float(index % 2), 0.0],
                    "action_steps": 2,
                    "action_dim": 2,
                    "action_step_mask": [1.0, 1.0],
                    "current_labels": [[1.0, float(index % 2)]],
                    "current_valid": [[1.0, 1.0]],
                }
            )

        decorated, qa = attach_training_action_donors(
            records, relations=("contact", "holding")
        )

        self.assertTrue(qa["marginal_preserved"])
        self.assertEqual(qa["different_episode_fraction"], 1.0)
        self.assertEqual(
            sorted(tuple(row["training_donor_actions"]) for row in decorated),
            sorted(tuple(row["actions"]) for row in records),
        )
        self.assertTrue(
            all(
                row["episode_id"] != row["training_action_donor_episode_id"]
                for row in decorated
            )
        )


if __name__ == "__main__":
    unittest.main()
