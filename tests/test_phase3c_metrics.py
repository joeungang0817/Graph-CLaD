from __future__ import annotations

import unittest

import numpy as np

from scripts.phase3c.metrics import evaluate_motion, evaluate_relation_predictions


class Phase3CMetricsTest(unittest.TestCase):
    def test_unknown_relation_is_excluded(self):
        result = evaluate_relation_predictions(
            [[-5.0, 5.0], [5.0, -5.0], [0.0, 0.0]],
            [[0, 1], [1, 0], [1, 1]],
            [[1, 1], [1, 1], [0, 0]],
        )
        self.assertEqual(result["per_relation"]["left"]["valid_count"], 2)
        self.assertIsNotNone(result["macro_pr_auc"])

    def test_single_class_returns_none(self):
        result = evaluate_relation_predictions([[0.0]], [[1]], [[1]])
        self.assertIsNone(result["macro_pr_auc"])
        self.assertIsNone(result["macro_f1"])

    def test_motion_metrics(self):
        result = evaluate_motion([1.0, 3.0], [0.0, 1.0])
        self.assertAlmostEqual(result["mae"], 1.5)
        self.assertAlmostEqual(result["rmse"], np.sqrt(2.5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
