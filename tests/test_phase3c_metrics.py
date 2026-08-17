from __future__ import annotations

import unittest

import numpy as np

from scripts.phase3c.metrics import (
    _average_precision,
    _best_f1,
    aggregate_relation_families,
    evaluate_motion,
    evaluate_relation_predictions,
    no_change_fpr,
)


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
        self.assertEqual(result["moving_count"], 1)
        self.assertAlmostEqual(result["moving_mae"], 2.0)

    def test_average_precision_groups_tied_scores(self):
        first = _average_precision(np.asarray([1, 0]), np.asarray([0.5, 0.5]))
        reversed_rows = _average_precision(np.asarray([0, 1]), np.asarray([0.5, 0.5]))
        self.assertAlmostEqual(first, 0.5)
        self.assertAlmostEqual(reversed_rows, 0.5)

    def test_best_f1_uses_highest_threshold_on_tie(self):
        value, threshold = _best_f1(
            np.asarray([0, 1]),
            np.asarray([0.1, 0.9]),
            [0.2, 0.8],
        )
        self.assertEqual(value, 1.0)
        self.assertEqual(threshold, 0.8)

    def test_fixed_thresholds_are_not_reoptimized_on_test(self):
        result = evaluate_relation_predictions(
            [[0.0], [0.2], [0.4], [0.6]],
            [[0], [0], [1], [1]],
            [[1], [1], [1], [1]],
            fixed_thresholds=[0.9],
        )
        self.assertEqual(result["threshold_source"], "validation_fixed")
        self.assertEqual(result["per_relation"]["left"]["threshold"], 0.9)
        self.assertEqual(no_change_fpr([[0.0], [2.0]], [[0], [0]], [[1], [1]], [0.5])["mean"], 1.0)

    def test_fold_specific_row_thresholds_are_supported(self):
        result = evaluate_relation_predictions(
            [[0.0], [0.2], [0.4], [0.6]],
            [[0], [0], [1], [1]],
            [[1], [1], [1], [1]],
            fixed_thresholds=[[0.4], [0.4], [0.6], [0.6]],
        )
        self.assertEqual(result["threshold_source"], "validation_fixed_per_row")
        self.assertIsNone(result["per_relation"]["left"]["threshold"])
        self.assertEqual(result["per_relation"]["left"]["threshold_values"], [0.4, 0.6])

    def test_inverse_relations_are_reported_once_per_family(self):
        report = aggregate_relation_families(
            {
                "left": {"valid_count": 10, "pr_auc": 1.0, "f1": 0.8, "brier": 0.1, "ece_10bin": 0.2},
                "right": {"valid_count": 10, "pr_auc": 0.0, "f1": 0.4, "brier": 0.3, "ece_10bin": 0.4},
                "contact": {"valid_count": 5, "pr_auc": 0.75, "f1": 0.5, "brier": 0.2, "ece_10bin": 0.1},
            }
        )
        self.assertAlmostEqual(report["per_family"]["horizontal"]["pr_auc"], 0.5)
        self.assertAlmostEqual(report["family_macro_pr_auc"], 0.625)
        self.assertEqual(report["per_family"]["horizontal"]["relations"], ["left", "right"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
