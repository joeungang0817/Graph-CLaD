import unittest

from scripts.phase3c.validate_action_timing import (
    _probe_steps,
    action_timing_status,
    max_abs_state_error,
)


class Phase3CActionTimingTest(unittest.TestCase):
    def test_uniform_probe_spans_trajectory(self):
        self.assertEqual(_probe_steps(101, 3), [0, 50, 99])
        self.assertEqual(_probe_steps(4, 10), [0, 1, 2])
        self.assertEqual(_probe_steps(101, 3, "head"), [0, 1, 2])

    def test_max_abs_state_error(self):
        self.assertAlmostEqual(max_abs_state_error([1.0, 2.0], [1.25, 1.5]), 0.5)

    def test_shape_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            max_abs_state_error([1.0], [1.0, 2.0])

    def test_nonfinite_state_fails(self):
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            max_abs_state_error([float("nan")], [0.0])

    def test_configured_tolerance_is_a_hard_gate(self):
        rows = [{"within_tolerance": True}, {"within_tolerance": False}]
        self.assertEqual(action_timing_status(rows, [], 1e-5), "fail")
        self.assertEqual(action_timing_status(rows, [], None), "pass")


if __name__ == "__main__":
    unittest.main(verbosity=2)
