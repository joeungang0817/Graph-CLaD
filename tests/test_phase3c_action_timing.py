import unittest

from scripts.phase3c.validate_action_timing import max_abs_state_error


class Phase3CActionTimingTest(unittest.TestCase):
    def test_max_abs_state_error(self):
        self.assertAlmostEqual(max_abs_state_error([1.0, 2.0], [1.25, 1.5]), 0.5)

    def test_shape_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            max_abs_state_error([1.0], [1.0, 2.0])

    def test_nonfinite_state_fails(self):
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            max_abs_state_error([float("nan")], [0.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
