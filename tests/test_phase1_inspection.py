"""Pure schema tests for the Phase 1 LIBERO inspection helpers."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inspect_libero_state import (  # noqa: E402
    extract_object_states,
    extract_robot_state,
    make_snapshot,
)


class FakeState:
    def __init__(self, position, quaternion, joint_state=None):
        self.position = position
        self.quaternion = quaternion
        self.joint_state = joint_state

    def get_geom_state(self):
        return {"pos": self.position, "quat": self.quaternion}

    def get_joint_state(self):
        return self.joint_state

    def is_open(self):
        return False

    def is_close(self):
        return True


class FakeEnv:
    def __init__(self):
        self.env = self
        self.objects_dict = {"red_cube": object()}
        self.fixtures_dict = {"target_zone": object()}
        self.object_sites_dict = {}
        self.object_states_dict = {
            "red_cube": FakeState([0.1, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0]),
            "target_zone": FakeState([0.4, 0.5, 0.6], [1.0, 0.0, 0.0, 0.0]),
        }
        self.obj_of_interest = ["red_cube"]
        self.obj_body_id = {"red_cube": 3, "target_zone": 4}


class Phase1InspectionTest(unittest.TestCase):
    def test_robot_fields_are_explicit_and_missing_values_are_visible(self):
        result = extract_robot_state(
            {
                "robot0_eef_pos": [1.0, 2.0, 3.0],
                "robot0_eef_quat": [1.0, 0.0, 0.0, 0.0],
                "robot0_gripper_qpos": [0.1, 0.2],
            }
        )
        self.assertEqual(result["eef_pos"], [1.0, 2.0, 3.0])
        self.assertEqual(result["gripper_qpos"], [0.1, 0.2])
        self.assertIsNone(result["joint_pos"])

    def test_object_identity_is_logical_not_runtime_body_id(self):
        result = extract_object_states(FakeEnv())
        by_id = {entry["logical_id"]: entry for entry in result}
        self.assertEqual(by_id["red_cube"]["body_id"], 3)
        self.assertTrue(by_id["red_cube"]["is_object_of_interest"])
        self.assertFalse(by_id["target_zone"]["is_object_of_interest"])
        self.assertEqual(by_id["red_cube"]["pose"]["position"], [0.1, 0.2, 0.3])

    def test_snapshot_is_json_serializable(self):
        snapshot = make_snapshot(
            FakeEnv(),
            {"robot0_eef_pos": [0.0, 0.0, 0.0]},
            step=0,
            robot_prefix="robot0",
        )
        json.dumps(snapshot)
        self.assertEqual(snapshot["step"], 0)
        self.assertIn("object_states", snapshot)


if __name__ == "__main__":
    unittest.main(verbosity=2)
