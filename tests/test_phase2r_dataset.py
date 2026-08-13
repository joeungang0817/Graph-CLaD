from scripts.build_phase2r_dataset import build_dataset, relation_labels
from scripts.phase2r_relation_handlers import transform_position_to_robot_base


def _node(node_id, position):
    return {
        "node_id": node_id,
        "features": {"position": position, "position_valid": 1},
    }


def test_inverse_geometry_relations():
    left = _node("left", [0.0, 0.0, 0.0])
    right = _node("right", [0.1, 0.0, 0.0])
    snapshot = {"contact_pairs": []}
    forward = relation_labels(left, right, snapshot)
    backward = relation_labels(right, left, snapshot)
    assert forward["right"]["value"] is True
    assert backward["left"]["value"] is True
    assert forward["contact"]["valid"] == 1


def test_unknown_predicates_are_not_false():
    left = _node("left", [0.0, 0.0, 0.0])
    right = _node("right", [0.1, 0.0, 0.0])
    labels = relation_labels(left, right, {})
    assert labels["contact"]["value"] is None
    assert labels["contact"]["valid"] == 0
    assert labels["open"]["value"] is None
    assert labels["open"]["valid"] == 0


def test_episode_split_and_nonzero_actions_are_audited():
    def snapshot(step, x):
        return {
            "step": step,
            "robot_state": {},
            "object_states": [
                {"logical_id": "obj", "node_type": "object", "pose": {"position": [x, 0, 0]}, "body_id": 1}
            ],
            "contact_pairs": [],
            "robot_base_pose": {
                "position": [0.0, 0.0, 0.0],
                "quaternion": [1.0, 0.0, 0.0, 0.0],
            },
        }

    episode = {
        "episode_id": "episode-a",
        "task_id": 0,
        "snapshots": [snapshot(i, 0.0 if i < 2 else 0.1) for i in range(3)],
        "actions": [
            {"action": [0.1], "from_step": 0, "to_step": 1},
            {"action": [0.1], "from_step": 1, "to_step": 2},
        ],
    }
    dataset = build_dataset({"capture_status": "test", "episodes": [episode]}, tau=1)
    assert dataset["audit"]["sample_count"] == 2
    assert dataset["audit"]["nonzero_action_count"] == 2
    assert dataset["split"]["assignments"]["episode-a"] == "train"
    assert dataset["coordinate_frame"] == "robot_base"


def test_robot_base_transform_subtracts_and_rotates():
    transformed = transform_position_to_robot_base(
        [1.0, 2.0, 3.0],
        {"position": [0.5, 1.0, 1.5], "quaternion": [1.0, 0.0, 0.0, 0.0]},
    )
    assert transformed == [0.5, 1.0, 1.5]


def test_semantic_wrapper_record_is_used_and_unknown_is_preserved():
    source = _node("obj", [0.0, 0.0, 0.0])
    target = _node("site", [0.0, 0.0, 0.1])
    labels = relation_labels(
        source,
        target,
        {
            "contact_pairs": [],
            "semantic_relation_records": {
                "on": {"obj->site": {"value": True, "valid": 1, "definition_version": "test"}},
                "inside": {"obj->site": {"value": None, "valid": 0, "status": "error"}},
            },
        },
    )
    assert labels["on"]["value"] is True
    assert labels["inside"]["valid"] == 0
    assert labels["open"]["valid"] == 0


def test_holding_uses_calibrated_closed_gripper_threshold():
    robot = _node("robot0", [0.0, 0.0, 0.0])
    obj = _node("object", [0.0, 0.0, 0.1])
    labels = relation_labels(
        robot,
        obj,
        {
            "robot_contact_pairs": [["robot0", "object"]],
            "robot_state": {"gripper_qpos": [0.02, -0.02]},
        },
    )
    assert labels["holding"]["valid"] == 1
    assert labels["holding"]["value"] is True


def test_inside_geometry_requires_explicit_container_capability():
    source = _node("object", [0.0, 0.0, 0.0])
    target = _node("bowl", [0.0, 0.0, 0.0])
    source["node_type"] = "object"
    target["node_type"] = "object"
    labels = relation_labels(
        source,
        target,
        {
            "object_states": [
                {"logical_id": "object", "node_type": "object"},
                {"logical_id": "bowl", "node_type": "object"},
            ],
            "geometry": {
                "object": {"center": [0, 0, 0], "aabb_min": [-1, -1, -1], "aabb_max": [1, 1, 1]},
                "bowl": {"center": [0, 0, 0], "aabb_min": [-1, -1, -1], "aabb_max": [1, 1, 1]},
            },
        },
    )
    assert labels["inside"]["valid"] == 0
