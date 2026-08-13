from scripts.validate_phase2r_scaleup import _frame_audit, _relation_coverage


def test_scaleup_frame_audit_is_per_task():
    capture = {
        "episodes": [
            {
                "task_id": 0,
                "snapshots": [
                    {
                        "robot_base_pose": {"position": [0, 0, 0], "quaternion": [1, 0, 0, 0]},
                        "geometry": {"obj": {"valid": 1}},
                    },
                    {
                        "robot_base_pose": {"position": [0, 0, 0], "quaternion": [1, 0, 0, 0]},
                        "geometry": {"obj": {"valid": 1}},
                    },
                ],
            },
            {
                "task_id": 1,
                "snapshots": [
                    {
                        "robot_base_pose": {"position": [1, 2, 3], "quaternion": [1, 0, 0, 0]},
                        "geometry": {"obj": {"valid": 1}},
                    },
                    {
                        "robot_base_pose": {"position": [1, 2, 3], "quaternion": [1, 0, 0, 0]},
                        "geometry": {"obj": {"valid": 1}},
                    },
                ],
            },
        ]
    }
    report = _frame_audit(capture)
    assert report["all_tasks_stable"] is True
    assert report["by_task"]["0"]["position_range"] == [0, 0, 0]
    assert report["by_task"]["1"]["position_range"] == [0, 0, 0]


def test_scaleup_relation_coverage_separates_edge_and_node_labels():
    dataset = {
        "samples": [
            {
                "task_id": 3,
                "graph_t": {
                    "edges": [{"relations": {"on": {"valid": 1}, "inside": {"valid": 0}, "holding": {"valid": 0}}}],
                    "node_semantics": {"obj": {"open": {"valid": 1}, "close": {"valid": 0}}},
                },
                "graph_target": {"edges": [], "node_semantics": {}},
                "relation_changes": {"a->b::on": 1},
            }
        ]
    }
    coverage = _relation_coverage(dataset)
    assert coverage["edge_valid_by_task"]["3"]["on"] == 1
    assert coverage["node_valid_by_task"]["3"]["open"] == 1
    assert coverage["changed_by_task"]["3"]["on"] == 1
