from scripts.validate_phase2r_pilot import validate_payload


def test_validation_accepts_reciprocal_relations_and_split():
    def edge(source, target, left, right):
        return {
            "source": source,
            "target": target,
            "relations": {
                "left": {"value": left, "valid": 1},
                "right": {"value": right, "valid": 1},
                "front": {"value": False, "valid": 1},
                "behind": {"value": False, "valid": 1},
                "above": {"value": False, "valid": 1},
                "below": {"value": False, "valid": 1},
                "contact": {"value": False, "valid": 1},
            },
        }

    graph = {
        "node_feature_dim": 1,
        "nodes": [
            {"node_id": "a", "node_type": "object", "feature_vector": [0]},
            {"node_id": "b", "node_type": "object", "feature_vector": [0]},
        ],
        "edges": [edge("a", "b", True, False), edge("b", "a", False, True)],
    }
    sample = {
        "episode_id": "ep-a",
        "split": "train",
        "start_step": 0,
        "target_step": 1,
        "tau": 1,
        "action_window": [[0.1]],
        "graph_t": graph,
        "graph_target": graph,
        "relation_changes": {"a->b::left": 1},
    }
    report = validate_payload({
        "tau": 1,
        "coordinate_frame": "robot_base",
        "split": {"assignments": {"ep-a": "train"}},
        "samples": [sample],
    })
    assert report["status"] == "pass"
    assert report["gate_results"]["inverse_relation_consistency"] is True
