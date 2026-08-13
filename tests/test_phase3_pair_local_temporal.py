from __future__ import annotations

import copy

import torch

from scripts.phase3 import offline_probe as probe
from scripts.phase3.pair_local_temporal import (
    HISTORY_DIM,
    HISTORY_FEATURE_KEY,
    attach_causal_pair_history,
    causal_pair_history_features,
    requested_history_keys,
)


def _graph(step: int, *, future_offset: float = 0.0):
    robot_x = 0.1 * step
    object_x = 0.2 + 0.1 * step + future_offset
    qpos = max(0.01, 0.04 - 0.01 * step)
    contact = step > 0
    nodes = [
        {
            "node_id": "robot0",
            "node_type": "robot",
            "features": {
                "position": [robot_x, 0.0, 0.0],
                "position_valid": 1,
                "gripper_qpos": [qpos, qpos],
                "gripper_qpos_valid": 1,
            },
        },
        {
            "node_id": "cube",
            "node_type": "object",
            "features": {
                "position": [object_x, 0.0, 0.0],
                "position_valid": 1,
            },
        },
    ]
    edges = []
    for source, target, pair in (
        ("robot0", "cube", ["robot", "object"]),
        ("cube", "robot0", ["object", "robot"]),
    ):
        edges.append(
            {
                "source": source,
                "target": target,
                "node_type_pair": pair,
                "features": {
                    "relative_position": [object_x - robot_x, 0.0, 0.0],
                    "distance": abs(object_x - robot_x),
                    "distance_valid": 1,
                },
                "relations": {
                    "contact": {"value": contact, "valid": 1},
                    "holding": {"value": contact, "valid": 1},
                },
            }
        )
    return {"nodes": nodes, "edges": edges}


def test_causal_history_uses_only_frames_through_current_t():
    graphs = [(step, _graph(step)) for step in range(4)]
    features, qa = causal_pair_history_features(graphs, ("robot0", "cube"))
    assert len(features) == HISTORY_DIM
    assert features[3] == features[7] == features[10] == features[12] == features[15] == 1.0
    assert abs(features[8] - 0.75) < 1e-8
    assert abs(features[9] - 0.75) < 1e-8
    assert abs(features[11] - 0.01) < 1e-8
    assert abs(features[13]) < 1e-8
    assert abs(features[14]) < 1e-8
    assert qa["last_step"] == 3

    sample = {
        "suite": "libero_spatial",
        "task_id": 1,
        "episode_id": "task1_demo0",
        "start_step": 3,
        "target_step": 9,
        "tau": 6,
        "graph_t": _graph(3),
        "graph_target": _graph(9, future_offset=100.0),
    }
    index = {(1, "task1_demo0", step): graph for step, graph in graphs}
    index[(1, "task1_demo0", 9)] = sample["graph_target"]
    attached, sample_qa = attach_causal_pair_history(sample, index, lookback_steps=3)
    attached_features = attached["graph_t"]["edges"][0]["features"][HISTORY_FEATURE_KEY]
    assert attached_features == features
    assert sample_qa["available_steps"] == [0, 1, 2, 3]
    assert sample_qa["future_frame_reads"] == 0
    assert requested_history_keys([sample], 3) == {
        (1, "task1_demo0", 0),
        (1, "task1_demo0", 1),
        (1, "task1_demo0", 2),
        (1, "task1_demo0", 3),
    }


def test_h0_h3_factorial_input_boundaries():
    shape = probe.ProbeShape(
        max_nodes=2,
        max_edges=2,
        node_dim=17,
        edge_dim=9 + HISTORY_DIM,
        action_dim=42,
        relation_dim=9,
        action_steps=6,
        action_step_dim=7,
        history_dim=HISTORY_DIM,
    )
    node_x = torch.randn(2, 2, shape.node_dim)
    node_mask = torch.ones(2, 2)
    edge_src = torch.tensor([[0, 1], [0, 1]])
    edge_tgt = torch.tensor([[1, 0], [1, 0]])
    edge_geometry = torch.randn(2, 2, shape.edge_dim)
    edge_mask = torch.ones(2, 2)
    actions = torch.randn(2, shape.action_dim)
    action_mask = torch.ones(2, shape.action_steps)

    model_ids = {
        "h0_pair_local_no_history_no_action": (False, False),
        "h1_pair_local_history_no_action": (True, False),
        "h2_pair_local_no_history_action_film": (False, True),
        "h3_pair_local_history_action_film": (True, True),
    }
    for model_id, (uses_history, uses_action) in model_ids.items():
        model = probe.RelationalDynamicsProbe(
            model_id, shape, hidden_dim=16, current_head_contract="action_free_pair"
        ).eval()
        assert model.uses_history is uses_history
        assert model.uses_action is uses_action
        with torch.no_grad():
            future, current = model(
                node_x,
                node_mask,
                edge_src,
                edge_tgt,
                edge_geometry,
                edge_mask,
                actions,
                action_step_mask=action_mask,
            )
            assert future.shape == current.shape == (2, 2, shape.relation_dim)
            changed_history = copy.deepcopy(edge_geometry)
            changed_history[..., -HISTORY_DIM:] += 100.0
            future_changed, current_changed = model(
                node_x,
                node_mask,
                edge_src,
                edge_tgt,
                changed_history,
                edge_mask,
                actions,
                action_step_mask=action_mask,
            )
        assert torch.equal(current, current_changed)
        if not uses_history:
            assert torch.equal(future, future_changed)
        else:
            assert not torch.equal(future, future_changed)
