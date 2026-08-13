import copy

import pytest


torch = pytest.importorskip("torch")

from scripts.phase3.offline_probe import (
    HOLDER_OBJECT_V2_EDGE_FEATURES,
    HOLDER_OBJECT_V2_NODE_FEATURES,
    ProbeShape,
    RelationalDynamicsProbe,
    StructuredActionEncoder,
    attach_training_action_donors,
    _checkpoint_selection,
    _episode_disjoint_action_permutation,
    _holding_metrics,
    _normalization,
    _parameter_count,
    _record_from_sample,
    _select_holding_threshold,
    _training_action_inputs,
)
from scripts.phase3.run_holder_action_smoke import (
    _holder_object_only,
    _robot_object_topology,
)


def _relation(value, valid=1):
    return {"value": value if valid else None, "valid": valid}


def _node(node_id, node_type, position, *, gripper=None, joint_velocity=None):
    gripper = gripper if gripper is not None else [0.0, 0.0]
    joint_velocity = joint_velocity if joint_velocity is not None else [0.0] * 7
    is_robot = node_type == "robot"
    return {
        "node_id": node_id,
        "node_type": node_type,
        "feature_vector": [99.0] * 24,
        "features": {
            "position": position,
            "position_valid": 1,
            "gripper_qpos": gripper,
            "gripper_qpos_valid": int(is_robot),
            "joint_vel": joint_velocity,
            "joint_vel_valid": int(is_robot),
        },
    }


def _edge(contact, holding):
    return {
        "source": "robot0",
        "target": "object0",
        "node_type_pair": ["robot", "object"],
        "features": {
            "relative_position": [0.1, -0.2, 0.3],
            "distance": 0.4,
            "distance_valid": 1,
        },
        "relations": {
            "contact": _relation(contact),
            "holding": _relation(holding),
        },
    }


def _sample(future_holding=True):
    nodes = [
        _node(
            "robot0",
            "robot",
            [1.0, 2.0, 3.0],
            gripper=[0.02, -0.02],
            joint_velocity=[0.1] * 7,
        ),
        _node("object0", "object", [1.1, 1.8, 3.3]),
    ]
    return {
        "episode_id": "ep0",
        "suite": "libero_spatial",
        "task_id": 0,
        "split": "train",
        "graph_t": {"nodes": nodes, "edges": [_edge(True, False)]},
        "graph_target": {
            "nodes": copy.deepcopy(nodes),
            "edges": [_edge(True, future_holding)],
        },
        "action_window": [
            {"action": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -1.0]},
            {"action": [0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 1.0]},
        ],
    }


def test_holder_object_v2_uses_compact_node_and_contact_aware_edge():
    record = _record_from_sample(
        _sample(),
        relations=("contact", "holding"),
        node_feature_contract="holder_object_v2",
        edge_feature_contract="holder_object_v2",
    )

    assert len(record["node_features"][0]) == 17
    assert len(record["edge_geometry"][0]) == 9
    assert len(HOLDER_OBJECT_V2_NODE_FEATURES) == 17
    assert len(HOLDER_OBJECT_V2_EDGE_FEATURES) == 9
    assert record["edge_geometry"][0] == pytest.approx(
        [0.1, -0.2, 0.3, 0.4, 1.0, 1.0, 1.0, 1.0, 0.0]
    )
    assert record["action_step_mask"] == [1.0, 1.0]


def test_holder_object_v2_features_do_not_read_future_holding():
    positive = _record_from_sample(
        _sample(future_holding=True),
        relations=("contact", "holding"),
        node_feature_contract="holder_object_v2",
        edge_feature_contract="holder_object_v2",
    )
    negative = _record_from_sample(
        _sample(future_holding=False),
        relations=("contact", "holding"),
        node_feature_contract="holder_object_v2",
        edge_feature_contract="holder_object_v2",
    )

    assert positive["node_features"] == negative["node_features"]
    assert positive["edge_geometry"] == negative["edge_geometry"]
    assert positive["actions"] == negative["actions"]
    assert positive["future_labels"] != negative["future_labels"]


def test_holder_object_v2_does_not_feed_current_holding_weak_label():
    free_sample = _sample()
    held_sample = copy.deepcopy(free_sample)
    held_sample["graph_t"]["edges"][0]["relations"]["holding"] = _relation(True)
    options = {
        "relations": ("contact", "holding"),
        "node_feature_contract": "holder_object_v2",
        "edge_feature_contract": "holder_object_v2",
    }

    free = _record_from_sample(free_sample, **options)
    held = _record_from_sample(held_sample, **options)

    assert free["edge_geometry"] == held["edge_geometry"]
    assert free["current_labels"] != held["current_labels"]


def test_v2_sparse_graph_prunes_isolated_fixture_nodes():
    sample = _sample()
    fixture = _node("fixture0", "fixture", [0.0, 0.0, 0.0])
    sample["graph_t"]["nodes"].append(copy.deepcopy(fixture))
    sample["graph_target"]["nodes"].append(copy.deepcopy(fixture))

    sparse = _holder_object_only(sample, prune_non_pair_nodes=True)

    assert [node["node_id"] for node in sparse["graph_t"]["nodes"]] == [
        "robot0",
        "object0",
    ]


def test_complete_topology_adds_message_edges_but_not_prediction_targets():
    sample = _sample()
    object1 = _node("object1", "object", [0.4, 0.5, 0.6])
    sample["graph_t"]["nodes"].append(copy.deepcopy(object1))
    sample["graph_target"]["nodes"].append(copy.deepcopy(object1))

    def add_edge(source, target, source_type, target_type):
        edge = _edge(False, False)
        edge["source"] = source
        edge["target"] = target
        edge["node_type_pair"] = [source_type, target_type]
        sample["graph_t"]["edges"].append(copy.deepcopy(edge))
        sample["graph_target"]["edges"].append(copy.deepcopy(edge))

    add_edge("robot0", "object1", "robot", "object")
    add_edge("object1", "robot0", "object", "robot")
    add_edge("object0", "object1", "object", "object")
    add_edge("object1", "object0", "object", "object")

    sparse = _robot_object_topology(sample, "sparse", prune_non_pair_nodes=True)
    complete = _robot_object_topology(sample, "complete", prune_non_pair_nodes=True)

    assert [node["node_id"] for node in sparse["graph_t"]["nodes"]] == [
        node["node_id"] for node in complete["graph_t"]["nodes"]
    ]
    assert len(sparse["graph_t"]["edges"]) == 3
    assert len(complete["graph_t"]["edges"]) == 5
    assert len(complete["prediction_edge_keys"]) == 3

    record = _record_from_sample(
        complete,
        relations=("contact", "holding"),
        node_feature_contract="holder_object_v2",
        edge_feature_contract="holder_object_v2",
    )
    node_ids = [node["node_id"] for node in complete["graph_t"]["nodes"]]
    object0 = node_ids.index("object0")
    object1_index = node_ids.index("object1")
    context_index = next(
        index
        for index, (source, target) in enumerate(
            zip(record["edge_src"], record["edge_tgt"])
        )
        if source == object0 and target == object1_index
    )
    assert record["current_valid"][context_index] == [0.0, 0.0]
    assert record["future_valid"][context_index] == [0.0, 0.0]
    assert record["edge_geometry"][context_index][6] == 1.0


def test_channel_v2_normalizes_each_action_channel_across_time():
    records = [
        {
            "node_features": [[0.0]],
            "edge_geometry": [[0.0]],
            "actions": [1.0, 10.0, 3.0, 30.0],
            "action_steps": 2,
            "action_dim": 2,
            "action_step_mask": [1.0, 1.0],
        }
    ]
    normalization = _normalization(records, "channel_v2")

    assert normalization["action_mean"].tolist() == pytest.approx([2.0, 20.0, 2.0, 20.0])
    assert normalization["action_std"].tolist() == pytest.approx([1.0, 10.0, 1.0, 10.0])


def test_structured_action_encoder_ignores_masked_steps():
    shape = ProbeShape(
        max_nodes=2,
        max_edges=1,
        node_dim=17,
        edge_dim=9,
        action_dim=14,
        relation_dim=2,
        action_steps=2,
        action_step_dim=7,
    )
    encoder = StructuredActionEncoder(shape, hidden_dim=16).eval()
    first = torch.tensor([[1.0] * 7 + [0.0] * 7])
    changed_padding = torch.tensor([[1.0] * 7 + [100.0] * 7])
    mask = torch.tensor([[1.0, 0.0]])

    with torch.no_grad():
        left = encoder(first, mask)
        right = encoder(changed_padding, mask)
    assert torch.allclose(left, right, atol=1e-6)


def test_b1_v2_has_pair_feature_parity():
    shape = ProbeShape(
        max_nodes=2,
        max_edges=1,
        node_dim=17,
        edge_dim=9,
        action_dim=42,
        relation_dim=2,
        action_steps=6,
        action_step_dim=7,
    )
    model = RelationalDynamicsProbe("b1_pair_feature_mlp_v2", shape, hidden_dim=16)

    assert model.edge_encoder[0].in_features == 16 * 3 + 9


def test_v2_current_head_is_action_free():
    shape = ProbeShape(
        max_nodes=2,
        max_edges=2,
        node_dim=17,
        edge_dim=9,
        action_dim=42,
        relation_dim=2,
        action_steps=6,
        action_step_dim=7,
    )
    model = RelationalDynamicsProbe(
        "g3v2_action_film_holder_object_gnn", shape, hidden_dim=16
    ).eval()
    inputs = (
        torch.randn(2, 2, 17),
        torch.ones(2, 2),
        torch.tensor([[0, 1], [0, 1]]),
        torch.tensor([[1, 0], [1, 0]]),
        torch.randn(2, 2, 9),
        torch.ones(2, 2),
    )
    step_mask = torch.ones(2, 6)

    with torch.no_grad():
        _, current_a = model(
            *inputs, torch.zeros(2, 42), action_step_mask=step_mask
        )
        _, current_b = model(
            *inputs, torch.randn(2, 42), action_step_mask=step_mask
        )
    assert torch.allclose(current_a, current_b, atol=1e-6)


@pytest.mark.parametrize(
    "model_id",
    [
        "b1_pair_feature_mlp_v2",
        "g1_sparse_holder_object_gnn",
        "s0_g1_no_action_holder_object_gnn_v2",
    ],
)
def test_corrected_common_current_head_is_action_free(model_id):
    shape = ProbeShape(
        max_nodes=2,
        max_edges=2,
        node_dim=17,
        edge_dim=9,
        action_dim=42,
        relation_dim=2,
        action_steps=6,
        action_step_dim=7,
    )
    model = RelationalDynamicsProbe(
        model_id,
        shape,
        hidden_dim=16,
        current_head_contract="action_free_pair",
    ).eval()
    inputs = (
        torch.randn(2, 2, 17),
        torch.ones(2, 2),
        torch.tensor([[0, 1], [0, 1]]),
        torch.tensor([[1, 0], [1, 0]]),
        torch.randn(2, 2, 9),
        torch.ones(2, 2),
    )
    with torch.no_grad():
        _, current_a = model(*inputs, torch.zeros(2, 42))
        _, current_b = model(*inputs, torch.randn(2, 42))
    assert torch.allclose(current_a, current_b, atol=1e-6)


def test_v2_action_film_starts_near_unconditioned_message():
    shape = ProbeShape(
        max_nodes=2,
        max_edges=2,
        node_dim=17,
        edge_dim=9,
        action_dim=42,
        relation_dim=2,
        action_steps=6,
        action_step_dim=7,
    )
    model = RelationalDynamicsProbe(
        "g3v2_action_film_holder_object_gnn", shape, hidden_dim=16
    )

    assert torch.count_nonzero(model.action_film.weight).item() == 0
    assert torch.count_nonzero(model.action_film.bias[:-1]).item() == 0
    assert model.action_film.bias[-1].item() == pytest.approx(4.0)


def test_s0_is_trained_without_action_encoder_and_is_action_invariant():
    shape = ProbeShape(
        max_nodes=2,
        max_edges=2,
        node_dim=17,
        edge_dim=9,
        action_dim=42,
        relation_dim=2,
        action_steps=6,
        action_step_dim=7,
    )
    model = RelationalDynamicsProbe(
        "s0_no_action_holder_object_gnn_v2", shape, hidden_dim=16
    ).eval()
    inputs = (
        torch.randn(2, 2, 17),
        torch.ones(2, 2),
        torch.tensor([[0, 1], [0, 1]]),
        torch.tensor([[1, 0], [1, 0]]),
        torch.randn(2, 2, 9),
        torch.ones(2, 2),
    )

    assert not hasattr(model, "action_encoder")
    with torch.no_grad():
        future_a, current_a = model(*inputs, torch.zeros(2, 42))
        future_b, current_b = model(*inputs, torch.randn(2, 42))
    assert torch.allclose(future_a, future_b, atol=0.0, rtol=0.0)
    assert torch.allclose(current_a, current_b, atol=0.0, rtol=0.0)


def test_s0_g1_is_an_exact_action_free_g1_style_block():
    shape = ProbeShape(
        max_nodes=2,
        max_edges=2,
        node_dim=17,
        edge_dim=9,
        action_dim=42,
        relation_dim=2,
        action_steps=6,
        action_step_dim=7,
    )
    model = RelationalDynamicsProbe(
        "s0_g1_no_action_holder_object_gnn_v2", shape, hidden_dim=16
    ).eval()
    inputs = (
        torch.randn(2, 2, 17),
        torch.ones(2, 2),
        torch.tensor([[0, 1], [0, 1]]),
        torch.tensor([[1, 0], [1, 0]]),
        torch.randn(2, 2, 9),
        torch.ones(2, 2),
    )

    assert not hasattr(model, "action_encoder")
    assert not hasattr(model, "node_norm")
    assert model.edge_encoder[0].in_features == 16 * 2 + 9
    with torch.no_grad():
        future_a, current_a = model(*inputs, torch.zeros(2, 42))
        future_b, current_b = model(*inputs, torch.randn(2, 42))
    assert torch.allclose(future_a, future_b, atol=0.0, rtol=0.0)
    assert torch.allclose(current_a, current_b, atol=0.0, rtol=0.0)


def test_constant_action_g1_matches_g1_parameters_and_ignores_sample_actions():
    shape = ProbeShape(
        max_nodes=2,
        max_edges=2,
        node_dim=17,
        edge_dim=9,
        action_dim=42,
        relation_dim=2,
        action_steps=6,
        action_step_dim=7,
    )
    g1 = RelationalDynamicsProbe(
        "g1_sparse_holder_object_gnn", shape, hidden_dim=16
    )
    constant = RelationalDynamicsProbe(
        "g1_constant_action_holder_object_gnn_v2", shape, hidden_dim=16
    ).eval()
    inputs = (
        torch.randn(2, 2, 17),
        torch.ones(2, 2),
        torch.tensor([[0, 1], [0, 1]]),
        torch.tensor([[1, 0], [1, 0]]),
        torch.randn(2, 2, 9),
        torch.ones(2, 2),
    )

    assert _parameter_count(constant) == _parameter_count(g1)
    with torch.no_grad():
        future_a, current_a = constant(*inputs, torch.zeros(2, 42))
        future_b, current_b = constant(*inputs, torch.randn(2, 42))
    assert torch.allclose(future_a, future_b, atol=0.0, rtol=0.0)
    assert torch.allclose(current_a, current_b, atol=0.0, rtol=0.0)


def test_training_action_shuffle_preserves_marginal_and_breaks_every_pair():
    actions = torch.arange(4 * 6, dtype=torch.float32).reshape(4, 6)
    step_mask = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2)
    shuffled_actions, shuffled_mask, changed = _training_action_inputs(
        {"actions": actions, "action_step_mask": step_mask}, "shuffled_batch"
    )

    assert changed is True
    assert torch.equal(shuffled_actions, torch.roll(actions, shifts=1, dims=0))
    assert torch.equal(shuffled_mask, torch.roll(step_mask, shifts=1, dims=0))
    assert all(
        not torch.equal(shuffled_actions[index], actions[index])
        for index in range(actions.shape[0])
    )
    assert torch.equal(
        torch.sort(shuffled_actions[:, 0]).values,
        torch.sort(actions[:, 0]).values,
    )


def test_global_action_shuffle_is_bijective_and_episode_disjoint():
    records = [
        {"episode_id": episode}
        for episode in ("a", "a", "a", "b", "b", "c", "c", "d")
    ]
    donors = _episode_disjoint_action_permutation(records)
    episodes = [record["episode_id"] for record in records]

    assert sorted(donors.tolist()) == list(range(len(records)))
    assert all(
        episodes[index] != episodes[int(donor)]
        for index, donor in enumerate(donors)
    )


def test_matched_training_shuffle_is_task_local_episode_disjoint_and_marginal_preserving():
    records = []
    for index, episode in enumerate(("a", "a", "b", "b", "c", "c", "d", "d")):
        records.append(
            {
                "sample_id": f"sample{index}",
                "suite": "libero_spatial",
                "task_id": 1,
                "episode_id": episode,
                "actions": [float(index), 0.0, 0.0, float(index % 2)],
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

    assert qa["same_task_fraction"] == 1.0
    assert qa["different_episode_fraction"] == 1.0
    assert sorted(tuple(row["training_donor_actions"]) for row in decorated) == sorted(
        tuple(row["actions"]) for row in records
    )
    assert all(
        row["episode_id"] != row["training_action_donor_episode_id"]
        for row in decorated
    )


def _holding_outputs():
    current = torch.tensor(
        [[[1, 0], [1, 0], [1, 1], [0, 1]]], dtype=torch.float32
    ).numpy()
    future = torch.tensor(
        [[[1, 1], [1, 0], [1, 0], [0, 1]]], dtype=torch.float32
    ).numpy()
    score = torch.tensor(
        [[[0.8, 0.9], [0.8, 0.6], [0.8, 0.2], [0.2, 0.8]]],
        dtype=torch.float32,
    ).numpy()
    valid = torch.ones_like(torch.tensor(current)).numpy()
    return {
        "current_true": current,
        "future_true": future,
        "future_score": score,
        "current_valid": valid,
        "future_valid": valid,
        "changed": (current != future).astype("float32"),
    }


def test_holding_metrics_separate_change_onset_release_and_hard_negative():
    metrics = _holding_metrics(_holding_outputs(), ("contact", "holding"), 0.5)

    assert metrics is not None
    assert metrics["future_state"]["f1"] == pytest.approx(0.8)
    assert metrics["change_event"]["f1"] == pytest.approx(0.8)
    assert metrics["onset"]["f1"] == pytest.approx(2.0 / 3.0)
    assert metrics["release"]["f1"] == pytest.approx(1.0)
    assert metrics["hard_negative"]["support"] == 1
    assert metrics["hard_negative"]["false_positive_rate"] == pytest.approx(1.0)
    assert metrics["future_state"]["pr_auc"] is not None
    assert metrics["conditional_oracle_current_change_event"] == metrics["change_event"]
    assert metrics["end_to_end_change_event"]["pr_auc"] is not None
    assert metrics["change_event"]["brier_score"] is not None
    assert metrics["change_event"]["ece"] is not None


def test_holding_threshold_is_fit_for_change_event_on_validation_outputs():
    threshold, metrics = _select_holding_threshold(
        _holding_outputs(), ("contact", "holding")
    )

    assert threshold == pytest.approx(0.61)
    assert metrics is not None
    assert metrics["change_event"]["f1"] == pytest.approx(1.0)


def test_checkpoint_prefers_holding_change_over_global_macro_f1():
    score, name = _checkpoint_selection({
        "future_relation": {"macro_f1": 0.99},
        "holding": {
            "change_event": {"positive": 4, "f1": 0.75},
            "future_state": {"positive": 8, "f1": 0.9},
        },
    })

    assert score == pytest.approx(0.75)
    assert name == "holding_change_event_f1"


def test_corrected_checkpoint_uses_threshold_free_holding_event_pr_auc():
    score, name = _checkpoint_selection(
        {
            "future_relation": {"macro_f1": 0.99},
            "holding": {
                "conditional_oracle_current_change_event": {
                    "positive": 4,
                    "f1": 0.20,
                    "pr_auc": 0.81,
                }
            },
        },
        criterion="holding_event_pr_auc",
    )

    assert score == pytest.approx(0.81)
    assert name == "holding_event_pr_auc"


def test_p0_does_not_count_unused_node_or_action_encoders():
    shape = ProbeShape(
        max_nodes=2,
        max_edges=2,
        node_dim=17,
        edge_dim=9,
        action_dim=42,
        relation_dim=2,
        action_steps=6,
        action_step_dim=7,
    )
    model = RelationalDynamicsProbe("p0_flat_mlp", shape, hidden_dim=16)

    assert not hasattr(model, "node_encoder")
    assert not hasattr(model, "action_encoder")
    assert _parameter_count(model) == sum(
        parameter.numel() for parameter in model.parameters()
    )


@pytest.mark.parametrize(
    "model_id",
    [
        "g2_flat_action_holder_object_gnn_v2",
        "g2_structured_action_holder_object_gnn",
        "g3v2_action_film_holder_object_gnn",
        "s0_no_action_holder_object_gnn_v2",
        "c_l_complete_late_action_gnn_v2",
        "c_e_complete_action_film_gnn_v2",
    ],
)
def test_structured_sparse_models_have_expected_output_shape(model_id):
    shape = ProbeShape(
        max_nodes=2,
        max_edges=2,
        node_dim=17,
        edge_dim=9,
        action_dim=42,
        relation_dim=2,
        action_steps=6,
        action_step_dim=7,
    )
    model = RelationalDynamicsProbe(model_id, shape, hidden_dim=16)
    future, current = model(
        torch.randn(3, 2, 17),
        torch.ones(3, 2),
        torch.tensor([[0, 1], [0, 1], [0, 1]]),
        torch.tensor([[1, 0], [1, 0], [1, 0]]),
        torch.randn(3, 2, 9),
        torch.ones(3, 2),
        torch.randn(3, 42),
        action_step_mask=torch.ones(3, 6),
    )

    assert future.shape == (3, 2, 2)
    assert current.shape == (3, 2, 2)
