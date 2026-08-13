from scripts.collect_libero_trajectory import _contact_pairs, _task_semantic_metadata, holding_probe_action


def test_bddl_in_goal_grants_only_explicit_container_capability(tmp_path):
    bddl = tmp_path / "task.bddl"
    bddl.write_text(
        "(:goal (And (In object_1 drawer_top_region) (On object_1 table_region)))",
        encoding="utf-8",
    )

    metadata = _task_semantic_metadata(bddl)

    assert metadata["available"] is True
    assert metadata["inside_pairs"] == [["object_1", "drawer_top_region"]]
    assert metadata["containment_capable_ids"] == ["drawer_top_region"]
    assert metadata["on_pairs"] == [["object_1", "table_region"]]


def test_gripper_body_contacts_are_mapped_to_robot_node():
    class Contact:
        geom1 = 0
        geom2 = 1

    class Data:
        ncon = 1
        contact = [Contact()]

    class Model:
        geom_bodyid = [10, 20]

        @staticmethod
        def body_id2name(body_id):
            return {10: "gripper0_finger1", 20: "object_body"}[body_id]

    class Sim:
        data = Data()
        model = Model()

    class Env:
        sim = Sim()

    pairs = _contact_pairs(
        Env(),
        [{"logical_id": "object", "body_id": 20}],
    )
    assert pairs == [["object", "robot0"]]


def test_runtime_gripper_suffix_and_bytes_are_mapped_to_robot_node():
    class Contact:
        geom1 = 0
        geom2 = 1

    class Data:
        ncon = 1
        contact = [Contact()]

    class Model:
        geom_bodyid = [10, 20]

        @staticmethod
        def body_id2name(body_id):
            return {10: b"robot0_gripper0_finger_joint1_tip", 20: "object_body"}[body_id]

    class Sim:
        data = Data()
        model = Model()

    class Env:
        sim = Sim()

    pairs = _contact_pairs(
        Env(),
        [{"logical_id": "object", "body_id": 20}],
    )
    assert pairs == [["object", "robot0"]]


def test_holding_probe_opens_then_closes_and_lifts():
    opening = holding_probe_action(0, 7, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    closing = holding_probe_action(50, 7, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    assert opening[-1] == -1.0
    assert closing[-1] == 1.0
    assert closing[2] > opening[2]
