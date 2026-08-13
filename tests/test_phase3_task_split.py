from scripts.phase3_task_split import make_task_family_split


def test_task_family_split_is_disjoint_and_episode_consistent():
    dataset = {
        "samples": [
            {"episode_id": "a", "suite": "libero_90", "task_id": 2, "split": "train"},
            {"episode_id": "a", "suite": "libero_90", "task_id": 2, "split": "train"},
            {"episode_id": "b", "suite": "libero_spatial", "task_id": 8, "split": "train"},
            {"episode_id": "c", "suite": "libero_spatial", "task_id": 0, "split": "test"},
        ]
    }
    output = make_task_family_split(
        dataset,
        validation_families=["libero_spatial:8"],
        test_families=["libero_90:2"],
    )

    assert [sample["split"] for sample in output["samples"]] == [
        "test",
        "test",
        "validation",
        "train",
    ]
    assert output["split"]["assignments"] == {"a": "test", "b": "validation", "c": "train"}
