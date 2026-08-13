import unittest

from scripts.phase3.build_eval_manifest import (
    _check_disjoint,
    _overlap_payload_hash_qa,
    _relation_support,
    sample_id,
)


def _relation(value: bool, valid: bool = True) -> dict:
    return {"value": value, "valid": valid}


def _sample(*, episode: str = "task0_demo0", start: int = 0, tau: int = 6) -> dict:
    edge_now = {
        "source": "robot0",
        "target": "bowl",
        "relations": {"holding": _relation(False), "contact": _relation(True)},
    }
    edge_future = {
        "source": "robot0",
        "target": "bowl",
        "relations": {"holding": _relation(True), "contact": _relation(True)},
    }
    return {
        "suite": "libero_spatial",
        "task_id": 0,
        "episode_id": episode,
        "split": "test",
        "start_step": start,
        "target_step": start + tau,
        "tau": tau,
        "graph_t": {"edges": [edge_now]},
        "graph_target": {"edges": [edge_future]},
    }


class EvalManifestTest(unittest.TestCase):
    def test_sample_id_is_stable_for_same_transition(self):
        self.assertEqual(sample_id(_sample()), sample_id(dict(_sample())))
        self.assertNotEqual(sample_id(_sample(start=1)), sample_id(_sample(start=2)))

    def test_holding_support_counts_valid_positive_and_change(self):
        support = _relation_support([_sample()], ["holding", "contact"])
        self.assertEqual(support["holding"], {"valid": 1, "positive": 1, "changed": 1})
        self.assertEqual(support["contact"], {"valid": 1, "positive": 1, "changed": 0})

    def test_challenge_is_allowed_to_overlap_natural_but_not_train(self):
        train = _sample(episode="train_episode")
        natural = _sample(episode="test_episode")
        challenge = dict(natural)
        warnings = _check_disjoint(
            {"name": "test_task0"},
            {"train": [train], "validation": [], "natural_test": [natural], "challenge_test": [challenge]},
        )
        self.assertEqual(warnings, [])

    def test_overlap_payload_hash_detects_graph_action_or_label_mismatch(self):
        natural = _sample(episode="test_episode")
        challenge = dict(natural)
        challenge["graph_target"] = {
            "edges": [
                {
                    "source": "robot0",
                    "target": "bowl",
                    "relations": {
                        "holding": _relation(False),
                        "contact": _relation(True),
                    },
                }
            ]
        }
        qa = _overlap_payload_hash_qa([natural], [challenge])
        warnings = _check_disjoint(
            {"name": "test_task0"},
            {
                "train": [],
                "validation": [],
                "natural_test": [natural],
                "challenge_test": [challenge],
            },
        )

        self.assertEqual(qa["payload_hash_mismatch_count"], 1)
        self.assertTrue(any("payload hash mismatch" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
