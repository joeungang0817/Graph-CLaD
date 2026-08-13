import unittest

from scripts.phase3.sampling import (
    category_aware_cap_v1,
    category_aware_episode_round_robin_cap,
    category_counts,
    episode_round_robin_cap,
)


def _row(episode: str, index: int, *categories: str) -> dict:
    return {
        "episode_id": episode,
        "index": index,
        "target_categories": list(categories),
    }


class Phase3SamplingTest(unittest.TestCase):
    def test_episode_round_robin_cycles_episodes(self):
        rows = [
            _row("a", 0, "background"),
            _row("a", 1, "background"),
            _row("b", 0, "background"),
            _row("b", 1, "background"),
        ]
        selected = episode_round_robin_cap(rows, 4)
        self.assertEqual(
            [(row["episode_id"], row["index"]) for row in selected],
            [("a", 0), ("b", 0), ("a", 1), ("b", 1)],
        )

    def test_v1_reproduces_greedy_episode_bias_and_v2_fixes_it(self):
        rows = [
            *[_row("a", index, "holding_changed") for index in range(3)],
            *[_row("b", index, "holding_changed") for index in range(3)],
        ]
        v1 = category_aware_cap_v1(
            rows,
            4,
            categories=("holding_changed",),
            category_quota=4,
        )
        v2 = category_aware_episode_round_robin_cap(
            rows,
            4,
            categories=("holding_changed",),
            category_quota=4,
        )
        self.assertEqual([row["episode_id"] for row in v1], ["a", "a", "a", "b"])
        self.assertEqual([row["episode_id"] for row in v2], ["a", "b", "a", "b"])

    def test_category_counts_are_multilabel(self):
        counts = category_counts(
            [_row("a", 0, "holding_changed", "future_holding_positive")]
        )
        self.assertEqual(counts["holding_changed"], 1)
        self.assertEqual(counts["future_holding_positive"], 1)
        self.assertEqual(sum(counts.values()), 2)


if __name__ == "__main__":
    unittest.main()
