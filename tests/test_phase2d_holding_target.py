import unittest

from scripts.phase2d.build_holding_target_dataset import classify_target_sample


def _edge(relation: str, value: bool, valid: bool = True) -> dict:
    return {
        "source": "robot0",
        "target": "object0",
        "relations": {relation: {"valid": valid, "value": value}},
    }


def _graph(*edges: dict) -> dict:
    merged = {}
    for edge in edges:
        key = (edge["source"], edge["target"])
        target = merged.setdefault(
            key,
            {"source": edge["source"], "target": edge["target"], "relations": {}},
        )
        target["relations"].update(edge["relations"])
    return {"edges": list(merged.values())}


class Phase2DHoldingTargetTest(unittest.TestCase):
    def test_future_positive_and_changed_are_multilabel(self):
        sample = {
            "start_step": 2,
            "target_step": 5,
            "graph_t": _graph(_edge("holding", False), _edge("contact", True)),
            "graph_target": _graph(_edge("holding", True), _edge("contact", True)),
        }
        record = {
            "events": [
                {"step": 4, "event": "holding_state_change", "to_state": "holding"}
            ]
        }
        result = classify_target_sample(record, sample)
        self.assertEqual(
            result["target_categories"],
            ["future_holding_positive", "holding_changed"],
        )
        self.assertEqual(result["target_event_tags"], ["holding_onset"])
        self.assertEqual(result["target_event_objects"], ["object0"])

    def test_contact_without_holding_is_hard_negative(self):
        sample = {
            "graph_t": _graph(_edge("holding", False), _edge("contact", True)),
            "graph_target": _graph(_edge("holding", False), _edge("contact", False)),
        }
        result = classify_target_sample({}, sample)
        self.assertEqual(result["target_categories"], ["hard_negative"])

    def test_invalid_holding_is_ambiguous(self):
        sample = {
            "graph_t": _graph(_edge("holding", False, valid=False)),
            "graph_target": _graph(_edge("holding", False)),
        }
        result = classify_target_sample({}, sample)
        self.assertEqual(result["target_categories"], ["ambiguous"])


if __name__ == "__main__":
    unittest.main()
