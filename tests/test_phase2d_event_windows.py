from __future__ import annotations

import unittest

from scripts.phase2d.event_windows import classify_sample


def _edge(source: str, target: str, holding: dict, contact: dict) -> dict:
    return {
        "source": source,
        "target": target,
        "relations": {"holding": holding, "contact": contact},
    }


class EventWindowTest(unittest.TestCase):
    def test_contact_without_holding_is_hard_negative(self):
        record = {"events": []}
        sample = {
            "start_step": 1,
            "target_step": 2,
            "graph_t": {"edges": [_edge("robot0", "bowl", {"value": False, "valid": 1}, {"value": True, "valid": 1})]},
            "graph_target": {"edges": [_edge("robot0", "bowl", {"value": False, "valid": 1}, {"value": True, "valid": 1})]},
        }
        result = classify_sample(record, sample)
        self.assertEqual(result["event_category"], "hard_negative")
        self.assertIn("contact_without_holding", result["event_tags"])


    def test_holding_event_is_positive_even_on_release(self):
        record = {
            "events": [
                {
                    "step": 2,
                    "event": "holding_state_change",
                    "object_id": "bowl",
                    "from_state": "holding",
                    "to_state": "release",
                }
            ]
        }
        sample = {
            "start_step": 1,
            "target_step": 2,
            "graph_t": {"edges": [_edge("robot0", "bowl", {"value": True, "valid": 1}, {"value": True, "valid": 1})]},
            "graph_target": {"edges": [_edge("robot0", "bowl", {"value": False, "valid": 1}, {"value": False, "valid": 1})]},
        }
        result = classify_sample(record, sample)
        self.assertEqual(result["event_category"], "positive_event")
        self.assertIn("holding_release", result["event_tags"])


    def test_unknown_holding_is_not_negative(self):
        record = {"events": []}
        sample = {
            "start_step": 0,
            "target_step": 1,
            "graph_t": {"edges": [_edge("robot0", "bowl", {"value": None, "valid": 0}, {"value": True, "valid": 1})]},
            "graph_target": {"edges": [_edge("robot0", "bowl", {"value": None, "valid": 0}, {"value": True, "valid": 1})]},
        }
        self.assertEqual(classify_sample(record, sample)["event_category"], "ambiguous")


if __name__ == "__main__":
    unittest.main()
