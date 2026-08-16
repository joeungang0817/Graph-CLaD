from __future__ import annotations

import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.phase3c.build_joined_manifest import build_joined_manifest
from scripts.phase3c.contracts import (
    PRIMARY_RELATIONS,
    assert_causal_input,
    forbidden_keys,
    model_input_view,
    relation_targets,
    support_report,
)


def _relation(value: bool | None, valid: int = 1) -> dict:
    return {
        "value": value if valid else None,
        "valid": int(valid),
        "status": "available" if valid else "unknown",
    }


def _graph(step: int, *, on: bool, valid_on: int = 1) -> dict:
    return {
        "step": step,
        "nodes": [
            {
                "node_id": "object0",
                "node_type": "object",
                "features": {"position": [0.1 + step * 0.001, 0.0, 0.2], "position_valid": 1},
            },
            {
                "node_id": "fixture0",
                "node_type": "fixture",
                "features": {"position": [0.2, 0.0, 0.2], "position_valid": 1},
            },
        ],
        "edges": [
            {
                "source": "object0",
                "target": "fixture0",
                "node_type_pair": ["object", "fixture"],
                "features": {
                    "relative_position": [0.1, 0.0, 0.0],
                    "distance": 0.1,
                    "distance_valid": 1,
                },
                "relations": {
                    relation: _relation(on if relation == "on" else False, valid_on if relation == "on" else 1)
                    for relation in PRIMARY_RELATIONS
                },
            }
        ],
    }


def _actions(offset: float) -> list[list[float]]:
    return [[offset + step, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for step in range(6)]


def _sample(
    start: int,
    current: dict,
    future: dict,
    *,
    action_offset: float,
    split: str = "train",
) -> dict:
    return {
        "episode_id": "task0_demo_0",
        "demo_key": "demo_0",
        "task_id": 0,
        "split": split,
        "start_step": start,
        "target_step": start + 6,
        "tau": 6,
        "action_window": _actions(action_offset),
        "graph_t": current,
        "graph_target": future,
    }


def _write_demo(path: Path, samples: list[dict]) -> None:
    record = {
        "episode_id": "task0_demo_0",
        "demo_key": "demo_0",
        "task_id": 0,
        "split": {"in_task": "train"},
        "samples": samples,
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


class Phase3CJoinContractTest(unittest.TestCase):
    def test_join_uses_left_action_and_right_target_only(self):
        graph0 = _graph(0, on=False)
        graph6 = _graph(6, on=False)
        graph12 = _graph(12, on=True)
        samples = [
            _sample(0, graph0, graph6, action_offset=10.0),
            _sample(6, graph6, graph12, action_offset=20.0),
            _sample(12, graph12, graph12, action_offset=30.0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "task.jsonl.gz"
            output = root / "joined.jsonl.gz"
            qa = root / "qa.json"
            _write_demo(source, samples)

            report = build_joined_manifest([source], output, qa)
            with gzip.open(output, "rt", encoding="utf-8") as handle:
                joined = [json.loads(line) for line in handle if line.strip()]

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["counters"]["joined_samples"], 2)
        self.assertEqual(report["counters"]["boundary_drops"], 1)
        self.assertEqual(len(joined), 2)
        first = joined[0]
        self.assertEqual((first["prev_step"], first["current_step"], first["target_step"]), (0, 6, 12))
        self.assertEqual(first["past_action_window"][0][0], 10.0)
        self.assertNotIn("future_action", first)
        self.assertNotIn("action_future", first)
        self.assertNotIn("action_window", first)
        self.assertEqual(first["target"]["relation_any_change"]["on"], 1)
        self.assertEqual(first["target"]["relation_valid"]["on"], 1)
        assert_causal_input(first)

    def test_join_selects_tau_six_from_mixed_horizon_source(self):
        graph0 = _graph(0, on=False)
        graph6 = _graph(6, on=False)
        graph12 = _graph(12, on=True)
        tau6 = [
            _sample(0, graph0, graph6, action_offset=10.0),
            _sample(6, graph6, graph12, action_offset=20.0),
        ]
        mixed = []
        for sample in tau6:
            for horizon in (1, 3):
                other = copy.deepcopy(sample)
                other["tau"] = horizon
                other["target_step"] = other["start_step"] + horizon
                other["action_window"] = other["action_window"][:horizon]
                mixed.append(other)
            mixed.append(sample)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mixed.jsonl.gz"
            output = root / "joined.jsonl.gz"
            qa = root / "qa.json"
            _write_demo(source, mixed)
            report = build_joined_manifest([source], output, qa, tau=6)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["counters"]["ignored_other_tau_samples"], 4)
        self.assertEqual(report["counters"]["joined_samples"], 1)

    def test_future_action_poison_does_not_change_model_input_view(self):
        graph0 = _graph(0, on=False)
        graph6 = _graph(6, on=False)
        graph12 = _graph(12, on=True)
        left = _sample(0, graph0, graph6, action_offset=1.0)
        right = _sample(6, graph6, graph12, action_offset=2.0)
        poisoned_right = copy.deepcopy(right)
        poisoned_right["action_window"] = [[9999.0] * 7 for _ in range(6)]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = root / "natural_a.jsonl.gz"
            source_b = root / "natural_b.jsonl.gz"
            output_a = root / "joined_a.jsonl.gz"
            output_b = root / "joined_b.jsonl.gz"
            qa_a = root / "qa_a.json"
            qa_b = root / "qa_b.json"
            _write_demo(source_a, [left, right])
            _write_demo(source_b, [left, poisoned_right])
            build_joined_manifest([source_a], output_a, qa_a)
            build_joined_manifest([source_b], output_b, qa_b)
            with gzip.open(output_a, "rt", encoding="utf-8") as handle:
                record_a = json.loads(next(handle))
            with gzip.open(output_b, "rt", encoding="utf-8") as handle:
                record_b = json.loads(next(handle))

        self.assertEqual(model_input_view(record_a), model_input_view(record_b))
        self.assertNotEqual(record_a["target"]["graph"], {})
        self.assertNotIn("future_action", forbidden_keys(model_input_view(record_a)))

    def test_graph_hash_mismatch_fails_atomically(self):
        graph0 = _graph(0, on=False)
        graph6 = _graph(6, on=False)
        mismatched = _graph(6, on=True)
        graph12 = _graph(12, on=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bad.jsonl.gz"
            output = root / "joined.jsonl.gz"
            qa = root / "qa.json"
            _write_demo(
                source,
                [
                    _sample(0, graph0, graph6, action_offset=1.0),
                    _sample(6, mismatched, graph12, action_offset=2.0),
                ],
            )
            with self.assertRaisesRegex(ValueError, "graph hash mismatch"):
                build_joined_manifest([source], output, qa)
            self.assertFalse(output.exists())
            self.assertTrue(qa.exists())
            report = json.loads(qa.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "fail")
            self.assertIn("graph hash mismatch", report["error"])

    def test_invalid_relation_is_masked_not_converted_to_negative(self):
        current = _graph(0, on=False, valid_on=0)
        future = _graph(6, on=True, valid_on=1)
        targets = relation_targets(current, future)
        self.assertEqual(targets["relation_valid"]["on"], 0)
        self.assertEqual(targets["relation_any_change"]["on"], 0)

    def test_support_report_uses_test_only_for_reporting(self):
        def row(split: str, label: int) -> dict:
            return {
                "split": split,
                "target": {
                    "relation_any_change": {relation: label for relation in PRIMARY_RELATIONS},
                    "relation_valid": {relation: 1 for relation in PRIMARY_RELATIONS},
                },
            }

        rows = [row("train", 1), row("train", 0), row("validation", 1), row("validation", 0), row("test", 1)]
        report = support_report(
            rows,
            min_train_positive=1,
            min_train_negative=1,
            min_validation_positive=1,
            min_validation_negative=1,
        )
        self.assertTrue(report["relations"]["on"]["eligible_without_test"])
        self.assertFalse(report["test_counts_inspected_for_selection"])
        self.assertEqual(report["relations"]["on"]["test_positive"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
