import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.phase3_dataset_io import build_smoke_dataset, load_samples


def _sample(episode_id: str, split: str) -> dict:
    return {"episode_id": episode_id, "split": split, "graph_t": {}, "graph_target": {}}


class Phase3DatasetIOTest(unittest.TestCase):
    def test_outer_demo_split_overrides_temporary_nested_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.jsonl.gz"
            record = {
                "split": {"in_task": "validation", "task_generalization": "train"},
                "samples": [{"episode_id": "demo_0", "split": "train"}],
            }
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            samples = load_samples([path])
        self.assertEqual(samples[0]["split"], "validation")

    def test_load_samples_flattens_gzipped_episode_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task0.jsonl.gz"
            record = {
                "split": {"in_task": "train"},
                "samples": [_sample("e0", "train"), _sample("e0", "train")],
            }
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")

            samples = load_samples([path])

        self.assertEqual(len(samples), 2)
        self.assertEqual({sample["split"] for sample in samples}, {"train"})

    def test_build_smoke_dataset_requires_and_balances_all_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.jsonl"
            rows = [
                {"samples": [_sample(f"{split}-{index}", split)]}
                for split in ("train", "validation", "test")
                for index in range(2)
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            dataset = build_smoke_dataset([path], per_split=2)

        self.assertEqual(dataset["split_counts"], {"test": 2, "train": 2, "validation": 2})
        self.assertEqual(len(dataset["samples"]), 6)
