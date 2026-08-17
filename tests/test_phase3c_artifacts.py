from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.phase3c.contracts import canonical_sha256


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Phase3CArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            cls.torch = None
            return
        cls.torch = torch

    def test_run_state_marker_is_mutually_exclusive(self):
        from scripts.phase3c.io import set_run_state

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            set_run_state(root, "RUNNING", {"fold": "test_task0"})
            self.assertTrue((root / "RUNNING.json").exists())
            set_run_state(root, "FAILED", {"error": "synthetic"})
            self.assertFalse((root / "RUNNING.json").exists())
            self.assertTrue((root / "FAILED.json").exists())
            set_run_state(root, "COMPLETED", {"runtime_manifest": "runtime.json"})
            self.assertEqual(
                sorted(path.name for path in root.glob("*.json")),
                ["COMPLETED.json"],
            )

    def test_versioned_run_configs_have_frozen_scope(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "configs" / "phase3c_core_smoke_example_v1.json").read_text(
                encoding="utf-8-sig"
            )
        )
        self.assertEqual(
            config["models"],
            [
                "C3-Sem-PastAct",
                "C3-SceneSet-PastAct",
                "C3-Pair-PastAct",
                "C3-GeomMPNN-PastAct",
                "C3-RelPool-PastAct",
                "C3-RelMPNN-PastAct",
            ],
        )
        self.assertEqual(config["folds"], ["test_task0"])
        self.assertEqual(config["seeds"], [0])
        self.assertEqual(config["updates"], 20)
        self.assertEqual(config["batch_size"], 64)

        base_deadline = json.loads(
            (
                root
                / "configs"
                / "phase3c_base_deadline_threefold_seed0_v1.json"
            ).read_text(encoding="utf-8-sig")
        )
        self.assertEqual(
            base_deadline["protocol"], "phase3c-deadline-threefold-seed0-v1"
        )
        self.assertEqual(base_deadline["claim_scope"], "deadline-constrained-pilot")
        self.assertEqual(
            base_deadline["folds"],
            ["test_task0", "test_task1", "test_task2"],
        )
        self.assertEqual(base_deadline["seeds"], [0])
        self.assertEqual(base_deadline["updates"], 500)
        self.assertEqual(base_deadline["batch_size"], 64)
        self.assertEqual(base_deadline["validation_interval"], 250)
        self.assertEqual(base_deadline["recon_weight"], 0.1)
        self.assertNotIn("reconstruction_weight", base_deadline)

        core_deadline = json.loads(
            (
                root
                / "configs"
                / "phase3c_core_deadline_threefold_seed0_v1.json"
            ).read_text(encoding="utf-8-sig")
        )
        self.assertEqual(core_deadline["protocol"], base_deadline["protocol"])
        self.assertEqual(
            core_deadline["models"],
            [
                "C3-Sem-PastAct",
                "C3-RelMPNN-PastAct",
                "C3-RelPool-PastAct",
            ],
        )
        self.assertEqual(core_deadline["folds"], base_deadline["folds"])
        self.assertEqual(core_deadline["seeds"], [0])
        self.assertEqual(core_deadline["updates"], 200)
        self.assertEqual(core_deadline["batch_size"], 64)
        self.assertEqual(core_deadline["validation_interval"], 100)
        self.assertEqual(core_deadline["minimum_updates"], 200)
        self.assertEqual(core_deadline["evaluation_split"], "test")
        self.assertEqual(core_deadline["motion_weight"], 0.1)
        self.assertNotIn("test_split", core_deadline)
        self.assertNotIn("parameter_matching", core_deadline)
        self.assertEqual(
            set(core_deadline["base_checkpoints"]), set(base_deadline["folds"])
        )

        analysis_deadline = json.loads(
            (
                root
                / "configs"
                / "phase3c_analysis_deadline_threefold_seed0_v1.json"
            ).read_text(encoding="utf-8-sig")
        )
        self.assertEqual(analysis_deadline["protocol"], base_deadline["protocol"])
        self.assertEqual(
            list(analysis_deadline["prediction_files"]), core_deadline["models"]
        )
        self.assertTrue(
            all(
                len(paths) == 3
                and all(path.endswith("predictions/evaluation.jsonl.gz") for path in paths)
                for paths in analysis_deadline["prediction_files"].values()
            )
        )
        self.assertEqual(analysis_deadline["expected_folds"], base_deadline["folds"])
        self.assertEqual(analysis_deadline["expected_seeds"], [0])
        self.assertEqual(analysis_deadline["replicates"], 2000)

        benchmark = json.loads(
            (
                root / "configs" / "phase3c_core_postcache_benchmark_v1.json"
            ).read_text(encoding="utf-8-sig")
        )
        self.assertEqual(
            benchmark["protocol"], "phase3c-deadline-postcache-throughput-v1"
        )
        self.assertEqual(benchmark["claim_scope"], "technical-throughput-only")
        self.assertEqual(benchmark["models"], ["C3-Sem-PastAct"])
        self.assertEqual(benchmark["folds"], ["test_task0"])
        self.assertEqual(benchmark["updates"], 100)
        self.assertEqual(benchmark["batch_size"], 64)

        six_model = json.loads(
            (
                root
                / "configs"
                / "phase3c_core_deadline_sixmodel_threefold_seed0_v1.json"
            ).read_text(encoding="utf-8-sig")
        )
        self.assertEqual(
            six_model["models"],
            [
                "C3-Sem-PastAct",
                "C3-RelMPNN-PastAct",
                "C3-RelPool-PastAct",
                "C3-SceneSet-PastAct",
                "C3-Pair-PastAct",
                "C3-GeomMPNN-PastAct",
            ],
        )
        self.assertEqual(six_model["folds"], base_deadline["folds"])
        self.assertEqual(six_model["updates"], core_deadline["updates"])
        self.assertEqual(six_model["batch_size"], core_deadline["batch_size"])

        analysis_six = json.loads(
            (
                root
                / "configs"
                / "phase3c_analysis_deadline_sixmodel_threefold_seed0_v1.json"
            ).read_text(encoding="utf-8-sig")
        )
        self.assertEqual(list(analysis_six["prediction_files"]), six_model["models"])
        self.assertTrue(
            all(len(paths) == 3 for paths in analysis_six["prediction_files"].values())
        )

        from scripts.phase3c.select_deadline_core_scope import select_scope

        with tempfile.TemporaryDirectory() as directory:
            runtime_path = Path(directory) / "runtime_manifest.json"
            runtime = {
                "status": "completed",
                "protocol": "phase3c-deadline-postcache-throughput-v1",
                "claim_scope": "technical-throughput-only",
                "model_id": "C3-Sem-PastAct",
                "fold": "test_task0",
                "seed": 0,
                "updates": 100,
                "batch_size": 64,
                "elapsed_seconds": 100.0,
                "best_validation_macro_pr_auc": 0.99,
            }
            runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
            fast = select_scope(runtime_path, remaining_hours=10.0)
            self.assertEqual(fast["selection"], "six-model")
            self.assertEqual(fast["performance_fields_consulted"], [])

            runtime["elapsed_seconds"] = 1200.0
            runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
            medium = select_scope(runtime_path, remaining_hours=10.0)
            self.assertEqual(medium["selection"], "three-model")

            runtime["elapsed_seconds"] = 2000.0
            runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
            slow = select_scope(runtime_path, remaining_hours=10.0)
            self.assertEqual(slow["selection"], "insufficient-time")

    def test_completed_base_runtime_checks_artifact_hashes(self):
        if self.torch is None:
            self.skipTest("torch is unavailable")
        from scripts.phase3c.run_base_clad import _completed_runtime

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            best = root / "best.pt"
            last = root / "last.pt"
            stdout = root / "stdout.log"
            stderr = root / "stderr.log"
            best.write_bytes(b"best")
            last.write_bytes(b"last")
            stdout.write_bytes(b"stdout")
            stderr.write_bytes(b"stderr")
            runtime = root / "runtime_manifest.json"
            runtime.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "fold": "test_task0",
                        "seed": 0,
                        "config_sha256": "config",
                        "checkpoint": str(best),
                        "checkpoint_sha256": _sha(best),
                        "resume_checkpoint": str(last),
                        "resume_checkpoint_sha256": _sha(last),
                        "stdout_log": str(stdout),
                        "stdout_log_sha256": _sha(stdout),
                        "stderr_log": str(stderr),
                        "stderr_log_sha256": _sha(stderr),
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNotNone(
                _completed_runtime(
                    runtime,
                    fold="test_task0",
                    seed=0,
                    config_sha256="config",
                )
            )
            with self.assertRaisesRegex(ValueError, "current code"):
                _completed_runtime(
                    runtime,
                    fold="test_task0",
                    seed=0,
                    config_sha256="config",
                    code_sha256="different-code",
                )
            best.write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                _completed_runtime(
                    runtime,
                    fold="test_task0",
                    seed=0,
                    config_sha256="config",
                )

    def test_completed_core_runtime_checks_all_artifact_hashes(self):
        if self.torch is None:
            self.skipTest("torch is unavailable")
        from scripts.phase3c.run_core import _completed_runtime

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {}
            for name in (
                "checkpoint",
                "resume_checkpoint",
                "metrics",
                "predictions",
                "stdout_log",
                "stderr_log",
            ):
                path = root / name
                path.write_bytes(name.encode("ascii"))
                artifacts[name] = path
            runtime_value = {
                "status": "completed",
                "model_id": "C3-RelMPNN-PastAct",
                "fold": "test_task0",
                "seed": 0,
                "config_sha256": "config",
            }
            for name, path in artifacts.items():
                runtime_value[name] = str(path)
                runtime_value[f"{name}_sha256"] = _sha(path)
            runtime = root / "runtime_manifest.json"
            runtime.write_text(json.dumps(runtime_value), encoding="utf-8")
            self.assertIsNotNone(
                _completed_runtime(
                    runtime,
                    model_id="C3-RelMPNN-PastAct",
                    fold="test_task0",
                    seed=0,
                    config_sha256="config",
                )
            )
            with self.assertRaisesRegex(ValueError, "current code"):
                _completed_runtime(
                    runtime,
                    model_id="C3-RelMPNN-PastAct",
                    fold="test_task0",
                    seed=0,
                    config_sha256="config",
                    trainer_source_sha256="different-trainer",
                )
            artifacts["metrics"].write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                _completed_runtime(
                    runtime,
                    model_id="C3-RelMPNN-PastAct",
                    fold="test_task0",
                    seed=0,
                    config_sha256="config",
                )

    def test_base_trainer_writes_best_and_resumes_last(self):
        if self.torch is None:
            self.skipTest("torch is unavailable")
        import torch
        import torch.nn as nn

        from scripts.phase3c import train_base_clad

        class FakeNormalization:
            def to_dict(self):
                return {"fake": True}

        class FakeStore:
            def __init__(self, root):
                self.root = Path(root)
                self.feature_dim = 4
                self.manifest = {
                    "source": {"joined_manifest_sha256": _sha(joined)}
                }

            def close(self):
                pass

            def verify_integrity(self, **kwargs):
                return {"verified_shards": 1, "orientation_attestation_sha256": "qa"}

        class FakeControlled(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))

            def training_loss(self, batch, *, action_mask_ratio=0.3):
                value = self.weight.square()
                return {
                    "loss_p": value,
                    "loss_s": value,
                    "loss_p_recon": value,
                    "loss_v_recon": value,
                }

            def update_ema_after_optimizer_step(self):
                pass

            def ema_initialization_state(self):
                return None

            def restore_ema_initialization_state(self, value):
                self.restored_ema_state = value

        def infinite_batches(*args, **kwargs):
            while True:
                yield [{"sample_id": "train"}]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            joined = root / "joined.jsonl"
            joined.write_text("{}\n", encoding="utf-8")
            store_root = root / "store"
            store_root.mkdir()
            (store_root / "manifest.json").write_text("{}\n", encoding="utf-8")
            output = root / "output"
            config = {
                "joined_manifest": str(joined),
                "semantic_store": str(store_root),
                "output_root": str(output),
                "fold": "test_task0",
                "held_out_task_id": 0,
                "seed": 0,
                "updates": 2,
                "batch_size": 1,
                "shuffle_buffer": 1,
                "vl_dim": 4,
                "hidden_dim": 4,
                "validation_interval": 1,
                "device": "cpu",
                "resume": True,
            }
            patches = (
                patch.object(train_base_clad, "SemanticFeatureStore", FakeStore),
                patch.object(train_base_clad, "ControlledCLaD", FakeControlled),
                patch.object(train_base_clad, "fit_normalization", return_value=FakeNormalization()),
                patch.object(
                    train_base_clad,
                    "iter_filtered_records",
                    side_effect=lambda *args, **kwargs: iter([{"sample_id": "validation"}]),
                ),
                patch.object(train_base_clad, "iter_shuffled_batches", side_effect=infinite_batches),
                patch.object(train_base_clad, "collate_phase3c", return_value={}),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                first = train_base_clad.train(config)
                second = train_base_clad.train(config)

            self.assertEqual(first["schema"], "phase3c-base-clad-run.v4")
            self.assertTrue((output / "checkpoints" / "best.pt").exists())
            self.assertTrue((output / "checkpoints" / "last.pt").exists())
            self.assertEqual(second["resumed_from_update"], 2)
            self.assertEqual(second["selected_update"], first["selected_update"])

    def test_core_runner_expands_six_models_three_folds(self):
        if self.torch is None:
            self.skipTest("torch is unavailable")
        from scripts.phase3c import run_core

        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def fake_train(config):
                calls.append(dict(config))
                return {
                    "status": "completed",
                    "model_id": config["model_id"],
                    "fold": config["fold"],
                    "seed": config["seed"],
                    "updates": config["updates"],
                }

            match = {
                "selected": {"width": 64, "parameters": 100},
                "within_tolerance": True,
                "relative_error": 0.0,
            }
            config = {
                "output_root": str(Path(directory) / "core"),
                "base_checkpoints": {
                    "test_task0": "base0.pt",
                    "test_task1": "base1.pt",
                    "test_task2": "base2.pt",
                },
                "models": list(run_core.CORE_MODELS),
                "folds": ["test_task0", "test_task1", "test_task2"],
                "seeds": [0],
                "updates": 10000,
                "vl_dim": 4,
                "structured_dim": 8,
                "parameter_reference_width": 64,
            }
            with patch.object(run_core, "train", side_effect=fake_train), patch.object(
                run_core, "select_width", return_value=match
            ), patch.object(run_core, "trainable_parameter_count", return_value=100):
                result = run_core.run(config)

            self.assertEqual(result["schema"], "phase3c-core-screen.v4")
            self.assertEqual(len(calls), 18)
            self.assertTrue(all(call["updates"] == 10000 for call in calls))
            self.assertEqual(
                {(call["model_id"], call["fold"]) for call in calls},
                {
                    (model, fold)
                    for model in run_core.CORE_MODELS
                    for fold in ("test_task0", "test_task1", "test_task2")
                },
            )

    def test_core_trainer_writes_best_and_resumes_last(self):
        if self.torch is None:
            self.skipTest("torch is unavailable")
        import torch
        import torch.nn as nn

        from scripts.phase3c import train_core
        from scripts.phase3c.contracts import PRIMARY_RELATIONS

        class FakeNormalization:
            def to_dict(self):
                return {"fake": True}

        class FakeStore:
            def __init__(self, root):
                self.feature_dim = 4
                self.manifest = {
                    "source": {"joined_manifest_sha256": _sha(joined)}
                }

            def close(self):
                pass

            def verify_integrity(self, **kwargs):
                return {"verified_shards": 1, "orientation_attestation_sha256": "qa"}

        class FakeClad(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()

            def encode_foresight(self, batch):
                return torch.zeros((1, 8), dtype=torch.float32)

        class FakeStructured(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

        class FakeAdapter(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.5))

            def forward(self, batch, semantic):
                return {
                    "relation_logits": self.weight.expand(1, len(PRIMARY_RELATIONS)),
                    "scene_motion": self.weight.reshape(1, 1),
                }

        class FakeBatch:
            target_relation_change = torch.zeros((1, len(PRIMARY_RELATIONS)))
            target_relation_mask = torch.ones((1, len(PRIMARY_RELATIONS)))
            target_scene_motion = torch.zeros((1, 1))

        def infinite_batches(*args, **kwargs):
            while True:
                yield [{"sample_id": "train"}]

        validation_calls = 0

        def fake_evaluate(*, adapter, split, prediction_path=None, **kwargs):
            nonlocal validation_calls
            if prediction_path is None:
                score = 0.5 + 0.1 * min(validation_calls, 1)
                validation_calls += 1
            else:
                score = 0.55
            per_relation = {
                name: {
                    "threshold": 0.5,
                    "pr_auc": score,
                    "f1": 0.5,
                    "valid_count": 2,
                    "positive_count": 1,
                    "brier": 0.25,
                    "ece_10bin": 0.1,
                }
                for name in PRIMARY_RELATIONS
            }
            if prediction_path is not None:
                Path(prediction_path).parent.mkdir(parents=True, exist_ok=True)
                Path(prediction_path).write_bytes(b"predictions")
            return {
                "status": "completed",
                "split": split,
                "rows": 1,
                "relation": {
                    "per_relation": per_relation,
                    "macro_pr_auc": score,
                    "family_macro_pr_auc": score,
                    "macro_f1": 0.5,
                    "family_macro_f1": 0.5,
                },
                "motion": {"mae": 0.1, "rmse": 0.1},
                "no_change_fpr": {"mean": 0.0},
                "prediction_path": str(prediction_path) if prediction_path else None,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            joined = root / "joined.jsonl"
            joined.write_text("{}\n", encoding="utf-8")
            store_root = root / "store"
            store_root.mkdir()
            store_manifest = store_root / "manifest.json"
            store_manifest.write_text("{}\n", encoding="utf-8")
            base = root / "base.pt"
            torch.save(
                {
                    "schema": "phase3c-base-clad-checkpoint.v4",
                    "kind": "validation_best",
                    "model_state": {},
                    "source_joined_manifest_sha256": _sha(joined),
                    "semantic_store_manifest_sha256": _sha(store_manifest),
                    "semantic_store_integrity_sha256": canonical_sha256(
                        {
                            "verified_shards": 1,
                            "orientation_attestation_sha256": "qa",
                        }
                    ),
                    "vl_dim": 4,
                    "hidden_dim": 4,
                },
                base,
            )
            output = root / "core"
            config = {
                "joined_manifest": str(joined),
                "semantic_store": str(store_root),
                "base_checkpoint": str(base),
                "output_root": str(output),
                "model_id": "C3-Sem-PastAct",
                "fold": "test_task0",
                "held_out_task_id": 0,
                "seed": 0,
                "updates": 2,
                "batch_size": 1,
                "shuffle_buffer": 1,
                "validation_interval": 1,
                "early_stopping_patience": 2,
                "minimum_updates": 2,
                "vl_dim": 4,
                "hidden_dim": 4,
                "structured_dim": 8,
                "device": "cpu",
                "resume": True,
            }
            support = {
                name: {"positive": 20, "negative": 20}
                for name in PRIMARY_RELATIONS
            }
            with patch.object(train_core, "SemanticFeatureStore", FakeStore), patch.object(
                train_core, "ControlledCLaD", FakeClad
            ), patch.object(train_core, "SemanticPastActEncoder", FakeStructured), patch.object(
                train_core, "Phase3CAdapter", FakeAdapter
            ), patch.object(train_core, "trainable_parameter_count", return_value=1), patch.object(
                train_core,
                "_relation_pos_weights",
                return_value=([1.0] * len(PRIMARY_RELATIONS), support, 1.0),
            ), patch.object(train_core, "_relation_counts", return_value=support), patch.object(
                train_core, "fit_normalization", return_value=FakeNormalization()
            ), patch.object(
                train_core,
                "iter_filtered_records",
                side_effect=lambda *args, **kwargs: iter([{"sample_id": "row"}]),
            ), patch.object(
                train_core, "iter_shuffled_batches", side_effect=infinite_batches
            ), patch.object(train_core, "collate_phase3c", return_value=FakeBatch()), patch.object(
                train_core,
                "relation_motion_loss",
                side_effect=lambda prediction, *args, **kwargs: {
                    "loss": prediction["relation_logits"].mean().square()
                },
            ), patch.object(train_core, "_evaluate_adapter", side_effect=fake_evaluate):
                first = train_core.train(config)
                second = train_core.train(config)

            self.assertEqual(first["schema"], "phase3c-core-run.v4")
            self.assertTrue((output / "checkpoints" / "best.pt").exists())
            self.assertTrue((output / "checkpoints" / "last.pt").exists())
            self.assertEqual(second["resumed_from_update"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
