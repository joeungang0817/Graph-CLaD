from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from scripts.phase3c.build_semantic_feature_store import (
    DecisionNCEEncoder,
    camera_inventory,
    configured_camera_frames,
    frame_digest,
    normalize_camera_image,
    required_frame_keys,
)
from scripts.phase3c.qa_camera_orientation import select_orientation_samples


class _FakeDecisionNCE:
    def eval(self):
        return self

    def parameters(self):
        return []

    def encode_images(self, images):
        import torch

        # Mean RGB plus channel-wise mean is deterministic and shape checked.
        means = images.mean(dim=(2, 3))
        return torch.cat([means, means[:, :1]], dim=1)

    def encode_texts(self, texts):
        import torch

        return torch.ones((len(texts), 4), dtype=torch.float32)


class Phase3CFeatureStoreTest(unittest.TestCase):
    def test_camera_normalization_is_explicit_and_deterministic(self):
        image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        normalized = normalize_camera_image(image, channel_order="bgr", vertical_flip=True)
        expected = image[::-1, :, ::-1]
        np.testing.assert_array_equal(normalized, expected)
        self.assertEqual(frame_digest(normalized), frame_digest(expected))

    def test_configured_camera_keys_are_exactly_two(self):
        observation = {
            "external": np.zeros((4, 5, 3), dtype=np.uint8),
            "wrist": np.ones((4, 5, 3), dtype=np.uint8),
            "proprio": np.zeros(8, dtype=np.float32),
        }
        inventory = camera_inventory(observation)
        self.assertIn("external", inventory)
        frames, report = configured_camera_frames(
            observation,
            [
                {"name": "external", "key": "external", "channel_order": "rgb", "vertical_flip": False},
                {"name": "wrist", "key": "wrist", "channel_order": "rgb", "vertical_flip": False},
            ],
        )
        self.assertEqual([frame.shape for frame in frames], [(4, 5, 3), (4, 5, 3)])
        self.assertEqual(sorted(report["selected"]), ["external", "wrist"])
        with self.assertRaises(KeyError):
            configured_camera_frames(
                observation,
                [{"key": "missing"}, {"key": "wrist"}],
            )

    def test_required_frame_keys_deduplicate_three_points(self):
        records = [
            {"task_id": 0, "demo_key": "demo_0", "prev_step": 0, "current_step": 6, "target_step": 12},
            {"task_id": 0, "demo_key": "demo_0", "prev_step": 6, "current_step": 12, "target_step": 18},
        ]
        self.assertEqual(required_frame_keys(records), {(0, "demo_0"): {0, 6, 12, 18}})
        with self.assertRaises(ValueError):
            required_frame_keys([dict(records[0], target_step=6)])

    def test_decisionnce_wrapper_freezes_and_checks_dimensions(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed in the local CPU environment")
        encoder = DecisionNCEEncoder(_FakeDecisionNCE(), model_id="fake", preprocess="rgb_01")
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        features = encoder.encode_images([image, image])
        self.assertEqual(features.shape, (2, 4))
        self.assertEqual(encoder.encode_texts(["task"])[0].shape, (4,))
        self.assertEqual(encoder.feature_dim, 4)

    def test_orientation_qa_selects_deterministic_task_coverage(self):
        selected = select_orientation_samples(
            {
                (0, "demo_1"): {0, 6, 12, 18, 24},
                (0, "demo_2"): {3, 9},
                (1, "demo_0"): {0, 6, 12},
            },
            max_tasks=2,
            frames_per_task=3,
        )
        self.assertEqual(
            selected,
            [
                (0, "demo_1", 0),
                (0, "demo_1", 12),
                (0, "demo_1", 24),
                (1, "demo_0", 0),
                (1, "demo_0", 6),
                (1, "demo_0", 12),
            ],
        )

    def test_official_loader_keeps_checkpoint_for_provenance_only(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed in the local CPU environment")

        calls = []

        class FakeOfficialModule:
            @staticmethod
            def load(model_id, **kwargs):
                calls.append((model_id, kwargs))
                return _FakeDecisionNCE()

        with patch(
            "scripts.phase3c.build_semantic_feature_store.importlib.import_module",
            return_value=FakeOfficialModule,
        ) as import_module:
            encoder = DecisionNCEEncoder.load(
                {
                    "model_id": "DecisionNCE-P",
                    "python_module": "DecisionNCE",
                    "checkpoint": "/cache/DecisionNCE-P",
                    "checkpoint_argument": None,
                    "preprocess": "rgb_01",
                    "device": "cpu",
                    "load_kwargs": {"device": "cpu"},
                }
            )

        import_module.assert_called_once_with("DecisionNCE")
        self.assertEqual(calls, [("DecisionNCE-P", {"device": "cpu"})])
        self.assertEqual(encoder.model_id, "DecisionNCE-P")


if __name__ == "__main__":
    unittest.main(verbosity=2)
