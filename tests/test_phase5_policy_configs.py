from __future__ import annotations

import json
import unittest
from pathlib import Path


class Phase5PolicyConfigTest(unittest.TestCase):
    def test_graph_configs_use_completed_three_model_core_checkpoint(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for filename in (
            "phase5_policy_graph_task0_smoke_v1.json",
            "phase5_policy_graph_task0_pilot_v1.json",
            "phase5_policy_graph_task0_smoke_v2.json",
            "phase5_policy_graph_task0_pilot_v2.json",
            "phase5_policy_graph_task0_deadline_v3.json",
        ):
            config = json.loads((root / "configs" / filename).read_text(encoding="utf-8"))
            section = config["stage2_policy"]
            checkpoint = section["structured_checkpoint"]
            self.assertEqual(section["structured_model_id"], "C3-RelMPNN-PastAct")
            self.assertIn("/core_deadline_threefold_seed0_v1/", checkpoint)
            self.assertNotIn("/core_deadline_sixmodel_threefold_seed0_v1/", checkpoint)
            self.assertTrue(
                checkpoint.endswith(
                    "/C3-RelMPNN-PastAct/test_task0/seed0/checkpoints/best.pt"
                )
            )

    def test_v2_configs_share_the_immutable_v2_manifest_root(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for filename in (
            "phase5_policy_semantic_task0_smoke_v2.json",
            "phase5_policy_graph_task0_smoke_v2.json",
            "phase5_policy_semantic_task0_pilot_v2.json",
            "phase5_policy_graph_task0_pilot_v2.json",
        ):
            config = json.loads((root / "configs" / filename).read_text(encoding="utf-8"))
            section = config["stage2_policy"]
            self.assertIn("/stage2_pilot_v2/", section["policy_manifest"])
            self.assertIn("/stage2_pilot_v2/", section["output_root"])

        manifest_config = json.loads(
            (root / "configs" / "phase5_policy_manifest_task0_v2.json").read_text(
                encoding="utf-8"
            )
        )["stage2_policy_manifest"]
        self.assertIn("/stage2_pilot_v2/", manifest_config["output"])
        self.assertIn("/stage2_pilot_v2/", manifest_config["qa_output"])

    def test_deadline_v3_configs_use_matched_four_thousand_updates(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for filename in (
            "phase5_policy_semantic_task0_deadline_v3.json",
            "phase5_policy_graph_task0_deadline_v3.json",
        ):
            config = json.loads((root / "configs" / filename).read_text(encoding="utf-8"))
            section = config["stage2_policy"]
            self.assertEqual(section["updates"], 4000)
            self.assertEqual(section["batch_size"], 64)
            self.assertEqual(section["validation_interval"], 500)
            self.assertIn("/stage2_deadline_v3/", section["output_root"])


if __name__ == "__main__":
    unittest.main()
