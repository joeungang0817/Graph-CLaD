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


if __name__ == "__main__":
    unittest.main()
