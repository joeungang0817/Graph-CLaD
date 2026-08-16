import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.phase3.build_eval_manifest import _expand_config_environment


ROOT = Path(__file__).resolve().parents[1]


class KCloudVpnConfigTest(unittest.TestCase):
    def test_linux_configs_are_portable_and_use_persistent_root_guard(self):
        for name in (
            "phase3_kcloudvpn_linux_eval_manifest_v2.json",
            "phase3_kcloudvpn_linux_pair_local_temporal_action_alignment_seed0_v1.json",
            "phase3_kcloudvpn_linux_pair_local_temporal_threefold_seed0_v1.json",
        ):
            with self.subTest(config=name):
                payload = json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
                encoded = json.dumps(payload)
                self.assertNotIn("/content/drive", encoded)
                self.assertIn("${GRAPH_CLAD_ARTIFACT_ROOT}", encoded)
        for name in (
            "phase3_kcloudvpn_linux_pair_local_temporal_action_alignment_seed0_v1.json",
            "phase3_kcloudvpn_linux_pair_local_temporal_threefold_seed0_v1.json",
        ):
            payload = json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
            runtime = payload["runtime"]
            self.assertTrue(runtime["require_cuda"])
            self.assertTrue(runtime["require_persistent_output"])
            self.assertFalse(runtime["require_persistent_drive_output"])

    def test_environment_expansion_keeps_json_types(self):
        with tempfile.TemporaryDirectory() as directory:
            old = os.environ.get("GRAPH_CLAD_ARTIFACT_ROOT")
            os.environ["GRAPH_CLAD_ARTIFACT_ROOT"] = directory
            try:
                expanded = _expand_config_environment(
                    {"root": "${GRAPH_CLAD_ARTIFACT_ROOT}/x", "flag": True, "count": 3}
                )
            finally:
                if old is None:
                    os.environ.pop("GRAPH_CLAD_ARTIFACT_ROOT", None)
                else:
                    os.environ["GRAPH_CLAD_ARTIFACT_ROOT"] = old
        self.assertEqual(Path(expanded["root"]), Path(directory) / "x")
        self.assertIs(expanded["flag"], True)
        self.assertEqual(expanded["count"], 3)


if __name__ == "__main__":
    unittest.main()
