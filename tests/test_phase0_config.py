import unittest

from scripts.phase0.smoke import normalize_config


class Phase0ConfigTest(unittest.TestCase):
    def test_documented_nested_smoke_config_maps_semantic_dimension(self):
        config = {
            "purpose": "metadata",
            "smoke_test": {
                "batch_size": 3,
                "semantic_feature_dim": 32,
                "hidden_dim": 32,
            },
        }
        self.assertEqual(
            normalize_config(config),
            {"batch_size": 3, "vl_dim": 32, "hidden_dim": 32},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

