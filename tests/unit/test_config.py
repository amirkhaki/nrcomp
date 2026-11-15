"""Unit tests for configuration parsing and schema validation."""

import sys
import unittest
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nrcomp.config import load_config, validate_config


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.config_dir = Path(__file__).resolve().parent.parent.parent / "configs"

    def test_load_all_stage_configs(self):
        expected_configs = [
            "codec_sanity.yaml",
            "generate_pilot.yaml",
            "generate_full.yaml",
            "label.yaml",
            "train_blind.yaml",
            "train_conditioned.yaml",
            "train_config_b.yaml",
            "evaluate.yaml",
        ]

        for cfg_name in expected_configs:
            cfg_path = self.config_dir / cfg_name
            self.assertTrue(cfg_path.exists(), f"Missing config file: {cfg_name}")
            cfg = load_config(cfg_path)
            self.assertIsInstance(cfg, dict)
            self.assertIn("stage", cfg)

    def test_validate_config_keys(self):
        cfg = {"stage": "train", "lr": 0.001}
        self.assertTrue(validate_config(cfg, ["stage", "lr"]))

        with self.assertRaises(ValueError):
            validate_config(cfg, ["stage", "epochs"])


if __name__ == "__main__":
    unittest.main()
