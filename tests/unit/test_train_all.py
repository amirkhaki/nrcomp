"""Unit tests for scripts/train_all.py and Makefile training targets (pure Python)."""

import subprocess
import sys
import unittest
from pathlib import Path

from nrcomp.config import load_config
from scripts.train_all import DEFAULT_ABLATION_CONFIGS, DEFAULT_PRIMARY_CONFIGS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestTrainAllSuite(unittest.TestCase):
    def test_all_configs_exist_and_are_valid_yaml(self):
        all_configs = DEFAULT_PRIMARY_CONFIGS + DEFAULT_ABLATION_CONFIGS
        for cfg_rel in all_configs:
            cfg_p = PROJECT_ROOT / cfg_rel
            self.assertTrue(cfg_p.exists(), f"Config file missing: {cfg_p}")
            cfg = load_config(cfg_p)
            self.assertIn("stage", cfg)
            self.assertEqual(cfg["stage"], "train")
            self.assertIn("model", cfg)
            self.assertIn("loss", cfg)
            self.assertIn("training", cfg)
            self.assertIn("config_name", cfg)

    def test_dry_run_execution(self):
        cmd = [sys.executable, "scripts/train_all.py", "--dry-run"]
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"train_all.py --dry-run failed: {res.stderr}")
        self.assertIn("Total Runs to Execute: 8", res.stdout)
        self.assertIn("configs/train_blind.yaml", res.stdout)
        self.assertIn("configs/train_conditioned.yaml", res.stdout)
        self.assertIn("configs/train_config_b.yaml", res.stdout)
        self.assertIn("configs/train_no_rank.yaml", res.stdout)
        self.assertIn("configs/train_pure_rank.yaml", res.stdout)
        self.assertIn("configs/train_unmasked.yaml", res.stdout)
        self.assertIn("configs/train_teacher_dists.yaml", res.stdout)
        self.assertIn("configs/train_loss_l1.yaml", res.stdout)

    def test_primary_only_and_ablations_only_dry_run(self):
        cmd_p = [sys.executable, "scripts/train_all.py", "--primary-only", "--dry-run"]
        res_p = subprocess.run(cmd_p, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        self.assertEqual(res_p.returncode, 0)
        self.assertIn("Total Runs to Execute: 3", res_p.stdout)

        cmd_a = [sys.executable, "scripts/train_all.py", "--ablations-only", "--dry-run"]
        res_a = subprocess.run(cmd_a, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        self.assertEqual(res_a.returncode, 0)
        self.assertIn("Total Runs to Execute: 5", res_a.stdout)

    def test_makefile_contains_targets(self):
        makefile_path = PROJECT_ROOT / "Makefile"
        self.assertTrue(makefile_path.exists())
        content = makefile_path.read_text(encoding="utf-8")
        self.assertIn("train-all:", content)
        self.assertIn("train-all-seeds:", content)
        self.assertIn("retrain-all:", content)
        self.assertIn("train-primary:", content)
        self.assertIn("train-ablations:", content)
        self.assertIn("train-pure-rank:", content)
        self.assertIn("train-teacher-dists:", content)
        self.assertIn("train-loss-l1:", content)


if __name__ == "__main__":
    unittest.main()
