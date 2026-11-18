"""Unit tests for metric registry and target metric handling (A2)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp.labels.registry import (
    METRIC_REGISTRY,
    MetricSpec,
    get_metric_spec,
    prepare_target_value,
)


class TestMetricRegistry(unittest.TestCase):
    def test_registry_entries(self):
        self.assertIn("haarpsi", METRIC_REGISTRY)
        self.assertIn("dists", METRIC_REGISTRY)

        haar = get_metric_spec("haarpsi")
        self.assertTrue(haar.higher_better)
        self.assertEqual(haar.transform, "identity")

        dists = get_metric_spec("dists")
        self.assertFalse(dists.higher_better)
        self.assertEqual(dists.transform, "one_minus")

    def test_prepare_target_value(self):
        # HaarPSI (higher = better, raw=0.82 -> target=0.82)
        val = prepare_target_value(0.82, "haarpsi")
        self.assertAlmostEqual(val, 0.82)

        # DISTS (lower = better, raw=0.18 -> target = 1.0 - 0.18 = 0.82)
        val_dists = prepare_target_value(0.18, "dists")
        self.assertAlmostEqual(val_dists, 0.82)

        # Bounds clipping
        val_clipped = prepare_target_value(-0.5, "dists")
        self.assertAlmostEqual(val_clipped, 1.0)

        # Unknown metric raises ValueError
        with self.assertRaises(ValueError):
            prepare_target_value(0.5, "non_existent_metric_xyz")

    def test_run_meta_records_target_metric(self):
        """Simulate creating run_meta.json as train.py does and verify schema."""
        from nrcomp.train import create_run_meta

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_meta = Path(tmp_dir) / "run_meta.json"
            cfg = {
                "target_metric": "dists",
                "model": {"name": "student_conditioned_b"},
                "training": {"lr": 0.0001, "batch_size": 16},
            }
            create_run_meta(cfg, out_meta, seed=42)

            self.assertTrue(out_meta.exists())
            with open(out_meta, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["target_metric"], "dists")
            self.assertEqual(data["target_transform"], "one_minus")
            self.assertEqual(data["seed"], 42)
            self.assertIn("git_sha", data)


if __name__ == "__main__":
    unittest.main()
