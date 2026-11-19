"""Unit tests for KonIQ stratified split generation and disjointness."""

import sys
import unittest
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nrcomp.data.splits import create_koniq_stratified_split


class TestSplits(unittest.TestCase):
    def test_stratified_split_disjointness_and_ratios(self):
        # Create 1000 synthetic items across 5 MOS quintiles
        items = [(f"img_{i:04d}", (i / 1000.0) * 100.0) for i in range(1000)]

        split_map = create_koniq_stratified_split(
            items,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=0,
            num_bins=5,
        )

        train_set = {k for k, v in split_map.items() if v == "train"}
        val_set = {k for k, v in split_map.items() if v == "val"}
        test_set = {k for k, v in split_map.items() if v == "test"}

        # Total count
        self.assertEqual(len(split_map), 1000)

        # Strict disjointness
        self.assertEqual(train_set & val_set, set())
        self.assertEqual(train_set & test_set, set())
        self.assertEqual(val_set & test_set, set())

        # Exact ratio check: 800 train, 100 val, 100 test
        self.assertEqual(len(train_set), 800)
        self.assertEqual(len(val_set), 100)
        self.assertEqual(len(test_set), 100)

    def test_split_determinism(self):
        items = [(f"img_{i:03d}", float(i)) for i in range(100)]
        split1 = create_koniq_stratified_split(items, seed=0)
        split2 = create_koniq_stratified_split(items, seed=0)
        self.assertEqual(split1, split2)

    def test_audit_manifest_detects_unbalanced_splits(self):
        import csv
        import tempfile
        from scripts.audit_manifest import audit_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "manifest.csv"

            # Create an unbalanced manifest: 98 train, 1 val, 1 test
            fieldnames = [
                "dataset", "image_id", "lambda_idx", "lambda_value",
                "bpp_actual", "bpp_est", "psnr", "haarpsi", "dists", "orig_mos",
                "latent_path", "split", "height", "width"
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(100):
                    sp = "train" if i < 98 else ("val" if i == 98 else "test")
                    for l_idx in range(6):
                        writer.writerow({
                            "dataset": "koniq",
                            "image_id": f"koniq_{i:03d}.jpg",
                            "lambda_idx": l_idx,
                            "lambda_value": 0.01 * (l_idx + 1),
                            "bpp_actual": 0.1 * (l_idx + 1),
                            "bpp_est": "",
                            "psnr": 30.0 + l_idx,
                            "haarpsi": 0.5 + 0.08 * l_idx,
                            "dists": 0.3 - 0.04 * l_idx,
                            "orig_mos": float(20 + i * 0.5),
                            "latent_path": f"/dummy/koniq_{i:03d}_{l_idx}.pt",
                            "split": sp,
                            "height": 512,
                            "width": 512,
                        })

            passed, errors = audit_manifest(csv_path, check_latents_exist=False)
            self.assertFalse(passed)
            self.assertTrue(any("invalid/unbalanced" in e for e in errors))

            # Now test balanced manifest: 80 train, 10 val, 10 test
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(100):
                    sp = "train" if i < 80 else ("val" if i < 90 else "test")
                    for l_idx in range(6):
                        writer.writerow({
                            "dataset": "koniq",
                            "image_id": f"koniq_{i:03d}.jpg",
                            "lambda_idx": l_idx,
                            "lambda_value": 0.01 * (l_idx + 1),
                            "bpp_actual": 0.1 * (l_idx + 1),
                            "bpp_est": "",
                            "psnr": 30.0 + l_idx,
                            "haarpsi": 0.5 + 0.08 * l_idx,
                            "dists": 0.3 - 0.04 * l_idx,
                            "orig_mos": float(20 + i * 0.5),
                            "latent_path": f"/dummy/koniq_{i:03d}_{l_idx}.pt",
                            "split": sp,
                            "height": 512,
                            "width": 512,
                        })

            passed, errors = audit_manifest(csv_path, check_latents_exist=False)
            self.assertTrue(passed, f"Expected pass, got errors: {errors}")


if __name__ == "__main__":
    unittest.main()
