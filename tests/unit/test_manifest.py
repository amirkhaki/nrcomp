"""Unit tests for Manifest management, dataclass validation, and atomic I/O."""

import tempfile
import unittest
from pathlib import Path

from nrcomp.data.manifest import (
    ManifestManager,
    ManifestRow,
    validate_manifest_dict,
)


class TestManifest(unittest.TestCase):
    def setUp(self):
        self.sample_row = {
            "dataset": "koniq",
            "image_id": "1001.jpg",
            "split": "train",
            "lambda_idx": 0,
            "lambda_value": 0.004,
            "orig_path": "data/raw/koniq-10k/1001.jpg",
            "latent_path": "data/latents/koniq/lambda_0/1001.npz",
            "recon_path": None,
            "height": 1024,
            "width": 768,
            "padded_height": 1024,
            "padded_width": 768,
            "bpp_actual": 0.125,
            "bpp_est": 0.120,
            "psnr": 30.5,
            "haarpsi": 0.85,
            "dists": 0.15,
            "orig_mos": 62.4,
        }

    def test_validation(self):
        self.assertTrue(validate_manifest_dict(self.sample_row))

        # Missing required key
        invalid_row = dict(self.sample_row)
        del invalid_row["orig_mos"]
        self.assertFalse(validate_manifest_dict(invalid_row))

    def test_dataclass_roundtrip(self):
        obj = ManifestRow.from_dict(self.sample_row)
        d = obj.to_dict()
        self.assertEqual(d["dataset"], "koniq")
        self.assertEqual(d["haarpsi"], 0.85)

    def test_manifest_manager_atomic_io(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_file = Path(tmp_dir) / "manifest.csv"
            manager = ManifestManager(manifest_file)
            self.assertEqual(len(manager), 0)

            # Add row
            manager.add_or_update_row(self.sample_row)
            self.assertEqual(len(manager), 1)

            # Update row with new label
            updated_row = dict(self.sample_row)
            updated_row["haarpsi"] = 0.92
            manager.add_or_update_row(updated_row)
            self.assertEqual(len(manager), 1)
            self.assertEqual(manager.rows[0]["haarpsi"], 0.92)

            # Save and reload
            manager.save()
            new_manager = ManifestManager(manifest_file)
            self.assertEqual(len(new_manager), 1)
            self.assertEqual(float(new_manager.rows[0]["haarpsi"]), 0.92)


if __name__ == "__main__":
    unittest.main()
