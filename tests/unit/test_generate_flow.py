"""Unit tests for Stage 2 generate pipeline components."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nrcomp.data.store import check_disk_space
from nrcomp.generate import collect_dataset_images, log_failure


class TestGeneratePipeline(unittest.TestCase):
    def test_check_disk_space(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            has_space, free_b, req_b = check_disk_space(tmp_dir, num_samples=10, bytes_per_sample=1024)
            self.assertTrue(has_space)
            self.assertGreater(free_b, 0)
            self.assertEqual(req_b, int(10 * 1024 * 1.1))

    def test_log_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fail_csv = Path(tmp_dir) / "failures.csv"
            log_failure(fail_csv, "koniq", "img_01.png", 0, "Test error message")
            self.assertTrue(fail_csv.exists())
            with open(fail_csv, "r") as f:
                content = f.read()
                self.assertIn("img_01.png", content)
                self.assertIn("Test error message", content)

    def test_collect_dataset_images_max_images(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            img_dir = tmp_p / "images"
            img_dir.mkdir()
            for i in range(20):
                (img_dir / f"img_{i:03d}.png").touch()

            d_cfg = {
                "name": "dummy",
                "image_dir": str(img_dir),
                "max_images": 5,
            }
            records = collect_dataset_images(d_cfg, tmp_p)
            self.assertEqual(len(records), 5)


if __name__ == "__main__":
    unittest.main()
