"""Unit tests for LatentStore saving and loading."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nrcomp.data.store import LatentStore


class TestLatentStore(unittest.TestCase):
    def test_save_and_load_latents_npz(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = LatentStore(root_dir=tmp_dir, store_format="fp16")
            path = store.get_path("koniq", "test_img.jpg", 0)
            self.assertEqual(path.suffix, ".npz")

            # Mock data as nested lists
            y_hat = [[[0.5, -0.2], [1.0, 0.0]]]
            z_hat = [[[0.1, -0.1]]]
            valid_mask = [[True, True]]

            # If numpy is available, test real saving/loading
            try:
                import numpy as np
                store.save_latents(path, y_hat, z_hat, valid_mask)
                self.assertTrue(path.exists())

                loaded = store.load_latents(path)
                self.assertIn("y_hat", loaded)
                self.assertIn("z_hat", loaded)
                self.assertIn("valid_mask", loaded)
                self.assertEqual(loaded["y_hat"].dtype, np.float16)
            except ImportError:
                pass


if __name__ == "__main__":
    unittest.main()
