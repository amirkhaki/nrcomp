"""Unit tests for SourceImageBatchSampler and collate_latent_samples."""

import sys
import unittest
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nrcomp.data.dataset import (
    MockArray,
    SourceImageBatchSampler,
    collate_latent_samples,
)


class TestBatching(unittest.TestCase):
    def test_sampler_structure(self):
        # 20 images, each having 6 lambdas (indices 0..5, 6..11, ...)
        image_to_indices = {
            f"img_{i}": list(range(i * 6, (i + 1) * 6)) for i in range(20)
        }

        batch_images = 4
        k = 3  # 3 lambdas per image
        sampler = SourceImageBatchSampler(
            image_to_indices,
            batch_images=batch_images,
            lambdas_per_image=k,
            shuffle=False,
            seed=42,
        )

        batches = list(sampler)
        self.assertEqual(len(batches), 20 // 4)  # 5 batches

        for batch in batches:
            self.assertEqual(len(batch), batch_images * k)  # 12 items per batch

            # Map batch indices back to images
            sampled_images = [f"img_{idx // 6}" for idx in batch]
            unique_images = set(sampled_images)
            self.assertEqual(len(unique_images), batch_images)

            # Each image must appear exactly k times
            for img in unique_images:
                self.assertEqual(sampled_images.count(img), k)

    def test_collate_spatial_padding(self):
        # Create 2 items with different spatial resolutions using MockArray
        # Item 1: (4, 10, 10) filled with 1.0
        y1 = MockArray((4, 10, 10), fill=1.0)
        m1 = MockArray((10, 10), fill=True)
        item1 = {
            "y_hat": y1,
            "valid_mask": m1,
            "image_id": "img_1",
            "lambda_idx": 0,
            "target": 0.85,
        }

        # Item 2: (4, 15, 12) filled with 2.0
        y2 = MockArray((4, 15, 12), fill=2.0)
        m2 = MockArray((15, 12), fill=True)
        item2 = {
            "y_hat": y2,
            "valid_mask": m2,
            "image_id": "img_2",
            "lambda_idx": 1,
            "target": 0.65,
        }

        collated = collate_latent_samples([item1, item2])

        # Batch size 2, channels 4, max_h 15, max_w 12
        y_batch = collated["y_hat"]
        mask_batch = collated["valid_mask"]

        self.assertEqual(y_batch.shape, (2, 4, 15, 12))
        self.assertEqual(mask_batch.shape, (2, 15, 12))

        # Check that item1 was properly zero-padded outside (10, 10)
        val0 = y_batch[0, 0, 12, 11]
        self.assertEqual(val0.item() if hasattr(val0, "item") else val0, 0.0)
        m0 = mask_batch[0, 12, 11]
        self.assertFalse(bool(m0.item() if hasattr(m0, "item") else m0))

        # Check that item1 has valid 1.0 inside (10, 10)
        val1 = y_batch[0, 0, 5, 5]
        self.assertEqual(val1.item() if hasattr(val1, "item") else val1, 1.0)
        m1 = mask_batch[0, 5, 5]
        self.assertTrue(bool(m1.item() if hasattr(m1, "item") else m1))

        # Check item2 has 2.0 inside (15, 12)
        val2 = y_batch[1, 0, 14, 11]
        self.assertEqual(val2.item() if hasattr(val2, "item") else val2, 2.0)
        m2 = mask_batch[1, 14, 11]
        self.assertTrue(bool(m2.item() if hasattr(m2, "item") else m2))


if __name__ == "__main__":
    unittest.main()
