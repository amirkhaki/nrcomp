"""Unit tests for LatentDataset filtering, mock indexing, and normalization."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nrcomp.data.dataset import (
    LatentDataset,
    compute_train_norm_stats,
)


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.sample_rows = [
            {
                "image_id": "img_001",
                "split": "train",
                "lambda_idx": 0,
                "latent_path": "data/latents/koniq/lambda_0/img_001.npz",
                "haarpsi": 0.85,
            },
            {
                "image_id": "img_001",
                "split": "train",
                "lambda_idx": 1,
                "latent_path": "data/latents/koniq/lambda_1/img_001.npz",
                "haarpsi": 0.88,
            },
            {
                "image_id": "img_002",
                "split": "val",
                "lambda_idx": 0,
                "latent_path": "data/latents/koniq/lambda_0/img_002.npz",
                "haarpsi": 0.72,
            },
            {
                "image_id": "img_003",
                "split": "train",
                "lambda_idx": 0,
                "latent_path": "data/latents/koniq/lambda_0/img_003.npz",
                "haarpsi": None,  # Missing target - should be excluded
            },
        ]

    def test_dataset_split_filtering(self):
        mock_store = MagicMock()
        train_ds = LatentDataset(self.sample_rows, latent_store=mock_store, split="train")
        val_ds = LatentDataset(self.sample_rows, latent_store=mock_store, split="val")

        self.assertEqual(len(train_ds), 2)  # img_003 excluded due to None haarpsi
        self.assertEqual(len(val_ds), 1)

    def test_image_to_indices_mapping(self):
        mock_store = MagicMock()
        train_ds = LatentDataset(self.sample_rows, latent_store=mock_store, split="train")
        mapping = train_ds.get_image_to_indices()

        self.assertIn("img_001", mapping)
        self.assertEqual(mapping["img_001"], [0, 1])

    def test_compute_norm_stats_fallback(self):
        mock_store = MagicMock()
        train_ds = LatentDataset(self.sample_rows, latent_store=mock_store, split="train")
        stats = compute_train_norm_stats(train_ds, max_samples=10)
        self.assertIn("y", stats)
        mean_y, std_y = stats["y"]
        self.assertEqual(len(mean_y), 320)
        self.assertEqual(len(std_y), 320)


if __name__ == "__main__":
    unittest.main()
