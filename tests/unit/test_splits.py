"""Unit tests for KonIQ stratified split generation and disjointness."""

import unittest
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


if __name__ == "__main__":
    unittest.main()
