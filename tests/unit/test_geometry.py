"""Unit tests for geometry, padding arithmetic, and valid mask computation."""

import unittest
from nrcomp.codec.geometry import (
    compute_padded_dimensions,
    compute_padding_offsets,
    compute_valid_mask,
)


class TestGeometry(unittest.TestCase):
    def test_compute_padded_dimensions(self):
        # Multiples of 64 remain unchanged
        self.assertEqual(compute_padded_dimensions(64, 128), (64, 128))
        self.assertEqual(compute_padded_dimensions(1024, 768), (1024, 768))

        # 500 pads to 512
        self.assertEqual(compute_padded_dimensions(500, 500), (512, 512))

        # 1 pads to 64
        self.assertEqual(compute_padded_dimensions(1, 65), (64, 128))

    def test_compute_padding_offsets(self):
        pad_l, pad_r, pad_t, pad_b = compute_padding_offsets(500, 500, factor=64)
        self.assertEqual((pad_l, pad_t), (0, 0))
        self.assertEqual(pad_r, 12)
        self.assertEqual(pad_b, 12)

    def test_valid_mask_exact_500x500(self):
        # 500x500 padded to 512x512 with factor 64
        # Latent grid at downsample 16 is 32 x 32
        mask = compute_valid_mask(500, 500, factor=64, downsample=16)
        self.assertEqual(len(mask), 32)
        self.assertEqual(len(mask[0]), 32)

        # Cell (30, 30): center is (30*16 + 8, 30*16 + 8) = (488, 488) < 500 -> Valid
        self.assertTrue(mask[30][30])

        # Cell (31, 31): center is (31*16 + 8, 31*16 + 8) = (504, 504) >= 500 -> Invalid
        self.assertFalse(mask[31][31])

        # Mixed border cells
        self.assertFalse(mask[30][31])
        self.assertFalse(mask[31][30])

        # Total valid cells: exactly 31 x 31 = 961
        total_valid = sum(sum(1 for cell in row if cell) for row in mask)
        self.assertEqual(total_valid, 31 * 31)

    def test_valid_mask_koniq_1024x768(self):
        # Both 1024 and 768 are already multiples of 64
        # Grid is 64 x 48
        mask = compute_valid_mask(1024, 768, factor=64, downsample=16)
        self.assertEqual(len(mask), 64)
        self.assertEqual(len(mask[0]), 48)

        # All cells should be valid
        all_valid = all(all(cell for cell in row) for row in mask)
        self.assertTrue(all_valid)


if __name__ == "__main__":
    unittest.main()
