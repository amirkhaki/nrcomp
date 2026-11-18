"""Unit tests for pooling reference oracles and student model structure (pure Python)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nrcomp.model.pooling import (
    masked_attention_pool_oracle,
    masked_mean_pool_oracle,
)
from nrcomp.model.student import LatentStudent
from nrcomp.model.trunk import ConvStem, ResBlock, Trunk


class TestModelOracle(unittest.TestCase):
    def test_masked_mean_pool_oracle_all_valid(self):
        # 2 channels, 2x2 spatial
        features = [
            [[1.0, 2.0], [3.0, 4.0]],  # mean = 2.5
            [[10.0, 20.0], [30.0, 40.0]],  # mean = 25.0
        ]
        mask = [[True, True], [True, True]]
        pooled = masked_mean_pool_oracle(features, mask)
        self.assertEqual(len(pooled), 2)
        self.assertAlmostEqual(pooled[0], 2.5)
        self.assertAlmostEqual(pooled[1], 25.0)

    def test_masked_mean_pool_oracle_with_padding(self):
        # 2 channels, 2x2 spatial with (1, 1) cell padded (False)
        features = [
            [[1.0, 2.0], [3.0, 999.0]],
            [[10.0, 20.0], [30.0, 999.0]],
        ]
        mask = [[True, True], [True, False]]  # 3 valid cells
        pooled = masked_mean_pool_oracle(features, mask)
        # channel 0: (1 + 2 + 3) / 3 = 2.0
        self.assertAlmostEqual(pooled[0], 2.0)
        # channel 1: (10 + 20 + 30) / 3 = 20.0
        self.assertAlmostEqual(pooled[1], 20.0)

    def test_masked_attention_pool_oracle_mask_invariance(self):
        # Features with extreme garbage in masked cell (1, 1)
        features_clean = [
            [[1.0, 2.0], [3.0, 0.0]],
        ]
        features_garbage = [
            [[1.0, 2.0], [3.0, 1e6]],
        ]
        weights_w = [1.0]
        bias_b = 0.0
        mask = [[True, True], [True, False]]

        res_clean = masked_attention_pool_oracle(features_clean, weights_w, bias_b, mask)
        res_garbage = masked_attention_pool_oracle(features_garbage, weights_w, bias_b, mask)

        # The garbage in the masked cell must not affect the pooled output
        self.assertAlmostEqual(res_clean[0], res_garbage[0], places=4)

    def test_student_model_interface(self):
        # Verify student can be initialized and holds expected properties
        model = LatentStudent(
            config="A",
            in_channels_y=320,
            width=256,
            num_res_blocks=6,
            lambda_mode="blind",
            num_lambdas=6,
            dropout=0.1,
        )
        self.assertEqual(model.config, "A")
        self.assertEqual(model.lambda_mode, "blind")
        self.assertEqual(model.width, 256)
        self.assertEqual(model.num_res_blocks, 6)

        # In blind mode, lambda_embed must be None
        self.assertIsNone(model.lambda_embed)

    def test_student_conditioned_config(self):
        model = LatentStudent(
            config="B",
            in_channels_y=320,
            in_channels_z=192,
            width=256,
            num_res_blocks=6,
            lambda_mode="conditioned",
            num_lambdas=6,
        )
        self.assertEqual(model.config, "B")
        self.assertEqual(model.lambda_mode, "conditioned")
        self.assertTrue(model.use_mask)

    def test_student_unmasked_config(self):
        model = LatentStudent(
            config="A",
            in_channels_y=320,
            width=256,
            num_res_blocks=6,
            lambda_mode="conditioned",
            num_lambdas=6,
            use_mask=False,
        )
        self.assertFalse(model.use_mask)


if __name__ == "__main__":
    unittest.main()
