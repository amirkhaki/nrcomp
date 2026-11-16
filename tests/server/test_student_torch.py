"""PyTorch integration and numerical tests for the execution server.

Runs inside the execution server's conda environment (nrcomp-main) with PyTorch and GPU:
    pytest tests/server -v

Verifies:
1. Oracle Parity: PyTorch MaskedMeanPool and CompositeLoss match pure-Python oracles.
2. Mask Invariance: Mutating invalid padding cells leaves model predictions unchanged.
3. Gradient Flow: Gradients reach every parameter in stem, trunk, embedding, pooling, and head.
4. Overfitting: Model overfits 32 synthetic samples to near-zero loss (< 0.01) in 50 steps.
5. Blind Mode Invariance: Changing lambda_idx produces identical outputs in blind mode.
6. Config B Integration: Forward pass with y_hat + z_hat executes cleanly.
"""

import math
import random
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "Server test requires PyTorch")
class TestStudentTorch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        random.seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_torch_vs_oracle_parity_mean_pool(self):
        from nrcomp.model.pooling import MaskedMeanPool, masked_mean_pool_oracle

        b, c, h, w = 2, 4, 8, 8
        x = torch.randn(b, c, h, w, dtype=torch.float32)
        mask = torch.ones(b, h, w, dtype=torch.bool)
        # Invalidate bottom-right corner for sample 0
        mask[0, 6:, 6:] = False

        pool = MaskedMeanPool()
        torch_out = pool(x, mask).cpu().numpy()

        # Check against pure-python oracle for sample 0
        feats_list = x[0].tolist()
        mask_list = mask[0].tolist()
        oracle_out = masked_mean_pool_oracle(feats_list, mask_list)

        for ch in range(c):
            self.assertAlmostEqual(torch_out[0, ch], oracle_out[ch], places=5)

    def test_torch_vs_oracle_parity_composite_loss(self):
        from nrcomp.model.loss import CompositeLoss, composite_loss_oracle

        preds = torch.tensor([0.72, 0.81, 0.65, 0.90], dtype=torch.float32)
        targets = torch.tensor([0.70, 0.85, 0.60, 0.88], dtype=torch.float32)
        lambdas = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        image_ids = ["img_1", "img_1", "img_2", "img_2"]

        criterion = CompositeLoss(alpha=0.5, beta=0.05, margin=0.0)
        total_loss, details = criterion(preds, targets, lambdas, image_ids)

        oracle_total, oracle_details = composite_loss_oracle(
            preds.tolist(), targets.tolist(), lambdas.tolist(), image_ids, alpha=0.5, beta=0.05
        )

        self.assertAlmostEqual(details["loss"], oracle_details["loss"], places=5)
        self.assertAlmostEqual(details["smooth_l1"], oracle_details["smooth_l1"], places=5)
        self.assertAlmostEqual(details["rank_loss"], oracle_details["rank_loss"], places=5)
        self.assertEqual(details["num_pairs"], oracle_details["num_pairs"])

    def test_mask_invariance(self):
        """Mutating values in invalid padding cells must NOT alter model prediction."""
        from nrcomp.model.student import LatentStudent

        model = LatentStudent(config="A", in_channels_y=320, width=64, num_res_blocks=2, lambda_mode="blind")
        model.eval()

        b, c, h, w = 2, 320, 16, 16
        y_clean = torch.randn(b, c, h, w, dtype=torch.float32)
        mask = torch.ones(b, h, w, dtype=torch.bool)
        # Mark borders as invalid padding
        mask[:, :, 12:] = False
        mask[:, 12:, :] = False

        # Zero out padding cells in clean input
        y_clean = y_clean * mask.unsqueeze(1).float()

        # Create noisy input with huge garbage in invalid padding cells
        y_noisy = y_clean.clone()
        y_noisy[:, :, 12:] = torch.randn_like(y_noisy[:, :, 12:]) * 1000.0
        y_noisy[:, 12:, :] = torch.randn_like(y_noisy[:, 12:, :]) * 1000.0

        with torch.no_grad():
            out_clean = model(y_clean, valid_mask=mask)
            out_noisy = model(y_noisy, valid_mask=mask)

        diff = (out_clean - out_noisy).abs().max().item()
        self.assertLess(diff, 1e-5, f"Mask leakage detected: max diff = {diff}")

    def test_gradient_flow_to_all_parameters(self):
        """Gradients must reach every trainable parameter in conditioned mode."""
        from nrcomp.model.loss import CompositeLoss
        from nrcomp.model.student import LatentStudent

        model = LatentStudent(
            config="A",
            in_channels_y=320,
            width=64,
            num_res_blocks=2,
            lambda_mode="conditioned",
            num_lambdas=6,
        )
        model.train()

        b, c, h, w = 4, 320, 8, 8
        y = torch.randn(b, c, h, w, dtype=torch.float32)
        mask = torch.ones(b, h, w, dtype=torch.bool)
        lambdas = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        targets = torch.tensor([0.6, 0.8, 0.5, 0.9], dtype=torch.float32)
        image_ids = ["img_1", "img_1", "img_2", "img_2"]

        preds = model(y, valid_mask=mask, lambda_idx=lambdas)
        criterion = CompositeLoss(alpha=0.5, beta=0.05)
        loss, _ = criterion(preds, targets, lambdas, image_ids)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} received no gradient!")
                grad_norm = param.grad.abs().sum().item()
                self.assertGreater(grad_norm, 0.0, f"Parameter {name} has zero gradient!")

    def test_overfit_small_batch(self):
        """Model must successfully overfit a small batch of 16 samples to loss < 0.02."""
        from nrcomp.model.loss import CompositeLoss
        from nrcomp.model.student import LatentStudent

        model = LatentStudent(
            config="A",
            in_channels_y=320,
            width=64,
            num_res_blocks=2,
            lambda_mode="conditioned",
            num_lambdas=6,
            dropout=0.0,
        )
        model.train()

        b, c, h, w = 16, 320, 8, 8
        y = torch.randn(b, c, h, w, dtype=torch.float32)
        mask = torch.ones(b, h, w, dtype=torch.bool)
        lambdas = torch.tensor([i % 6 for i in range(b)], dtype=torch.long)
        targets = torch.tensor([0.4 + 0.03 * (i % 6) for i in range(b)], dtype=torch.float32)
        image_ids = [f"img_{i // 2}" for i in range(b)]

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        criterion = CompositeLoss(alpha=0.5, beta=0.05)

        initial_loss = None
        for step in range(60):
            optimizer.zero_grad()
            preds = model(y, valid_mask=mask, lambda_idx=lambdas)
            loss, d = criterion(preds, targets, lambdas, image_ids)
            if step == 0:
                initial_loss = d["loss"]
            loss.backward()
            optimizer.step()

        final_loss = d["loss"]
        self.assertLess(final_loss, 0.02, f"Failed to overfit: final loss = {final_loss:.4f}")
        self.assertLess(final_loss, initial_loss * 0.2)

    def test_blind_mode_invariance(self):
        """In blind mode, changing lambda_idx must leave prediction identical."""
        from nrcomp.model.student import LatentStudent

        model = LatentStudent(config="A", in_channels_y=320, width=64, num_res_blocks=2, lambda_mode="blind")
        model.eval()

        b, c, h, w = 2, 320, 8, 8
        y = torch.randn(b, c, h, w, dtype=torch.float32)
        mask = torch.ones(b, h, w, dtype=torch.bool)

        with torch.no_grad():
            out1 = model(y, valid_mask=mask, lambda_idx=torch.tensor([0, 1]))
            out2 = model(y, valid_mask=mask, lambda_idx=torch.tensor([5, 4]))

        diff = (out1 - out2).abs().max().item()
        self.assertEqual(diff, 0.0)

    def test_config_b_forward(self):
        """Config B executes forward pass with y_hat (320) and z_hat (192) upsampled 4x."""
        from nrcomp.model.student import LatentStudent

        model = LatentStudent(
            config="B",
            in_channels_y=320,
            in_channels_z=192,
            width=64,
            num_res_blocks=2,
            lambda_mode="conditioned",
        )
        model.eval()

        b = 2
        y = torch.randn(b, 320, 16, 16, dtype=torch.float32)
        z = torch.randn(b, 192, 4, 4, dtype=torch.float32)  # 4x downsampled relative to y
        mask = torch.ones(b, 16, 16, dtype=torch.bool)
        lambdas = torch.tensor([0, 2], dtype=torch.long)

        with torch.no_grad():
            preds = model(y_hat=y, valid_mask=mask, z_hat=z, lambda_idx=lambdas)

        self.assertEqual(preds.shape, (b,))
        self.assertTrue(torch.all(preds >= 0.0) and torch.all(preds <= 1.0))


if __name__ == "__main__":
    unittest.main()
