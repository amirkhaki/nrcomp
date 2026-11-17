"""Unit tests for EarlyStopping logic (pure Python)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nrcomp.train import EarlyStopping


class TestEarlyStopping(unittest.TestCase):
    def test_improvement_resets_counter(self):
        stopper = EarlyStopping(patience=3, min_delta=0.01, warmup_epochs=2)
        # Epoch 1
        improved = stopper.step(0.50, epoch=1)
        self.assertTrue(improved)
        self.assertEqual(stopper.counter, 0)
        self.assertFalse(stopper.early_stop)

        # Epoch 2 (warmup)
        improved = stopper.step(0.55, epoch=2)
        self.assertTrue(improved)
        self.assertEqual(stopper.counter, 0)

        # Epoch 3: slight improvement exceeding min_delta
        improved = stopper.step(0.57, epoch=3)
        self.assertTrue(improved)
        self.assertEqual(stopper.counter, 0)

    def test_patience_triggers_early_stop(self):
        stopper = EarlyStopping(patience=3, min_delta=0.01, warmup_epochs=1)
        # Epoch 1: initial score
        stopper.step(0.60, epoch=1)

        # Epoch 2: no improvement (delta < 0.01)
        improved = stopper.step(0.605, epoch=2)
        self.assertFalse(improved)
        self.assertEqual(stopper.counter, 1)
        self.assertFalse(stopper.early_stop)

        # Epoch 3: worse score
        improved = stopper.step(0.59, epoch=3)
        self.assertFalse(improved)
        self.assertEqual(stopper.counter, 2)
        self.assertFalse(stopper.early_stop)

        # Epoch 4: still no improvement -> triggers
        improved = stopper.step(0.60, epoch=4)
        self.assertFalse(improved)
        self.assertEqual(stopper.counter, 3)
        self.assertTrue(stopper.early_stop)

    def test_patience_zero_disables(self):
        stopper = EarlyStopping(patience=0, min_delta=0.01, warmup_epochs=1)
        stopper.step(0.50, epoch=1)
        for ep in range(2, 20):
            stopper.step(0.40, epoch=ep)
            self.assertFalse(stopper.early_stop)


if __name__ == "__main__":
    unittest.main()
