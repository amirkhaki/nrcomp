"""Unit tests for ELICWrapper contract and mock execution."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nrcomp.codec.wrapper import ELICWrapper


class TestELICWrapper(unittest.TestCase):
    def test_init_sets_properties(self):
        wrapper = ELICWrapper(weights_dir="weights/updated", device="cpu", entropy_coder="ans")
        self.assertEqual(wrapper.device, "cpu")
        self.assertEqual(wrapper.entropy_coder, "ans")
        self.assertEqual(wrapper.weights_dir, Path("weights/updated"))

    def test_get_model_not_loaded_raises(self):
        wrapper = ELICWrapper(weights_dir="weights/updated", device="cpu")
        with self.assertRaises(KeyError):
            wrapper.get_model(0)


if __name__ == "__main__":
    unittest.main()
