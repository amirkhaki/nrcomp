"""Unit tests verifying single source of truth for ELIC lambdas (A5)."""

import re
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp.lambdas import (
    get_lambda_checkpoints,
    get_lambda_names,
    get_lambda_values,
    load_lambdas_config,
)


class TestLambdasSingleSource(unittest.TestCase):
    def test_load_lambdas_config(self):
        lambdas = load_lambdas_config()
        self.assertEqual(len(lambdas), 6)
        self.assertEqual([item["idx"] for item in lambdas], list(range(6)))

        # Verify values in order
        expected_values = [0.004, 0.008, 0.016, 0.032, 0.15, 0.45]
        self.assertEqual(get_lambda_values(), expected_values)
        self.assertEqual(get_lambda_values(lambdas), expected_values)
        self.assertEqual(get_lambda_names(lambdas), [item["name"] for item in lambdas])
        self.assertEqual(get_lambda_checkpoints(lambdas), [item["checkpoint"] for item in lambdas])

        # Specifically verify 4th lambda value is 0.016 (not 0.0016)
        self.assertEqual(lambdas[2]["name"], "0.016")
        self.assertEqual(lambdas[2]["value"], 0.016)

        # Verify checkpoints
        ckpts = get_lambda_checkpoints()
        for ckpt in ckpts:
            self.assertTrue(ckpt.startswith("elic_lambda_") and ckpt.endswith(".pth.tar"))

    def test_no_hardcoded_cheng_lambdas(self):
        """Verify that stale Cheng2020 anchor lambdas (0.0016, 0.0036, 0.0075) do not appear in active code."""
        stale_pattern = re.compile(r"0\.0016|0\.0036|0\.0075")
        active_dirs = [PROJECT_ROOT / "nrcomp", PROJECT_ROOT / "scripts", PROJECT_ROOT / "configs"]

        violations = []
        for d in active_dirs:
            for p in d.rglob("*.py"):
                if "archive" in p.parts:
                    continue
                # scripts/download_weights.py has historical comment on Google Drive link text
                if p.name == "download_weights.py":
                    continue
                content = p.read_text(encoding="utf-8")
                # Filter out comments mentioning 0.0016 explanation
                lines = content.splitlines()
                for line_idx, line in enumerate(lines, 1):
                    if stale_pattern.search(line) and not line.strip().startswith("#"):
                        violations.append(f"{p.relative_to(PROJECT_ROOT)}:{line_idx}: {line.strip()}")

        self.assertEqual(violations, [], f"Found hardcoded stale lambda references: {violations}")


if __name__ == "__main__":
    unittest.main()
