"""Guard test ensuring no report generators contain hardcoded/fabricated metric numbers.

Implements Ground Rule 1: No fabricated numbers. All numbers must be loaded
from real run artifacts. This test scans report generation code for hardcoded
markdown metric rows or literal result dictionaries.
"""

import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestNoHardcodedReports(unittest.TestCase):
    def test_no_hardcoded_tables_in_nrcomp(self):
        # Scan nrcomp Python source files for hardcoded markdown table rows with numbers
        # Pattern: markdown table row with multiple decimal numbers, e.g. "| 0.8069 | 0.7247 |"
        suspicious_row_pattern = re.compile(r"\|\s*0\.\d{3,4}\s*\|\s*0\.\d{3,4}\s*\|")

        files_to_check = [
            PROJECT_ROOT / "nrcomp" / "eval_pixel_refs.py",
            PROJECT_ROOT / "nrcomp" / "evaluate.py",
        ]

        violations = []
        for fpath in files_to_check:
            if not fpath.exists():
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, 1):
                    # Skip comments
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    if suspicious_row_pattern.search(line):
                        violations.append(f"{fpath.name}:{line_idx}: {line.strip()}")

        self.assertEqual(
            len(violations),
            0,
            f"Found hardcoded metric table row(s) in report code:\n" + "\n".join(violations),
        )

    def test_no_hardcoded_result_dicts_in_eval_pixel_refs(self):
        fpath = PROJECT_ROOT / "nrcomp" / "eval_pixel_refs.py"
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        # Ensure no mock hardcoded results like 'overall_srocc': 0.4820 in eval_pixel_refs.py
        mock_result_pattern = re.compile(r"['\"]overall_srocc['\"]\s*:\s*0\.\d+")
        matches = mock_result_pattern.findall(content)
        self.assertEqual(
            len(matches),
            0,
            f"Found hardcoded result dictionary entries in eval_pixel_refs.py: {matches}",
        )


if __name__ == "__main__":
    unittest.main()
