"""Dataset split utilities.

Generates deterministic 80/10/10 train/val/test splits for KonIQ-10k
stratified by MOS quintile with seed 0, and assigns LiveC to test-only.
"""

import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple


def create_koniq_stratified_split(
    items: List[Tuple[str, float]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
    num_bins: int = 5,
) -> Dict[str, str]:
    """Create a stratified split of (image_id, mos) items into train/val/test.

    Args:
        items: List of (image_id, mos) tuples.
        train_ratio: Proportion for training (default 0.8).
        val_ratio: Proportion for validation (default 0.1).
        test_ratio: Proportion for testing (default 0.1).
        seed: Random seed for reproducibility (default 0).
        num_bins: Number of MOS quintiles for stratification (default 5).

    Returns:
        Dict mapping image_id to split name ('train', 'val', 'test').
    """
    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1.0"
    if not items:
        return {}

    # Sort items by MOS
    sorted_items = sorted(items, key=lambda x: x[1])
    n = len(sorted_items)
    bin_size = n / num_bins

    rng = random.Random(seed)
    split_map: Dict[str, str] = {}

    for b in range(num_bins):
        start_idx = int(b * bin_size)
        end_idx = int((b + 1) * bin_size) if b < num_bins - 1 else n
        bin_items = [x[0] for x in sorted_items[start_idx:end_idx]]

        # Deterministic shuffle within bin
        rng.shuffle(bin_items)
        bin_n = len(bin_items)

        n_train = int(bin_n * train_ratio)
        n_val = int(bin_n * val_ratio)

        for img_id in bin_items[:n_train]:
            split_map[img_id] = "train"
        for img_id in bin_items[n_train : n_train + n_val]:
            split_map[img_id] = "val"
        for img_id in bin_items[n_train + n_val :]:
            split_map[img_id] = "test"

    return split_map


def save_koniq_split_csv(
    split_map: Dict[str, str], mos_map: Dict[str, float], output_path: Path
):
    """Save split mapping to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "split", "orig_mos"])
        for img_id, split in sorted(split_map.items()):
            writer.writerow([img_id, split, mos_map.get(img_id, 0.0)])


def load_koniq_split_csv(input_path: Path) -> Dict[str, str]:
    """Load precomputed split mapping from CSV."""
    split_map = {}
    with open(input_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_map[row["image_id"]] = row["split"]
    return split_map
