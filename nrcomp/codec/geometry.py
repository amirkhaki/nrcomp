"""Geometry, padding arithmetic, and valid mask calculations.

Designed to be pure-Python / numpy compatible so it can run offline on dev
machines without requiring PyTorch.
"""

from typing import List, Tuple


def compute_padded_dimensions(h: int, w: int, factor: int = 64) -> Tuple[int, int]:
    """Calculate dimensions padded up to the nearest multiple of factor."""
    new_h = ((h + factor - 1) // factor) * factor
    new_w = ((w + factor - 1) // factor) * factor
    return new_h, new_w


def compute_padding_offsets(
    h: int, w: int, factor: int = 64
) -> Tuple[int, int, int, int]:
    """Calculate padding amounts: (pad_left, pad_right, pad_top, pad_bottom).

    Uses right/bottom extension so that the original image begins at (0, 0).
    """
    new_h, new_w = compute_padded_dimensions(h, w, factor=factor)
    pad_left = 0
    pad_right = new_w - w
    pad_top = 0
    pad_bottom = new_h - h
    return pad_left, pad_right, pad_top, pad_bottom


def compute_valid_mask(
    h: int, w: int, factor: int = 64, downsample: int = 16
) -> List[List[bool]]:
    """Compute 2D boolean valid_mask for latent cells.

    Each latent cell (r, c) represents a (downsample x downsample) receptive patch.
    A cell is marked True if its spatial center (y_center, x_center) lies strictly
    inside the original image dimensions [0, h) x [0, w).

    Returns:
        List of lists of booleans with shape (Hp // downsample, Wp // downsample).
    """
    new_h, new_w = compute_padded_dimensions(h, w, factor=factor)
    grid_h = new_h // downsample
    grid_w = new_w // downsample

    mask = []
    for r in range(grid_h):
        row = []
        y_center = r * downsample + (downsample // 2)
        for c in range(grid_w):
            x_center = c * downsample + (downsample // 2)
            is_valid = (y_center < h) and (x_center < w)
            row.append(is_valid)
        mask.append(row)
    return mask
