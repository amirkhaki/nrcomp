"""Masked spatial pooling operations for latent feature grids.

Implements:
- MaskedMeanPool: Average over valid (unpadded) cells in fp32.
- MaskedAttentionPool: Learned query spatial attention over valid cells in fp32.
Also provides pure-Python reference oracles for offline testing.
"""

import math
from typing import List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
    BaseModule = nn.Module
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False

    class BaseModule:  # type: ignore
        def __init__(self, *args, **kwargs):
            self._modules = {}
            self._parameters = {}
            self._buffers = {}

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)


class MaskedMeanPool(BaseModule):
    """Averages feature maps over valid (unpadded) cells in float32.

    pool(F) = sum_{h, w} (F_{c, h, w} * M_{h, w}) / max(sum_{h, w} M_{h, w}, 1)
    """

    def forward(
        self, x: "torch.Tensor", mask: Optional["torch.Tensor"] = None
    ) -> "torch.Tensor":
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for MaskedMeanPool.")

        # Always execute pooling math in fp32 to prevent numerical precision loss
        orig_dtype = x.dtype
        x_f32 = x.float()

        if mask is None:
            return x_f32.mean(dim=(-2, -1)).to(orig_dtype)

        m_f32 = mask.float()
        if m_f32.ndim == 3:
            m_f32 = m_f32.unsqueeze(1)  # (B, 1, H, W)

        numerator = (x_f32 * m_f32).sum(dim=(-2, -1))  # (B, C)
        denominator = m_f32.sum(dim=(-2, -1)).clamp(min=1.0)  # (B, 1)
        pooled = numerator / denominator
        return pooled.to(orig_dtype)


class MaskedAttentionPool(BaseModule):
    """Learned query spatial attention over valid cells in float32.

    Uses a 1x1 conv to compute unnormalized spatial attention logits,
    sets invalid cells to -1e9 before softmax, and computes the weighted sum.
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.channels = channels

        if HAS_TORCH:
            self.attn_conv = nn.Conv2d(channels, 1, kernel_size=1)
        else:
            self.attn_conv = None

    def forward(
        self, x: "torch.Tensor", mask: Optional["torch.Tensor"] = None
    ) -> "torch.Tensor":
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for MaskedAttentionPool.")

        orig_dtype = x.dtype
        x_f32 = x.float()
        b, c, h, w = x_f32.shape

        logits = self.attn_conv(x_f32)  # (B, 1, H, W)

        if mask is not None:
            m_bool = mask.bool()
            if m_bool.ndim == 3:
                m_bool = m_bool.unsqueeze(1)
            # Mask out invalid cells with a large negative number
            logits = logits.masked_fill(~m_bool, -1e9)

        # Softmax over spatial dimensions (H * W)
        attn_flat = logits.view(b, 1, -1)
        weights_flat = F.softmax(attn_flat, dim=-1)
        weights = weights_flat.view(b, 1, h, w)

        # If an entire sample has no valid cells, guard against NaN
        if mask is not None:
            m_f32 = mask.float()
            if m_f32.ndim == 3:
                m_f32 = m_f32.unsqueeze(1)
            weights = weights * m_f32
            w_sum = weights.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
            weights = weights / w_sum

        pooled = (x_f32 * weights).sum(dim=(-2, -1))  # (B, C)
        return pooled.to(orig_dtype)


# ---------------------------------------------------------------------------
# Pure-Python Reference Oracles (for offline testing & verification)
# ---------------------------------------------------------------------------

def masked_mean_pool_oracle(
    features: List[List[List[float]]], mask: Optional[List[List[bool]]] = None
) -> List[float]:
    """Pure-python reference oracle for MaskedMeanPool.

    Args:
        features: shape (C, H, W)
        mask: shape (H, W) of bools
    Returns:
        List of C pooled floats
    """
    c = len(features)
    h = len(features[0])
    w = len(features[0][0])

    if mask is None:
        mask = [[True] * w for _ in range(h)]

    valid_count = sum(1 for r in range(h) for col in range(w) if mask[r][col])
    den = max(valid_count, 1)

    result = []
    for ch in range(c):
        ch_sum = sum(
            features[ch][r][col]
            for r in range(h)
            for col in range(w)
            if mask[r][col]
        )
        result.append(ch_sum / den)
    return result


def masked_attention_pool_oracle(
    features: List[List[List[float]]],
    weights_w: List[float],
    bias_b: float,
    mask: Optional[List[List[bool]]] = None,
) -> List[float]:
    """Pure-python reference oracle for MaskedAttentionPool."""
    c = len(features)
    h = len(features[0])
    w = len(features[0][0])

    if mask is None:
        mask = [[True] * w for _ in range(h)]

    # Compute unnormalized logits: dot product with weights_w + bias_b
    logits = []
    for r in range(h):
        for col in range(w):
            if not mask[r][col]:
                logits.append(-1e9)
            else:
                val = sum(features[ch][r][col] * weights_w[ch] for ch in range(c)) + bias_b
                logits.append(val)

    # Safe softmax
    max_logit = max(logits)
    exp_logits = [math.exp(v - max_logit) if v > -1e8 else 0.0 for v in logits]
    total_exp = sum(exp_logits)
    if total_exp <= 1e-12:
        weights = [1.0 / len(logits)] * len(logits)
    else:
        weights = [v / total_exp for v in exp_logits]

    # Weighted sum
    result = [0.0] * c
    idx = 0
    for r in range(h):
        for col in range(w):
            weight = weights[idx]
            idx += 1
            for ch in range(c):
                result[ch] += features[ch][r][col] * weight
    return result
