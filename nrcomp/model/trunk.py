"""Trunk architecture: ConvStem, ResBlock with mask protection, and Trunk.

Per design doc Section 9.1 and Milestone M4 specification:
- ConvStem: 1x1 Conv2d to width (256), followed by GroupNorm(32, 256).
- ResBlock: Conv2d(3x3) -> GELU -> Conv2d(3x3) with residual connection.
  Multiplies features by valid_mask to prevent padding leakage.
- Trunk: 6 sequential ResBlocks at width 256.
"""

from typing import Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    BaseModule = nn.Module
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False

    class BaseModule:  # type: ignore
        def __init__(self, *args, **kwargs):
            self._modules = {}
            self._parameters = {}
            self._buffers = {}

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def register_buffer(self, name, val):
            setattr(self, name, val)


class ConvStem(BaseModule):
    """1x1 Conv stem mapping input channels to trunk width with GroupNorm."""

    def __init__(self, in_channels: int, out_channels: int = 256, num_groups: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = min(num_groups, out_channels)

        if HAS_TORCH:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.norm = nn.GroupNorm(num_groups=self.num_groups, num_channels=out_channels)
        else:
            self.conv = None
            self.norm = None

    def forward(self, x: "torch.Tensor", mask: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for ConvStem forward pass.")
        out = self.norm(self.conv(x))
        if mask is not None:
            out = out * mask
        return out


class ResBlock(BaseModule):
    """Residual block with mask protection to prevent padding boundary leakage."""

    def __init__(self, channels: int = 256):
        super().__init__()
        self.channels = channels

        if HAS_TORCH:
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.act = nn.GELU()
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        else:
            self.conv1 = None
            self.act = None
            self.conv2 = None

    def forward(self, x: "torch.Tensor", mask: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for ResBlock forward pass.")
        res = self.conv2(self.act(self.conv1(x)))
        out = x + res
        if mask is not None:
            out = out * mask
        return out


class Trunk(BaseModule):
    """Trunk consisting of a stack of ResBlocks at uniform channel width."""

    def __init__(self, depth: int = 6, width: int = 256):
        super().__init__()
        self.depth = depth
        self.width = width

        if HAS_TORCH:
            self.blocks = nn.ModuleList([ResBlock(channels=width) for _ in range(depth)])
        else:
            self.blocks = []

    def forward(self, x: "torch.Tensor", mask: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for Trunk forward pass.")
        for block in self.blocks:
            x = block(x, mask=mask)
        return x
