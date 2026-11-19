"""Trunk architecture: ConvStem, ConvNeXt-style ResBlock with FiLM, and Trunk.

Per design.md Section 7:
- "Lightweight ConvNeXt-style convolutional blocks operating directly on latent grids."
- "Lambda conditioning ... injected via FiLM ... into intermediate residual blocks."

Mask invariance (the reason GroupNorm was removed):
  Every normalization here is PER POSITION over channels (LayerNorm over C). No statistic
  is ever computed across spatial positions or across batch samples, so the output at a
  valid cell cannot depend on how much extra zero-padding the batch contains. Features at
  invalid cells are multiplied by the mask after the stem and after every block, so the
  7x7 depthwise convolution always sees zeros outside the valid region.
"""

from typing import Optional

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

        def register_buffer(self, name, val):
            setattr(self, name, val)


class ChannelLayerNorm(BaseModule):
    """LayerNorm over the channel dimension of (B, C, H, W), computed in float32."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.eps = eps
        if HAS_TORCH:
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for ChannelLayerNorm forward pass.")
        x32 = x.float().permute(0, 2, 3, 1)
        x32 = F.layer_norm(x32, (self.channels,), self.weight.float(), self.bias.float(), self.eps)
        return x32.permute(0, 3, 1, 2)


class ConvStem(BaseModule):
    """1x1 Conv stem mapping input channels to trunk width, followed by per-position LayerNorm."""

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if HAS_TORCH:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.norm = ChannelLayerNorm(out_channels)
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
    """ConvNeXt-style block: dw7x7 -> LayerNorm -> [FiLM] -> 1x1 (x4) -> GELU -> 1x1 -> layer scale.

    `film_dim` > 0 enables FiLM conditioning: scale/shift are predicted from a conditioning
    vector (the lambda embedding) and applied right after the LayerNorm.
    """

    def __init__(
        self,
        channels: int = 256,
        kernel_size: int = 7,
        expansion: int = 4,
        layer_scale_init: float = 0.1,
        film_dim: Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels
        self.film_dim = film_dim
        if HAS_TORCH:
            self.dwconv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
            self.norm = ChannelLayerNorm(channels)
            self.pw1 = nn.Conv2d(channels, channels * expansion, kernel_size=1)
            self.act = nn.GELU()
            self.pw2 = nn.Conv2d(channels * expansion, channels, kernel_size=1)
            self.gamma = nn.Parameter(layer_scale_init * torch.ones(channels))
            if film_dim:
                self.film = nn.Linear(film_dim, 2 * channels)
                nn.init.normal_(self.film.weight, std=0.02)
                nn.init.zeros_(self.film.bias)
            else:
                self.film = None
        else:
            self.dwconv = None
            self.norm = None
            self.pw1 = None
            self.act = None
            self.pw2 = None
            self.film = None

    def forward(
        self,
        x: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
        cond: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for ResBlock forward pass.")
        h = self.norm(self.dwconv(x))
        if self.film is not None:
            if cond is None:
                raise ValueError("FiLM-enabled ResBlock requires a conditioning vector.")
            scale, shift = self.film(cond.float()).chunk(2, dim=1)
            h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.pw2(self.act(self.pw1(h)))
        out = x + h.float() * self.gamma.view(1, -1, 1, 1)
        if mask is not None:
            out = out * mask
        return out


# Descriptive alias: the block is a ConvNeXt block.
ConvNeXtBlock = ResBlock


class Trunk(BaseModule):
    """Stack of ConvNeXt-style blocks at uniform channel width."""

    def __init__(
        self,
        depth: int = 6,
        width: int = 256,
        kernel_size: int = 7,
        expansion: int = 4,
        layer_scale_init: float = 0.1,
        film_dim: Optional[int] = None,
    ):
        super().__init__()
        self.depth = depth
        self.width = width
        self.film_dim = film_dim
        if HAS_TORCH:
            self.blocks = nn.ModuleList(
                [
                    ResBlock(
                        channels=width,
                        kernel_size=kernel_size,
                        expansion=expansion,
                        layer_scale_init=layer_scale_init,
                        film_dim=film_dim,
                    )
                    for _ in range(depth)
                ]
            )
        else:
            self.blocks = []

    def forward(
        self,
        x: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
        cond: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for Trunk forward pass.")
        for block in self.blocks:
            x = block(x, mask=mask, cond=cond)
        return x
