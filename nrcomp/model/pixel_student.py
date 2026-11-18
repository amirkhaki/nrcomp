"""Pixel-Domain NR Reference Student Model (E6a).

Implements Section 10 (E6a): A pixel-domain NR model trained on decoded reconstructions
with the same labels and splits, providing the benchmark for what latent-domain prediction
gains or gives up in accuracy vs decode+inference latency.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    BaseModule = nn.Module
except ImportError:
    HAS_TORCH = False
    BaseModule = object


class ConvBlock(BaseModule):
    """Convolution + GroupNorm + GELU block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        if HAS_TORCH:
            num_groups = min(32, out_channels)
            while out_channels % num_groups != 0:
                num_groups //= 2
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.norm = nn.GroupNorm(num_groups, out_channels)
            self.act = nn.GELU()

    def forward(self, x: Any) -> Any:
        return self.act(self.norm(self.conv(x)))


class ResStage(BaseModule):
    """Downsample stage with residual connection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if HAS_TORCH:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False) if in_channels != out_channels else nn.Identity()
            self.conv1 = ConvBlock(in_channels, out_channels, stride=2)
            self.conv2 = ConvBlock(out_channels, out_channels, stride=1)

    def forward(self, x: Any) -> Any:
        res = self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return out + res


class PixelStudent(BaseModule):
    """Lightweight pixel-domain convolutional student predicting quality s in (0, 1) from RGB images.

    Downsamples 1024x768 (or 500x500) images through 4 residual stages to feature vectors,
    pooled via adaptive average pooling, and mapped through an MLP head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        lambda_mode: str = "conditioned",
        num_lambdas: int = 6,
        lambda_embed_dim: int = 64,
        dropout: float = 0.1,
        backbone: str = "custom",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.lambda_mode = lambda_mode.lower()
        self.num_lambdas = num_lambdas
        self.lambda_embed_dim = lambda_embed_dim
        self.dropout_rate = dropout
        self.backbone_name = backbone

        if not HAS_TORCH:
            return

        c = base_channels
        # Stem: 3 -> 64 (stride 2 -> H/2, W/2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(min(32, c), c),
            nn.GELU(),
        )

        # 3 Downsample Stages:
        # Stage 1: 64 -> 128 (stride 2 -> H/4, W/4)
        self.stage1 = ResStage(c, c * 2)
        # Stage 2: 128 -> 256 (stride 2 -> H/8, W/8)
        self.stage2 = ResStage(c * 2, c * 4)
        # Stage 3: 256 -> 256 (stride 2 -> H/16, W/16)
        self.stage3 = ResStage(c * 4, c * 4)

        # Global Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Lambda Conditioning
        if self.lambda_mode == "conditioned":
            self.lambda_embed = nn.Embedding(num_lambdas, lambda_embed_dim)
            head_in = (c * 4) + lambda_embed_dim
        else:
            self.lambda_embed = None
            head_in = c * 4

        # Head
        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Any, lambda_idx: Optional[Any] = None) -> Any:
        """Forward pass taking (B, 3, H, W) images in [0, 1]."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for PixelStudent forward pass.")

        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        pooled = self.pool(feat).flatten(1)  # (B, 256)

        if self.lambda_mode == "conditioned":
            if lambda_idx is None:
                raise ValueError("lambda_idx must be provided in conditioned mode.")
            e_lam = self.lambda_embed(lambda_idx)
            pooled = torch.cat([pooled, e_lam], dim=1)

        # Run head in fp32
        out = self.head(pooled.float())
        return out
