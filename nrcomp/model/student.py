"""Latent-domain Student model predicting compression degradation from latents.

Per Milestone M4 specification:
- Config A: y_hat only (320 channels).
- Config B: y_hat (320) + 4x nearest-neighbor upsampled z_hat (192) fused via 1x1 conv.
- Blind mode: No lambda conditioning, Head input width 512.
- Conditioned mode: Learned lambda embedding added before trunk AND concatenated
  into the MLP head (Head input width 768).
- Mask protection: Features multiplied by valid_mask after stem and after every ResBlock.
- Precision: Trunk runs in autocast (bf16), while pooling, head, and loss math
  are explicitly enforced in float32 to prevent output quantization blurring.
"""

from typing import Any, Dict, Optional, Tuple, Union

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


from nrcomp.model.pooling import MaskedAttentionPool, MaskedMeanPool
from nrcomp.model.trunk import ConvStem, Trunk


class LatentStudent(BaseModule):
    """No-Reference compression degradation predictor operating on codec latents."""

    def __init__(
        self,
        config: str = "A",
        in_channels_y: int = 320,
        in_channels_z: int = 192,
        width: int = 256,
        num_res_blocks: int = 6,
        lambda_mode: str = "blind",
        num_lambdas: int = 6,
        lambda_embed_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = config.upper()
        self.in_channels_y = in_channels_y
        self.in_channels_z = in_channels_z
        self.width = width
        self.num_res_blocks = num_res_blocks
        self.lambda_mode = lambda_mode.lower()
        self.num_lambdas = num_lambdas
        self.lambda_embed_dim = lambda_embed_dim
        self.dropout_rate = dropout

        # Normalization buffers
        if HAS_TORCH:
            self.register_buffer("mean_y", torch.zeros(1, in_channels_y, 1, 1))
            self.register_buffer("std_y", torch.ones(1, in_channels_y, 1, 1))
            if self.config == "B":
                self.register_buffer("mean_z", torch.zeros(1, in_channels_z, 1, 1))
                self.register_buffer("std_z", torch.ones(1, in_channels_z, 1, 1))
        else:
            self.mean_y = None
            self.std_y = None
            self.mean_z = None
            self.std_z = None

        self.lambda_embed = None
        self.stem_y = None
        self.stem_z = None
        self.fuse_b = None
        self.trunk = None
        self.mean_pool = None
        self.attn_pool = None
        self.head = None

        if not HAS_TORCH:
            return

        # 1. Stem(s)
        self.stem_y = ConvStem(in_channels_y, width)
        if self.config == "B":
            self.stem_z = ConvStem(in_channels_z, width)
            self.fuse_b = nn.Conv2d(width * 2, width, kernel_size=1, bias=False)

        # 2. Lambda Embedding (Conditioned mode only)
        if self.lambda_mode == "conditioned":
            self.lambda_embed = nn.Embedding(num_lambdas, lambda_embed_dim)
        else:
            self.lambda_embed = None

        # 3. Residual Trunk with mask protection
        self.trunk = Trunk(depth=num_res_blocks, width=width)

        # 4. Pooling
        self.mean_pool = MaskedMeanPool()
        self.attn_pool = MaskedAttentionPool(channels=width)

        # 5. MLP Head (enforced in fp32)
        # In conditioned mode: 512 (pooled) + 256 (lambda_embed) = 768
        # In blind mode: 512 (pooled)
        head_in_dim = (width * 2) + (lambda_embed_dim if self.lambda_mode == "conditioned" else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, width),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(width, 1),
            nn.Sigmoid(),
        )

    def set_normalization(
        self,
        mean_y: Any,
        std_y: Any,
        mean_z: Optional[Any] = None,
        std_z: Optional[Any] = None,
        epsilon: float = 1e-5,
    ):
        """Set per-channel normalization statistics from training split."""
        if not HAS_TORCH:
            return

        def _to_buffer(val, target_channels):
            t = torch.as_tensor(val, dtype=torch.float32)
            if t.numel() == target_channels:
                return t.view(1, target_channels, 1, 1)
            raise ValueError(f"Expected {target_channels} elements, got {t.numel()}")

        mean_y_t = _to_buffer(mean_y, self.in_channels_y)
        std_y_t = _to_buffer(std_y, self.in_channels_y).clamp(min=epsilon)

        self.mean_y.copy_(mean_y_t)
        self.std_y.copy_(std_y_t)

        if self.config == "B" and mean_z is not None and std_z is not None:
            mean_z_t = _to_buffer(mean_z, self.in_channels_z)
            std_z_t = _to_buffer(std_z, self.in_channels_z).clamp(min=epsilon)
            self.mean_z.copy_(mean_z_t)
            self.std_z.copy_(std_z_t)

    def forward(
        self,
        y_hat: "torch.Tensor",
        valid_mask: Optional["torch.Tensor"] = None,
        z_hat: Optional["torch.Tensor"] = None,
        lambda_idx: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for LatentStudent forward pass.")

        # Ensure mask is formatted as (B, 1, H, W) float
        m_f32 = None
        if valid_mask is not None:
            m_f32 = valid_mask.float()
            if m_f32.ndim == 3:
                m_f32 = m_f32.unsqueeze(1)

        # 1. Input Normalization
        y_norm = (y_hat.float() - self.mean_y) / self.std_y
        feat = self.stem_y(y_norm, mask=m_f32)

        # Config B Fusion
        if self.config == "B":
            if z_hat is None:
                raise ValueError("Config B requires z_hat input tensor.")
            z_norm = (z_hat.float() - self.mean_z) / self.std_z
            # Stem Z (downsampled 4x relative to y_hat)
            feat_z = self.stem_z(z_norm)
            # Upsample 4x (nearest) to match y_hat grid
            feat_z_up = F.interpolate(
                feat_z, size=(feat.size(-2), feat.size(-1)), mode="nearest"
            )
            # Concat and 1x1 fuse to width
            feat = self.fuse_b(torch.cat([feat, feat_z_up], dim=1))
            if m_f32 is not None:
                feat = feat * m_f32

        # 2. Lambda conditioning (Trunk input)
        e_lambda = None
        if self.lambda_mode == "conditioned":
            if lambda_idx is None:
                raise ValueError("Conditioned mode requires lambda_idx tensor.")
            e_lambda = self.lambda_embed(lambda_idx)  # (B, 256)
            # Broadcast add into trunk feature map
            feat = feat + e_lambda.unsqueeze(-1).unsqueeze(-1)
            if m_f32 is not None:
                feat = feat * m_f32

        # 3. Trunk feature extraction with inter-layer mask protection
        feat = self.trunk(feat, mask=m_f32)

        # 4. Pooling in fp32
        mean_p = self.mean_pool(feat, mask=m_f32).float()  # (B, width)
        attn_p = self.attn_pool(feat, mask=m_f32).float()  # (B, width)
        pooled = torch.cat([mean_p, attn_p], dim=1)        # (B, 2 * width)

        # 5. Head input
        if self.lambda_mode == "conditioned" and e_lambda is not None:
            head_in = torch.cat([pooled, e_lambda.float()], dim=1)  # (B, 768)
        else:
            head_in = pooled  # (B, 512)

        # 6. MLP Head explicitly in fp32 outside autocast
        with torch.cuda.amp.autocast(enabled=False):
            pred = self.head(head_in.float()).squeeze(-1)  # (B,)

        return pred
