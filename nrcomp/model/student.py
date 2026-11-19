"""Latent-domain Student model predicting compression degradation from latents.

Architecture (design.md Section 7):
- Config A: y_hat only (320 channels).
- Config B: y_hat (320) and z_hat (192). z_hat is upsampled 4x with a transposed
  convolution to the y_hat grid and concatenated along channels (320 + 192 = 512).
- Trunk: ConvNeXt-style blocks (see trunk.py); no cross-position normalization.
- Blind mode: no lambda information anywhere.
- Conditioned mode: a learned lambda embedding drives FiLM inside every trunk block and is
  also concatenated into the MLP head input.
- Pooling: masked mean pooling + masked attention pooling (concatenated).
- Head: Linear -> LayerNorm -> GELU -> Dropout -> Linear -> Sigmoid (s in [0, 1]).
- Mask protection: input, stem output and every block output are multiplied by the valid
  mask; predictions for a sample are independent of extra batch padding (server-tested).
- Precision: trunk runs under autocast; normalization, pooling, head and loss run in fp32.
"""

from typing import Any, Optional

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
        lambda_embed_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_mask: bool = True,
        kernel_size: int = 7,
        expansion: int = 4,
        layer_scale_init: float = 0.1,
    ):
        super().__init__()
        self.config = config.upper()
        if self.config not in ("A", "B"):
            raise ValueError(f"config must be 'A' or 'B', got {config!r}")
        self.in_channels_y = in_channels_y
        self.in_channels_z = in_channels_z
        self.width = width
        self.num_res_blocks = num_res_blocks
        self.lambda_mode = lambda_mode.lower()
        if self.lambda_mode not in ("blind", "conditioned"):
            raise ValueError(f"lambda_mode must be 'blind' or 'conditioned', got {lambda_mode!r}")
        self.num_lambdas = num_lambdas
        self.lambda_embed_dim = lambda_embed_dim if lambda_embed_dim is not None else width
        self.dropout_rate = dropout
        self.use_mask = use_mask

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
        self.z_up = None
        self.stem = None
        self.trunk = None
        self.mean_pool = None
        self.attn_pool = None
        self.head = None

        if not HAS_TORCH:
            return

        conditioned = self.lambda_mode == "conditioned"

        # 1. Config B: transposed-conv 4x upsampling of z_hat, concatenated with y_hat.
        stem_in = in_channels_y
        if self.config == "B":
            self.z_up = nn.ConvTranspose2d(in_channels_z, in_channels_z, kernel_size=4, stride=4)
            stem_in = in_channels_y + in_channels_z

        # 2. Stem
        self.stem = ConvStem(stem_in, width)

        # 3. Lambda embedding (conditioned mode only)
        if conditioned:
            self.lambda_embed = nn.Embedding(num_lambdas, self.lambda_embed_dim)

        # 4. Trunk with FiLM in every block (conditioned mode)
        self.trunk = Trunk(
            depth=num_res_blocks,
            width=width,
            kernel_size=kernel_size,
            expansion=expansion,
            layer_scale_init=layer_scale_init,
            film_dim=self.lambda_embed_dim if conditioned else None,
        )

        # 5. Pooling
        self.mean_pool = MaskedMeanPool()
        self.attn_pool = MaskedAttentionPool(channels=width)

        # 6. MLP head (fp32): 2 * width pooled (+ lambda embedding in conditioned mode)
        head_in_dim = (width * 2) + (self.lambda_embed_dim if conditioned else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, width),
            nn.LayerNorm(width),
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
        """Set per-channel normalization statistics from the training split."""
        if not HAS_TORCH:
            return

        def _to_buffer(val, target_channels):
            t = torch.as_tensor(val, dtype=torch.float32)
            if t.numel() == target_channels:
                return t.view(1, target_channels, 1, 1)
            raise ValueError(f"Expected {target_channels} elements, got {t.numel()}")

        self.mean_y.copy_(_to_buffer(mean_y, self.in_channels_y))
        self.std_y.copy_(_to_buffer(std_y, self.in_channels_y).clamp(min=epsilon))

        if self.config == "B":
            if mean_z is None or std_z is None:
                raise ValueError("Config B requires mean_z and std_z normalization statistics.")
            self.mean_z.copy_(_to_buffer(mean_z, self.in_channels_z))
            self.std_z.copy_(_to_buffer(std_z, self.in_channels_z).clamp(min=epsilon))

    def forward(
        self,
        y_hat: "torch.Tensor",
        valid_mask: Optional["torch.Tensor"] = None,
        z_hat: Optional["torch.Tensor"] = None,
        lambda_idx: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for LatentStudent forward pass.")

        if not self.use_mask:
            valid_mask = None

        # Mask as (B, 1, H, W) float32
        m = None
        if valid_mask is not None:
            m = valid_mask.float()
            if m.ndim == 3:
                m = m.unsqueeze(1)

        # 0/1. Zero invalid input cells, then per-channel normalization (re-masked).
        y = y_hat.float()
        if m is not None:
            y = y * m
        y_norm = (y - self.mean_y) / self.std_y
        if m is not None:
            y_norm = y_norm * m

        # Config B: z_hat -> transposed-conv 4x upsample -> concat with y (512 channels)
        if self.config == "B":
            if z_hat is None:
                raise ValueError("Config B requires z_hat input tensor.")
            z = z_hat.float()
            z_mask = None
            if m is not None:
                # a z cell is valid if ANY of its y cells is valid
                z_mask = F.adaptive_max_pool2d(m, (z.size(-2), z.size(-1)))
                z = z * z_mask
            z_norm = (z - self.mean_z) / self.std_z
            if z_mask is not None:
                z_norm = z_norm * z_mask
            z_up = self.z_up(z_norm).float()
            if z_up.shape[-2:] != y_norm.shape[-2:]:
                z_up = F.interpolate(z_up, size=y_norm.shape[-2:], mode="nearest")
            inp = torch.cat([y_norm, z_up], dim=1)
            if m is not None:
                inp = inp * m
        else:
            inp = y_norm

        feat = self.stem(inp, mask=m)

        # 2. Lambda conditioning: FiLM inside every block + head concat
        e_lambda = None
        if self.lambda_mode == "conditioned":
            if lambda_idx is None:
                raise ValueError("Conditioned mode requires lambda_idx tensor.")
            e_lambda = self.lambda_embed(lambda_idx)  # (B, lambda_embed_dim)

        # 3. Trunk
        feat = self.trunk(feat, mask=m, cond=e_lambda)

        # 4. Pooling in fp32
        mean_p = self.mean_pool(feat, mask=m).float()
        attn_p = self.attn_pool(feat, mask=m).float()
        pooled = torch.cat([mean_p, attn_p], dim=1)

        # 5. Head input
        head_in = torch.cat([pooled, e_lambda.float()], dim=1) if e_lambda is not None else pooled

        # 6. MLP head explicitly in fp32 outside autocast
        try:
            amp_ctx = torch.amp.autocast(device_type=head_in.device.type, enabled=False)
        except (AttributeError, TypeError):
            amp_ctx = torch.cuda.amp.autocast(enabled=False)

        with amp_ctx:
            pred = self.head(head_in.float()).squeeze(-1)  # (B,)

        return pred
