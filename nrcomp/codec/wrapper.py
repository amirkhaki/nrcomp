"""ELIC codec wrapper for compression, decompression, and latent extraction.

Implements the contract defined in Section 6.3 of the design document:
- codec.compress(img_uint8_hwc, lambda_idx) -> (bytes, meta)
- codec.decompress(bytes, meta) -> img_uint8_hwc
- codec.extract_latents(img_uint8_hwc, lambda_idx) -> dict(y_hat, z_hat, valid_mask, meta)
- codec.decode_from_latents(y_hat, meta) -> img_uint8_hwc
"""

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from nrcomp.codec.geometry import (
    compute_padded_dimensions,
    compute_padding_offsets,
    compute_valid_mask,
)


class ELICWrapper:
    """Wrapper for VincentChandelier/ELiC-ReImplemetation models across all lambdas."""

    def __init__(
        self,
        weights_dir: Union[str, Path],
        device: str = "cuda",
        entropy_coder: str = "ans",
    ):
        self.weights_dir = Path(weights_dir)
        self.device = device
        self.entropy_coder = entropy_coder
        self.models: Dict[int, Any] = {}
        self.lambda_values: Dict[int, float] = {}

        # Add submodule to sys.path if not present
        submodule_path = (
            Path(__file__).resolve().parent.parent.parent
            / "third_party"
            / "ELiC-ReImplemetation"
        )
        if str(submodule_path) not in sys.path:
            sys.path.insert(0, str(submodule_path))

        # Enforce deterministic floating-point precision on CUDA
        # to prevent rANS entropy coder desynchronization
        try:
            import torch
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = False
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = False
        except ImportError:
            pass

    def load_model(self, lambda_idx: int, checkpoint_path: Union[str, Path]):
        """Load and cache an updated ELIC model checkpoint for lambda_idx."""
        import torch
        from compressai.zoo import load_state_dict
        import compressai
        from Network import TestModel

        compressai.set_entropy_coder(self.entropy_coder)
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_absolute():
            ckpt_path = self.weights_dir / ckpt_path

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        try:
            raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            raw_ckpt = torch.load(ckpt_path, map_location="cpu")

        state_dict = load_state_dict(raw_ckpt)
        model_cls = TestModel()
        net = model_cls.from_state_dict(state_dict).eval().to(self.device)

        # Patch net.decompress to use robust latent extraction logic
        wrapper_extract = self._extract_latents_from_strings
        def robust_decompress(strings, shape):
            import time
            y_hat, _ = wrapper_extract(net, strings, shape)
            t0 = time.time()
            x_hat = net.g_s(y_hat).clamp(0.0, 1.0)
            return {"x_hat": x_hat, "time": {"y_dec": time.time() - t0}}
        net.decompress = robust_decompress

        self.models[lambda_idx] = net

    def get_model(self, lambda_idx: int) -> Any:
        if lambda_idx not in self.models:
            raise KeyError(
                f"Model for lambda_idx {lambda_idx} not loaded. Call load_model first."
            )
        return self.models[lambda_idx]

    def _prepare_input_tensor(
        self, img_uint8_hwc: Any
    ) -> Tuple[Any, Tuple[int, int], Tuple[int, int, int, int]]:
        """Convert uint8 HWC image to padded float32 NCHW tensor on device."""
        import numpy as np
        import torch

        img_np = np.asarray(img_uint8_hwc, dtype=np.float32) / 255.0
        h, w = img_np.shape[0], img_np.shape[1]

        # Pad to multiple of 64 using reflect padding
        pad_l, pad_r, pad_t, pad_b = compute_padding_offsets(h, w, factor=64)

        # HWC -> CHW -> NCHW
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

        if pad_r > 0 or pad_b > 0:
            # torch.nn.functional.pad order: (left, right, top, bottom)
            # Reflect padding requires input > padding; fallback to replicate if image is small
            pad_mode = "reflect" if (w > pad_r and h > pad_b) else "replicate"
            tensor = torch.nn.functional.pad(
                tensor, (pad_l, pad_r, pad_t, pad_b), mode=pad_mode
            )

        return tensor, (h, w), (pad_l, pad_r, pad_t, pad_b)

    def compress(
        self, img_uint8_hwc: Any, lambda_idx: int
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Compress image to bitstream bytes and metadata."""
        import torch

        net = self.get_model(lambda_idx)
        x_padded, (orig_h, orig_w), (pad_l, pad_r, pad_t, pad_b) = (
            self._prepare_input_tensor(img_uint8_hwc)
        )
        padded_h, padded_w = x_padded.shape[-2], x_padded.shape[-1]

        with torch.no_grad():
            out_enc = net.compress(x_padded)

        meta = {
            "original_size": (orig_h, orig_w),
            "padded_size": (padded_h, padded_w),
            "padding": (pad_l, pad_r, pad_t, pad_b),
            "lambda_idx": lambda_idx,
            "shape": out_enc["shape"],
        }

        bitstream_payload = pickle.dumps(
            {"strings": out_enc["strings"], "shape": out_enc["shape"]}
        )
        return bitstream_payload, meta

    def decompress(self, bitstream_bytes: bytes, meta: Dict[str, Any]) -> Any:
        """Decompress bitstream bytes back to original-resolution uint8 RGB image."""
        import pickle
        import torch

        payload = pickle.loads(bitstream_bytes)
        strings = payload["strings"]
        shape = payload["shape"]
        lambda_idx = meta["lambda_idx"]

        net = self.get_model(lambda_idx)
        with torch.no_grad():
            y_hat, _ = self._extract_latents_from_strings(net, strings, shape)
        return self.decode_from_latents(y_hat, meta)

    def extract_latents_from_bitstream(
        self, bitstream_bytes: bytes, meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract quantized latents y_hat and z_hat directly from bitstream payload."""
        import pickle
        import torch

        payload = pickle.loads(bitstream_bytes)
        strings = payload["strings"]
        shape = payload["shape"]
        lambda_idx = meta["lambda_idx"]

        net = self.get_model(lambda_idx)
        with torch.no_grad():
            y_hat, z_hat = self._extract_latents_from_strings(net, strings, shape)

        orig_h, orig_w = meta["original_size"]
        valid_mask = compute_valid_mask(orig_h, orig_w, factor=64, downsample=16)

        return {
            "y_hat": y_hat.squeeze(0),
            "z_hat": z_hat.squeeze(0),
            "valid_mask": torch.tensor(valid_mask, dtype=torch.bool, device=self.device),
            "meta": meta,
        }

    def _extract_latents_from_strings(
        self, net: Any, strings: list, shape: tuple
    ) -> Tuple[Any, Any]:
        """Internal helper: entropy-decode strings to obtain y_hat and z_hat."""
        import torch

        z_strings = strings[1]
        z_hat = net.entropy_bottleneck.decompress(z_strings, shape)
        B, _, _, _ = z_hat.size()

        latent_means, latent_scales = net.h_s(z_hat).chunk(2, 1)
        y_strings = strings[0]

        ctx_params_anchor = torch.zeros(
            (B, net.M * 2, z_hat.shape[2] * 4, z_hat.shape[3] * 4)
        ).to(z_hat.device)
        ctx_params_anchor_split = torch.split(
            ctx_params_anchor, [2 * i for i in net.groups[1:]], 1
        )

        y_hat_slices = []
        for slice_index in range(len(net.groups) - 1):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = net.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(
                    2, 1
                )
            else:
                support_slices = torch.concat(
                    [y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1
                )
                support_slices_ch = net.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(
                    2, 1
                )

            support = (
                torch.concat([latent_means, latent_scales], dim=1)
                if slice_index == 0
                else torch.concat(
                    [
                        support_slices_ch_mean,
                        support_slices_ch_scale,
                        latent_means,
                        latent_scales,
                    ],
                    dim=1,
                )
            )

            means_anchor, scales_anchor = net.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)
            ).chunk(2, 1)

            B_anchor, C_anchor, H_anchor, W_anchor = means_anchor.size()
            means_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(z_hat.device)
            scales_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(z_hat.device)
            y_anchor_decode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor
            ).to(z_hat.device)

            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            indexes_anchor = net.gaussian_conditional.build_indexes(scales_anchor_encode)
            anchor_strings = y_strings[slice_index][0]
            anchor_quantized = net.gaussian_conditional.decompress(
                anchor_strings, indexes_anchor, means=means_anchor_encode
            )

            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]

            # checkerboard process 2
            masked_context = net.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = net.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)
            ).chunk(2, 1)

            means_non_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(z_hat.device)
            scales_non_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(z_hat.device)

            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[:, :, 1::2, 0::2]

            indexes_non_anchor = net.gaussian_conditional.build_indexes(
                scales_non_anchor_encode
            )
            non_anchor_strings = y_strings[slice_index][1]
            non_anchor_quantized = net.gaussian_conditional.decompress(
                non_anchor_strings, indexes_non_anchor, means=means_non_anchor_encode
            )

            y_non_anchor_quantized = torch.zeros_like(means_anchor)
            y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
            y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]

            y_slice_hat = y_anchor_decode + y_non_anchor_quantized
            y_hat_slices.append(y_slice_hat)

        y_hat = torch.cat(y_hat_slices, dim=1)
        return y_hat, z_hat

    def extract_latents(
        self, img_uint8_hwc: Any, lambda_idx: int
    ) -> Dict[str, Any]:
        """Extract quantized latents y_hat and z_hat, valid_mask, and metadata."""
        import torch

        net = self.get_model(lambda_idx)
        x_padded, (orig_h, orig_w), (pad_l, pad_r, pad_t, pad_b) = (
            self._prepare_input_tensor(img_uint8_hwc)
        )
        padded_h, padded_w = x_padded.shape[-2], x_padded.shape[-1]

        with torch.no_grad():
            out_enc = net.compress(x_padded)
            # Reconstruct latents through entropy decoding to guarantee bit-exact decoder state
            y_hat, z_hat = self._extract_latents_from_strings(
                net, out_enc["strings"], out_enc["shape"]
            )

        valid_mask = compute_valid_mask(orig_h, orig_w, factor=64, downsample=16)

        meta = {
            "original_size": (orig_h, orig_w),
            "padded_size": (padded_h, padded_w),
            "padding": (pad_l, pad_r, pad_t, pad_b),
            "lambda_idx": lambda_idx,
            "shape": out_enc["shape"],
            "raw_strings": out_enc["strings"],
        }

        # Return y_hat squeezed from (1, C, H, W) to (C, H, W)
        return {
            "y_hat": y_hat.squeeze(0),
            "z_hat": z_hat.squeeze(0),
            "valid_mask": torch.tensor(valid_mask, dtype=torch.bool, device=self.device),
            "meta": meta,
        }

    def decode_from_latents(self, y_hat: Any, meta: Dict[str, Any]) -> Any:
        """Decode reconstruction directly from quantized latent y_hat."""
        import numpy as np
        import torch

        if y_hat.dim() == 3:
            y_hat = y_hat.unsqueeze(0)
        y_hat = y_hat.to(self.device).float()

        lambda_idx = meta["lambda_idx"]
        net = self.get_model(lambda_idx)

        with torch.no_grad():
            x_hat = net.g_s(y_hat).clamp(0.0, 1.0)

        orig_h, orig_w = meta["original_size"]
        x_hat = x_hat[:, :, :orig_h, :orig_w]
        img_np = (
            (x_hat.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)
            .round()
            .astype(np.uint8)
        )
        return img_np
