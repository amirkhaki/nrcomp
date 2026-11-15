#!/usr/bin/env python3
"""Test if cuDNN / TF32 non-determinism causes index mismatches between compress and decompress."""

import sys
import copy
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "ELiC-ReImplemetation"))

from nrcomp.codec.wrapper import ELICWrapper


def test_determinism(enable_deterministic_settings: bool):
    import torch

    if enable_deterministic_settings:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("\n--- Testing WITH deterministic settings (TF32 disabled, deterministic cuDNN) ---")
    else:
        print("\n--- Testing WITHOUT deterministic settings (default PyTorch) ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = ELICWrapper(weights_dir=PROJECT_ROOT / "weights" / "updated", device=device)
    wrapper.load_model(5, "elic_lambda_0.45_updated.pth.tar")
    net = wrapper.get_model(5)

    for img_name in ["kodim01.png", "kodim03.png", "kodim05.png"]:
        img_path = PROJECT_ROOT / "data" / "raw" / "kodak" / img_name
        if not img_path.exists():
            continue
        orig_np = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        x_padded, (orig_h, orig_w), _ = wrapper._prepare_input_tensor(orig_np)

        with torch.no_grad():
            out_enc = net.compress(x_padded)
            strings = out_enc["strings"]
            shape = out_enc["shape"]

            # Dec 1: net.decompress
            out_dec = net.decompress(copy.deepcopy(strings), shape)
            x_hat_dec = out_dec["x_hat"][:, :, :orig_h, :orig_w].clamp(0.0, 1.0)
            mse_dec = torch.nn.functional.mse_loss(x_padded[:, :, :orig_h, :orig_w], x_hat_dec).item()
            psnr_dec = -10.0 * np.log10(mse_dec) if mse_dec > 0 else float("inf")

            # Dec 2: extract_latents + decode
            y_hat, z_hat = wrapper._extract_latents_from_strings(net, copy.deepcopy(strings), shape)
            x_hat_lat = net.g_s(y_hat)[:, :, :orig_h, :orig_w].clamp(0.0, 1.0)
            mse_lat = torch.nn.functional.mse_loss(x_padded[:, :, :orig_h, :orig_w], x_hat_lat).item()
            psnr_lat = -10.0 * np.log10(mse_lat) if mse_lat > 0 else float("inf")

            max_diff = (x_hat_dec - x_hat_lat).abs().max().item() * 255.0
            print(f"[{img_name}] PSNR dec: {psnr_dec:.2f} dB | PSNR lat: {psnr_lat:.2f} dB | Max diff: {max_diff:.3f}")


if __name__ == "__main__":
    test_determinism(False)
    test_determinism(True)
