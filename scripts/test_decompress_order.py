#!/usr/bin/env python3
"""Investigate why kodim03 fails when decompress is called twice."""

import sys
import copy
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "ELiC-ReImplemetation"))

from nrcomp.codec.wrapper import ELICWrapper


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = ELICWrapper(weights_dir=PROJECT_ROOT / "weights" / "updated", device=device)
    wrapper.load_model(5, "elic_lambda_0.45_updated.pth.tar")
    net = wrapper.get_model(5)

    img_path = PROJECT_ROOT / "data" / "raw" / "kodak" / "kodim03.png"
    orig_np = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    x_padded, (orig_h, orig_w), _ = wrapper._prepare_input_tensor(orig_np)

    with torch.no_grad():
        out_enc = net.compress(x_padded)
        strings = out_enc["strings"]
        shape = out_enc["shape"]

        print("Testing: calling extract_latents FIRST (without calling net.decompress before it):")
        strings_copy = copy.deepcopy(strings)
        y_hat_1, z_hat_1 = wrapper._extract_latents_from_strings(net, strings_copy, shape)
        x_hat_1 = net.g_s(y_hat_1).clamp(0.0, 1.0)
        mse_1 = torch.nn.functional.mse_loss(x_padded, x_hat_1).item()
        print(f"  First extract: MSE = {mse_1:.6f}, PSNR = {-10 * np.log10(mse_1):.2f} dB")

        print("Testing: calling net.decompress SECOND:")
        strings_copy2 = copy.deepcopy(strings)
        out_dec = net.decompress(strings_copy2, shape)
        x_hat_2 = out_dec["x_hat"].clamp(0.0, 1.0)
        mse_2 = torch.nn.functional.mse_loss(x_padded, x_hat_2).item()
        print(f"  Second decompress: MSE = {mse_2:.6f}, PSNR = {-10 * np.log10(mse_2):.2f} dB")

        print("Testing: calling extract_latents THIRD on fresh copy:")
        strings_copy3 = copy.deepcopy(strings)
        y_hat_3, z_hat_3 = wrapper._extract_latents_from_strings(net, strings_copy3, shape)
        x_hat_3 = net.g_s(y_hat_3).clamp(0.0, 1.0)
        mse_3 = torch.nn.functional.mse_loss(x_padded, x_hat_3).item()
        print(f"  Third extract: MSE = {mse_3:.6f}, PSNR = {-10 * np.log10(mse_3):.2f} dB")


if __name__ == "__main__":
    main()
