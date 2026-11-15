#!/usr/bin/env python3
"""Diagnostic script to inspect Kodak encoding/decoding and latent consistency."""

import sys
import math
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "ELiC-ReImplemetation"))

from nrcomp.codec.wrapper import ELICWrapper


def compute_psnr_float(t1, t2) -> float:
    import torch
    mse = torch.nn.functional.mse_loss(t1, t2).item()
    if mse == 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running diagnostic on device: {device}")

    weights_dir = PROJECT_ROOT / "weights" / "updated"
    wrapper = ELICWrapper(weights_dir=weights_dir, device=device)

    # Test lambdas: lowest (0.004) and highest (0.45)
    test_lambdas = [(0, "0.004", "elic_lambda_0.004_updated.pth.tar"),
                    (5, "0.45", "elic_lambda_0.45_updated.pth.tar")]

    for idx, name, ckpt in test_lambdas:
        print(f"\n=======================================================")
        print(f" Testing Lambda {name} (idx {idx})")
        print(f"=======================================================")
        wrapper.load_model(idx, ckpt)
        net = wrapper.get_model(idx)

        # Test both landscape (kodim01) and portrait (kodim05)
        for img_name in ["kodim01.png", "kodim03.png", "kodim05.png"]:
            img_path = PROJECT_ROOT / "data" / "raw" / "kodak" / img_name
            if not img_path.exists():
                print(f"[!] {img_path} not found.")
                continue

            orig_pil = Image.open(img_path).convert("RGB")
            orig_np = np.array(orig_pil, dtype=np.uint8)

            # Prepare tensor
            x_padded, (orig_h, orig_w), _ = wrapper._prepare_input_tensor(orig_np)

            with torch.no_grad():
                # 1. Direct forward inference (no entropy coding)
                out_inf = net.inference(x_padded)
                x_hat_inf = out_inf["x_hat"][:, :, :orig_h, :orig_w].clamp(0.0, 1.0)
                psnr_inf = compute_psnr_float(x_padded[:, :, :orig_h, :orig_w], x_hat_inf)

                # 2. Compress
                out_enc = net.compress(x_padded)
                strings, shape = out_enc["strings"], out_enc["shape"]

                # 3. Submodule decompress
                out_dec = net.decompress(strings, shape)
                x_hat_dec = out_dec["x_hat"][:, :, :orig_h, :orig_w].clamp(0.0, 1.0)
                psnr_dec = compute_psnr_float(x_padded[:, :, :orig_h, :orig_w], x_hat_dec)

                # 4. Wrapper _extract_latents_from_strings
                y_hat, z_hat = wrapper._extract_latents_from_strings(net, strings, shape)
                x_hat_lat = net.g_s(y_hat).clamp(0.0, 1.0)[:, :, :orig_h, :orig_w]
                psnr_lat = compute_psnr_float(x_padded[:, :, :orig_h, :orig_w], x_hat_lat)

                # 5. Measure differences
                diff_dec_lat = (x_hat_dec - x_hat_lat).abs().max().item() * 255.0
                diff_inf_dec = (x_hat_inf - x_hat_dec).abs().max().item() * 255.0

            print(f" Image {img_name} ({orig_w}x{orig_h}):")
            print(f"   PSNR inf (no entropy coding): {psnr_inf:.2f} dB")
            print(f"   PSNR dec (net.decompress):    {psnr_dec:.2f} dB")
            print(f"   PSNR lat (from y_hat):        {psnr_lat:.2f} dB")
            print(f"   Max pixel diff (dec vs lat):  {diff_dec_lat:.3f}")
            print(f"   Max pixel diff (inf vs dec):  {diff_inf_dec:.3f}")


if __name__ == "__main__":
    main()
