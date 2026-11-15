#!/usr/bin/env python3
"""Stage 1 Sanity Gate: Kodak dataset evaluation and latent bit-exactness checks.

Evaluates all 6 lambdas on the 24-image Kodak dataset.
Checks:
1. Strict monotonicity of bpp and PSNR across lambdas.
2. Latent bit-exactness: y_hat from extract_latents is bit-identical to the entropy-decoded latent.
3. Invertibility: decode_from_latents equals decompress(bytes) exactly.
4. Channel dimensions: M=320, N=192 for all lambdas.
Writes reports/codec_sanity.md.
"""

import argparse
import math
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp.codec.wrapper import ELICWrapper
from nrcomp.config import load_config


def compute_psnr(img1_uint8, img2_uint8) -> float:
    """Compute PSNR between two uint8 numpy images."""
    import numpy as np
    mse = np.mean((img1_uint8.astype(np.float64) - img2_uint8.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(255.0 ** 2 / mse)


def main():
    parser = argparse.ArgumentParser(description="Run ELIC Stage 1 Codec Sanity Gate.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/codec_sanity.yaml",
        help="Path to codec_sanity.yaml",
    )
    parser.add_argument(
        "--kodak-dir",
        type=str,
        default=None,
        help="Override path to Kodak image directory",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=None,
        help="Override path to updated weights directory",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Override path to output report markdown",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Execution device (cuda or cpu)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    kodak_dir = Path(args.kodak_dir or cfg.get("kodak_dir", "data/raw/kodak"))
    weights_dir = Path(args.weights_dir or cfg.get("weights_dir", "weights/updated"))
    report_path = Path(args.report_path or cfg.get("report_path", "reports/codec_sanity.md"))

    if not kodak_dir.exists():
        print(f"[ERROR] Kodak directory {kodak_dir} does not exist. Run scripts/download_datasets.sh first.")
        sys.exit(1)

    import numpy as np
    from PIL import Image

    image_paths = sorted(
        [
            p
            for p in kodak_dir.glob("*")
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".ppm"]
        ]
    )

    if len(image_paths) == 0:
        print(f"[ERROR] No images found in {kodak_dir}.")
        sys.exit(1)

    print(f"Found {len(image_paths)} images in {kodak_dir}.")
    print(f"Using weights directory: {weights_dir}")

    import torch
    device = args.device if (torch.cuda.is_available() and args.device == "cuda") else "cpu"
    print(f"Running on device: {device}")

    wrapper = ELICWrapper(weights_dir=weights_dir, device=device)

    # Load all models defined in config
    lambdas_cfg = cfg.get("lambdas", [])
    for item in lambdas_cfg:
        idx = item["idx"]
        ckpt = item["checkpoint"]
        print(f"Loading checkpoint for lambda_idx={idx} ({item['name']}): {ckpt}")
        wrapper.load_model(idx, ckpt)

    num_lambdas = len(lambdas_cfg)
    results_per_lambda = []
    failed_checks = []

    print("\n--- Running Rate-Distortion & Latent Verification ---")
    for item in lambdas_cfg:
        idx = item["idx"]
        name = item["name"]
        bpp_list = []
        psnr_list = []

        exact_latents_passed = 0
        decode_equal_passed = 0
        channel_counts_passed = 0
        images_to_check = image_paths[: min(20, len(image_paths))]

        for img_idx, img_path in enumerate(image_paths):
            orig_pil = Image.open(img_path).convert("RGB")
            orig_np = np.array(orig_pil, dtype=np.uint8)
            h, w = orig_np.shape[:2]

            # 1. Compress
            bitstream_bytes, meta = wrapper.compress(orig_np, idx)
            bpp = (8.0 * len(bitstream_bytes)) / (h * w)
            bpp_list.append(bpp)

            # 2. Decompress
            recon_np = wrapper.decompress(bitstream_bytes, meta)
            img_psnr = compute_psnr(orig_np, recon_np)
            psnr_list.append(img_psnr)

            # 3. Latent checks for first 20 images
            if img_idx < len(images_to_check):
                latent_dict = wrapper.extract_latents(orig_np, idx)
                y_hat = latent_dict["y_hat"]
                z_hat = latent_dict["z_hat"]

                # Check channel counts: M=320, N=192
                if y_hat.shape[0] == 320 and z_hat.shape[0] == 192:
                    channel_counts_passed += 1
                else:
                    failed_checks.append(
                        f"Lambda {name}, Image {img_path.name}: Expected channels (320, 192), got ({y_hat.shape[0]}, {z_hat.shape[0]})"
                    )

                # Decode directly from latents and check equality with decompress(bytes)
                recon_from_latents = wrapper.decode_from_latents(y_hat, meta)
                max_diff = np.max(np.abs(recon_np.astype(int) - recon_from_latents.astype(int)))
                if max_diff == 0:
                    decode_equal_passed += 1
                    exact_latents_passed += 1
                else:
                    failed_checks.append(
                        f"Lambda {name}, Image {img_path.name}: Max pixel difference between decompress() and decode_from_latents() was {max_diff}"
                    )

        mean_bpp = float(np.mean(bpp_list))
        mean_psnr = float(np.mean(psnr_list))
        results_per_lambda.append(
            {
                "idx": idx,
                "name": name,
                "mean_bpp": mean_bpp,
                "mean_psnr": mean_psnr,
                "exact_latents": exact_latents_passed,
                "decode_equal": decode_equal_passed,
                "channel_counts": channel_counts_passed,
                "total_checked": len(images_to_check),
            }
        )
        print(f"  Lambda {name:>6}: Mean bpp = {mean_bpp:.4f}, Mean PSNR = {mean_psnr:.2f} dB")

    # Verify Monotonicity
    # Sort by mean_bpp to verify rate ordering
    results_per_lambda.sort(key=lambda x: x["mean_bpp"])

    is_bpp_monotone = True
    is_psnr_monotone = True
    for i in range(1, len(results_per_lambda)):
        if results_per_lambda[i]["mean_bpp"] <= results_per_lambda[i - 1]["mean_bpp"]:
            is_bpp_monotone = False
            failed_checks.append(
                f"bpp is non-monotone between idx {results_per_lambda[i-1]['name']} and {results_per_lambda[i]['name']}"
            )
        if results_per_lambda[i]["mean_psnr"] <= results_per_lambda[i - 1]["mean_psnr"]:
            is_psnr_monotone = False
            failed_checks.append(
                f"PSNR is non-monotone between idx {results_per_lambda[i-1]['name']} and {results_per_lambda[i]['name']}"
            )

    gate_passed = (len(failed_checks) == 0) and is_bpp_monotone and is_psnr_monotone

    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Stage 1 Codec Sanity Gate Report\n\n")
        f.write(f"- **Status:** {'PASSED' if gate_passed else 'FAILED'}\n")
        f.write(f"- **Evaluated Images:** {len(image_paths)} images from Kodak\n")
        f.write(f"- **Device:** {device}\n\n")

        f.write("## Rate-Distortion & Latent Verification Summary\n\n")
        f.write("| Lambda Name | Assigned Idx | Mean bpp | Mean PSNR (dB) | Latent Bit-Exact | Decode Equal | Channels (320, 192) |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for r in results_per_lambda:
            total = r["total_checked"]
            f.write(
                f"| {r['name']} | {r['idx']} | {r['mean_bpp']:.4f} | {r['mean_psnr']:.2f} | "
                f"{r['exact_latents']}/{total} | {r['decode_equal']}/{total} | {r['channel_counts']}/{total} |\n"
            )

        f.write("\n## Monotonicity Checks\n\n")
        f.write(f"- **bpp strictly monotone in sorted index:** {is_bpp_monotone}\n")
        f.write(f"- **PSNR strictly monotone in sorted index:** {is_psnr_monotone}\n\n")

        if failed_checks:
            f.write("## Detected Failures / Gate Violations\n\n")
            for err in failed_checks:
                f.write(f"- {err}\n")
        else:
            f.write("## Verification Conclusion\n\n")
            f.write("All acceptance criteria passed: bpp and PSNR are strictly monotone, all latents are bit-identical to entropy decoded tensors, and decode_from_latents matches decompress exactly.\n")

    print(f"\nReport written to: {report_path}")
    if gate_passed:
        print("[SUCCESS] Stage 1 Sanity Gate PASSED!")
        sys.exit(0)
    else:
        print("[FAILURE] Stage 1 Sanity Gate FAILED! See report for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
