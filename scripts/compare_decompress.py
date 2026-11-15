#!/usr/bin/env python3
"""Find the exact line where net.decompress diverges from wrapper._extract_latents_from_strings."""

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

        # Run extract
        y_hat_extract, z_hat_extract = wrapper._extract_latents_from_strings(net, copy.deepcopy(strings), shape)
        x_hat_lat = net.g_s(y_hat_extract).clamp(0.0, 1.0)
        psnr_extract = -10 * np.log10(torch.nn.functional.mse_loss(x_padded, x_hat_lat).item())

        # Intercept net.decompress intermediate tensors
        orig_decompress = net.decompress

        # Step by step inside net.decompress
        z_hat_dec = net.entropy_bottleneck.decompress(copy.deepcopy(strings[1]), shape)
        print(f"z_hat equal: {torch.equal(z_hat_extract, z_hat_dec)}")

        # Step-by-step slice comparison
        latent_means, latent_scales = net.h_s(z_hat_dec).chunk(2, 1)
        y_strings = copy.deepcopy(strings[0])
        B = z_hat_dec.size(0)
        ctx_params_anchor = torch.zeros(
            (B, net.M * 2, z_hat_dec.shape[2] * 4, z_hat_dec.shape[3] * 4)
        ).to(z_hat_dec.device)
        ctx_params_anchor_split = torch.split(
            ctx_params_anchor, [2 * i for i in net.groups[1:]], 1
        )

        y_hat_slices_manual = []
        for slice_index in range(len(net.groups) - 1):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices_manual[0]
                support_slices_ch = net.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            else:
                support_slices = torch.concat(
                    [y_hat_slices_manual[0], y_hat_slices_manual[slice_index - 1]], dim=1
                )
                support_slices_ch = net.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)

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
            means_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat_dec.device)
            scales_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat_dec.device)
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(z_hat_dec.device)

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

            masked_context = net.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = net.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)
            ).chunk(2, 1)

            means_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat_dec.device)
            scales_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat_dec.device)
            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[:, :, 1::2, 0::2]

            indexes_non_anchor = net.gaussian_conditional.build_indexes(scales_non_anchor_encode)
            non_anchor_strings = y_strings[slice_index][1]
            non_anchor_quantized = net.gaussian_conditional.decompress(
                non_anchor_strings, indexes_non_anchor, means=means_non_anchor_encode
            )

            y_non_anchor_quantized = torch.zeros_like(means_anchor)
            y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
            y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]

            y_slice_hat = y_anchor_decode + y_non_anchor_quantized
            y_hat_slices_manual.append(y_slice_hat)

        y_hat_manual = torch.cat(y_hat_slices_manual, dim=1)
        print(f"y_hat_extract equal y_hat_manual: {torch.equal(y_hat_extract, y_hat_manual)}")

        out_dec = net.decompress(copy.deepcopy(strings), shape)
        x_hat_dec = out_dec["x_hat"].clamp(0.0, 1.0)
        psnr_dec = -10 * np.log10(torch.nn.functional.mse_loss(x_padded, x_hat_dec).item())

        print(f"PSNR extract: {psnr_extract:.2f} dB")
        print(f"PSNR net.decompress: {psnr_dec:.2f} dB")
        print(f"Difference in output: {(x_hat_lat - x_hat_dec).abs().max().item() * 255:.3f}")


if __name__ == "__main__":
    main()
