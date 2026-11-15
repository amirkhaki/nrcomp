#!/usr/bin/env python3
"""Download and stage datasets using PyIQA's built-in dataset API and Kodak fetcher."""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

# Default to HF mirror to prevent network timeouts
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def download_kodak(target_dir: Path):
    """Download the 24 classic Kodak benchmark images for Stage 1 sanity check."""
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        [
            p
            for p in target_dir.glob("*")
            if p.suffix.lower() in [".png", ".jpg", ".ppm"]
        ]
    )
    if len(existing) >= 24:
        print(f"[-] Kodak dataset already contains {len(existing)} images in {target_dir}. Skipping.")
        return

    print(f"[+] Downloading 24 Kodak benchmark images into {target_dir}...")
    url_sources = [
        "https://r0k.us/graphics/kodak/kodak/kodim{:02d}.png",
        "https://raw.githubusercontent.com/girfa/ColorImageDatasets/master/Kodak-PCD0992/kodim{:02d}.png",
    ]

    for i in range(1, 25):
        filename = f"kodim{i:02d}.png"
        dest_path = target_dir / filename
        if dest_path.exists() and dest_path.stat().st_size > 0:
            continue

        downloaded = False
        for pattern in url_sources:
            url = pattern.format(i)
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req, timeout=15) as resp, open(dest_path, "wb") as f:
                    f.write(resp.read())
                downloaded = True
                break
            except Exception:
                continue

        if downloaded:
            print(f"  [+] Downloaded {filename} ({dest_path.stat().st_size // 1024} KB)")
        else:
            print(f"  [!] Failed to download {filename} from all mirrors.")

    final_count = len(list(target_dir.glob("kodim*.png")))
    if final_count >= 24:
        print(f"[SUCCESS] All 24 Kodak images ready in {target_dir}.")
    else:
        print(f"[WARN] Only {final_count}/24 Kodak images downloaded.")


def main():
    parser = argparse.ArgumentParser(description="Download and initialize datasets.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw",
        help="Destination directory for raw datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["kodak", "koniq10k", "livec"],
        help="Datasets to load (default: kodak koniq10k livec)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    try:
        from pyiqa import get_dataset_info, load_dataset
        available_keys = list(get_dataset_info().keys())
        print(f"[INFO] Supported PyIQA dataset keys: {available_keys}")
    except ImportError:
        print("[WARN] pyiqa not found in current Python environment. Proceeding with external fetchers...")
        available_keys = []

    for ds_name in args.datasets:
        if ds_name.lower() == "kodak":
            download_kodak(data_root / "kodak")
            continue

        target_name = ds_name.lower()
        if target_name in ["livec", "clive"]:
            target_name = "livec" if "livec" in available_keys else "live"

        print(f"\n[+] Fetching/Verifying '{target_name}' via pyiqa.load_dataset into {data_root}...")
        try:
            dataset = load_dataset(target_name, data_root=str(data_root))
            print(f"[SUCCESS] Dataset '{target_name}' ready. Total items: {len(dataset)}")
        except Exception as e:
            print(f"[!] pyiqa.load_dataset failed for '{target_name}': {e}")
            print(f"    You can also use 'bash scripts/download_datasets.sh' as fallback.")


if __name__ == "__main__":
    main()
