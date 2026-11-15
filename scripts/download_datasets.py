#!/usr/bin/env python3
"""Download and stage datasets using PyIQA's built-in dataset API."""

import argparse
import os
import sys
from pathlib import Path

# Default to HF mirror to prevent network timeouts
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def main():
    parser = argparse.ArgumentParser(description="Download and initialize datasets via PyIQA.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw",
        help="Destination directory for raw datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["kodak", "koniq10k", "clive"],
        help="Datasets to load (default: kodak koniq10k clive)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    try:
        from pyiqa import get_dataset_info, load_dataset
    except ImportError:
        print("[ERROR] pyiqa not found in current Python environment.")
        print("Please run this script inside 'nrcomp-main' where pyiqa is installed:")
        print("  conda run -n nrcomp-main python scripts/download_datasets.py")
        sys.exit(1)

    available_keys = list(get_dataset_info().keys())
    print(f"[INFO] Supported PyIQA dataset keys: {available_keys}")

    for ds_name in args.datasets:
        target_name = ds_name
        # Match common alias for LIVE Challenge
        if target_name == "livec" and "clive" in available_keys:
            target_name = "clive"
        elif target_name not in available_keys and ds_name in available_keys:
            target_name = ds_name

        print(f"\n[+] Fetching/Verifying '{target_name}' via pyiqa.load_dataset into {data_root}...")
        try:
            dataset = load_dataset(target_name, data_root=str(data_root))
            print(f"[SUCCESS] Dataset '{target_name}' ready. Total items: {len(dataset)}")
        except Exception as e:
            print(f"[!] pyiqa.load_dataset failed for '{target_name}': {e}")
            print(f"    You can also use 'bash scripts/download_datasets.sh' as fallback.")


if __name__ == "__main__":
    main()
