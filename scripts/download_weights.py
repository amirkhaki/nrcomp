#!/usr/bin/env python3
"""Download and prepare ELIC checkpoints for all lambdas."""

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

# Google Drive IDs from community repository README
CHECKPOINTS = [
    {
        "name": "lambda_0.45",
        "lambda_val": 0.45,
        "gdrive_id": "1uuKQJiozcBfgGMJ8CfM6lrXOZWv6RUDN",
        "filename": "elic_lambda_0.45.pth.tar",
    },
    {
        "name": "lambda_0.15",
        "lambda_val": 0.15,
        "gdrive_id": "1s544Uxv0gBY3WvKBcGNb3Fb22zfmd9PL",
        "filename": "elic_lambda_0.15.pth.tar",
    },
    {
        "name": "lambda_0.032",
        "lambda_val": 0.032,
        "gdrive_id": "1Moody9IR8CuAGwLCZ_ZMTfZXT0ehQhqc",
        "filename": "elic_lambda_0.032.pth.tar",
    },
    {
        "name": "lambda_0.016",  # README link text says 0.0016, table header says 0.016
        "lambda_val": 0.016,
        "gdrive_id": "1MWlYAmpHbWlGtG7MBBTPEew800grY5yC",
        "filename": "elic_lambda_0.016.pth.tar",
    },
    {
        "name": "lambda_0.008",
        "lambda_val": 0.008,
        "gdrive_id": "1VNE7rx-rBFLnNFkz56Zc-cPr6xrBBJdL",
        "filename": "elic_lambda_0.008.pth.tar",
    },
    {
        "name": "lambda_0.004",
        "lambda_val": 0.004,
        "gdrive_id": "1YGVJ9bpeEq0xfqka2xkaMzhDkeYFJi6q",
        "filename": "elic_lambda_0.004.pth.tar",
    },
]


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def load_checksums(checksums_file: Path) -> dict:
    """Load known checksums from file."""
    checksums = {}
    if not checksums_file.exists():
        return checksums
    with open(checksums_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                checksums[parts[1]] = parts[0]
    return checksums


def save_checksums(checksums_file: Path, checksums: dict):
    """Write checksums dictionary to file."""
    with open(checksums_file, "w") as f:
        f.write("# SHA256 checksums for ELIC lambda checkpoints\n")
        f.write("# Format: <sha256>  <filename>\n")
        for filename in sorted(checksums.keys()):
            f.write(f"{checksums[filename]}  {filename}\n")


def download_checkpoint(gdrive_id: str, dest_path: Path):
    """Download a file using gdown or warn if unavailable."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    print(f"Downloading from {url} to {dest_path}...")
    try:
        import gdown
        gdown.download(id=gdrive_id, output=str(dest_path), quiet=False)
    except ImportError:
        print("[!] gdown not found. Attempting download via curl...")
        cmd = ["curl", "-L", "-o", str(dest_path), url]
        subprocess.run(cmd, check=True)


def update_entropy_model(raw_path: Path, updated_dir: Path, python_exec: str):
    """Run updata.py from the ELIC submodule to prepare CDFs."""
    submodule_dir = Path(__file__).resolve().parent.parent / "third_party" / "ELiC-ReImplemetation"
    updata_script = submodule_dir / "updata.py"
    if not updata_script.exists():
        raise FileNotFoundError(f"updata.py not found at {updata_script}. Ensure git submodule is initialized.")

    updated_dir.mkdir(parents=True, exist_ok=True)
    updated_name = raw_path.stem + "_updated"
    print(f"Updating entropy model for {raw_path.name} -> {updated_name}...")

    cmd = [
        python_exec,
        str(updata_script),
        str(raw_path),
        "-n", updated_name,
        "-d", str(updated_dir),
    ]
    # Set PYTHONPATH to include submodule directory for Network import
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{submodule_dir}:{current_pythonpath}"
    subprocess.run(cmd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description="Download and update ELIC checkpoints.")
    parser.add_argument("--output-dir", type=str, default="weights", help="Base weights directory")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading, only verify/update")
    parser.add_argument("--skip-update", action="store_true", help="Skip running updata.py")
    parser.add_argument("--python-exec", type=str, default=sys.executable, help="Python executable for updata.py")
    args = parser.parse_args()

    base_dir = Path(args.output_dir)
    raw_dir = base_dir / "raw"
    updated_dir = base_dir / "updated"
    checksums_file = base_dir / "CHECKSUMS.txt"

    raw_dir.mkdir(parents=True, exist_ok=True)
    known_checksums = load_checksums(checksums_file)
    updated_checksums = dict(known_checksums)

    for item in CHECKPOINTS:
        filename = item["filename"]
        raw_path = raw_dir / filename

        if not raw_path.exists():
            if args.skip_download:
                print(f"[ERROR] Missing {raw_path}. Run without --skip-download or place manually.")
                sys.exit(1)
            download_checkpoint(item["gdrive_id"], raw_path)

        if not raw_path.exists() or raw_path.stat().st_size == 0:
            print(f"[ERROR] Checkpoint {raw_path} is missing or empty. Please check Google Drive link.")
            sys.exit(1)

        # Verify or record checksum
        sha = compute_sha256(raw_path)
        print(f"[SHA256] {filename}: {sha}")
        if filename in known_checksums:
            if known_checksums[filename] != sha:
                print(f"[ERROR] Checksum mismatch for {filename}!")
                print(f"  Expected: {known_checksums[filename]}")
                print(f"  Actual:   {sha}")
                sys.exit(1)
            else:
                print(f"  [OK] Checksum matches verified hash.")
        else:
            updated_checksums[filename] = sha

        # Update entropy model if not skipped
        if not args.skip_update:
            update_entropy_model(raw_path, updated_dir, args.python_exec)

    save_checksums(checksums_file, updated_checksums)
    print("\n[SUCCESS] All ELIC checkpoints verified and ready!")


if __name__ == "__main__":
    main()
