#!/usr/bin/env python3
"""Download and prepare ELIC checkpoints for all lambdas."""

import argparse
import hashlib
import http.cookiejar
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
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


def is_valid_checkpoint(filepath: Path) -> bool:
    """Check if file exists, is non-empty, and is a valid binary archive (not an HTML error page)."""
    if not filepath.exists():
        return False
    # Checkpoints are ~140MB; anything under 10MB is either truncated or an HTML page
    if filepath.stat().st_size < 10 * 1024 * 1024:
        return False
    try:
        with open(filepath, "rb") as f:
            header = f.read(64)
            if header.startswith(b"<!") or header.startswith(b"<html") or b"<HTML" in header or b"<head" in header:
                return False
            return True
    except Exception:
        return False


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


def _download_via_urllib(gdrive_id: str, dest_path: Path):
    """Download large Google Drive file with cookie & virus-warning confirmation handling."""
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    init_url = f"https://drive.google.com/uc?export=download&id={gdrive_id}"
    req = urllib.request.Request(init_url, headers={"User-Agent": "Mozilla/5.0"})

    with opener.open(req) as resp:
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            dl_resp = resp
        else:
            html = resp.read().decode("utf-8", errors="ignore")
            m = re.search(r'<form[^>]*action=\"([^\"]+)\"[^>]*>(.*?)</form>', html, re.DOTALL)
            if not m:
                raise RuntimeError(f"Could not parse Google Drive virus scan warning form for ID {gdrive_id}")
            action = m.group(1)
            params = {}
            for inp in re.finditer(r'<input[^>]+name=\"([^\"]+)\"[^>]+value=\"([^\"]*)\"', m.group(2)):
                params[inp.group(1)] = inp.group(2)
            query = urllib.parse.urlencode(params)
            download_url = f"{action}?{query}"
            req2 = urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0"})
            dl_resp = opener.open(req2)

        total_size = int(dl_resp.headers.get("Content-Length", 0))
        part_path = dest_path.with_suffix(dest_path.suffix + ".part")
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        print(f"  Streaming ~{total_size / (1024*1024):.1f} MB...")
        with open(part_path, "wb") as f:
            while chunk := dl_resp.read(chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"  [{pct:5.1f}%] {downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB", end="\r")
        print()

        if is_valid_checkpoint(part_path):
            part_path.replace(dest_path)
        else:
            if part_path.exists():
                part_path.unlink()
            raise RuntimeError(f"Downloaded file {part_path} failed validation (size or header check).")


def download_checkpoint(gdrive_id: str, dest_path: Path):
    """Download checkpoint using gdown, CLI gdown, or resilient urllib stream."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")

    # Clean up any leftover corrupt/HTML files
    if dest_path.exists() and not is_valid_checkpoint(dest_path):
        print(f"[!] Removing corrupted/HTML file: {dest_path}")
        dest_path.unlink()
    if part_path.exists():
        part_path.unlink()

    # Strategy 1: Python gdown package
    try:
        import gdown
        print(f"Attempting download via gdown (Python package) for ID {gdrive_id}...")
        res = gdown.download(id=gdrive_id, output=str(part_path), quiet=False, fuzzy=True)
        if res and is_valid_checkpoint(part_path):
            part_path.replace(dest_path)
            print(f"[+] Downloaded successfully via gdown: {dest_path}")
            return
    except Exception as e:
        print(f"  [!] gdown python module failed or not installed: {e}")

    # Strategy 2: CLI gdown executable
    if shutil.which("gdown"):
        try:
            print(f"Attempting download via 'gdown' CLI for ID {gdrive_id}...")
            cmd = ["gdown", "--id", gdrive_id, "-O", str(part_path)]
            subprocess.run(cmd, check=True)
            if is_valid_checkpoint(part_path):
                part_path.replace(dest_path)
                print(f"[+] Downloaded successfully via gdown CLI: {dest_path}")
                return
        except Exception as e:
            print(f"  [!] gdown CLI failed: {e}")

    # Strategy 3: Resilient urllib with CookieJar + Google Drive confirmation parsing
    try:
        print(f"Attempting direct stream via urllib with Google Drive confirmation handling...")
        _download_via_urllib(gdrive_id, dest_path)
        if is_valid_checkpoint(dest_path):
            print(f"[+] Downloaded successfully via urllib: {dest_path}")
            return
    except Exception as e:
        print(f"  [!] urllib download failed: {e}")

    raise RuntimeError(
        f"Failed to download checkpoint for Google Drive ID {gdrive_id}.\n"
        f"Please install gdown (`conda run -n nrcomp-codec pip install gdown`) or download manually to {dest_path}."
    )


KNOWN_BAD_HASHES = {
    "84a21060a9c0d79466b8720c740b09d05c833fcf46f43a33509713d1241bc012",  # Google Drive HTML warning page
}


def update_entropy_model(raw_path: Path, updated_dir: Path, force: bool = False):
    """Update entropy model CDF parameters (replaces updata.py with PyTorch 2.6 compatibility)."""
    updated_dir.mkdir(parents=True, exist_ok=True)

    # elic_lambda_0.45.pth.tar -> elic_lambda_0.45_updated.pth.tar
    updated_filename = raw_path.name.replace(".pth.tar", "_updated.pth.tar")
    if updated_filename == raw_path.name:
        updated_filename = raw_path.stem + "_updated.pth.tar"
    dest_path = updated_dir / updated_filename

    if dest_path.exists() and not force and is_valid_checkpoint(dest_path):
        print(f"  [OK] Updated checkpoint already exists at {dest_path}")
        return

    print(f"Updating entropy model: {raw_path.name} -> {updated_filename}...")

    # Add submodule to sys.path
    submodule_dir = Path(__file__).resolve().parent.parent / "third_party" / "ELiC-ReImplemetation"
    if str(submodule_dir) not in sys.path:
        sys.path.insert(0, str(submodule_dir))

    import torch
    import compressai
    import compressai.models.priors
    if not hasattr(compressai.models.priors, "GaussianConditional"):
        try:
            import compressai.entropy_models
            compressai.models.priors.GaussianConditional = compressai.entropy_models.GaussianConditional
        except (ImportError, AttributeError):
            pass

    from Network import TestModel

    try:
        checkpoint = torch.load(raw_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(raw_path, map_location="cpu")

    if "network" in checkpoint:
        state_dict = checkpoint["network"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_cls = TestModel()
    net = model_cls.from_state_dict(state_dict)
    net.update(force=True)

    updated_state_dict = net.state_dict()
    torch.save(updated_state_dict, dest_path)
    print(f"  [+] Saved updated checkpoint to {dest_path}")


def main():
    parser = argparse.ArgumentParser(description="Download and update ELIC checkpoints.")
    parser.add_argument("--output-dir", type=str, default="weights", help="Base weights directory")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading, only verify/update")
    parser.add_argument("--skip-update", action="store_true", help="Skip running updata.py")
    parser.add_argument("--force-update", action="store_true", help="Force recomputing updated models")
    args = parser.parse_args()

    base_dir = Path(args.output_dir)
    raw_dir = base_dir / "raw"
    updated_dir = base_dir / "updated"
    checksums_file = base_dir / "CHECKSUMS.txt"

    raw_dir.mkdir(parents=True, exist_ok=True)
    known_checksums = load_checksums(checksums_file)
    # Purge any known corrupted hashes from previous runs
    known_checksums = {k: v for k, v in known_checksums.items() if v not in KNOWN_BAD_HASHES}
    updated_checksums = dict(known_checksums)

    for item in CHECKPOINTS:
        filename = item["filename"]
        raw_path = raw_dir / filename

        if not is_valid_checkpoint(raw_path):
            if raw_path.exists():
                print(f"[!] Checkpoint {raw_path} exists but is invalid/corrupt ({raw_path.stat().st_size} bytes). Removing...")
                raw_path.unlink()
            if args.skip_download:
                print(f"[ERROR] Missing valid checkpoint {raw_path}. Run without --skip-download or place manually.")
                sys.exit(1)
            download_checkpoint(item["gdrive_id"], raw_path)

        if not is_valid_checkpoint(raw_path):
            print(f"[ERROR] Checkpoint {raw_path} is invalid after download. Please check Google Drive link.")
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
            update_entropy_model(raw_path, updated_dir, force=args.force_update)

    save_checksums(checksums_file, updated_checksums)
    print("\n[SUCCESS] All ELIC checkpoints verified and ready!")


if __name__ == "__main__":
    main()
