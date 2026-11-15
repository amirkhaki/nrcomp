#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo " Setting up nrcomp Conda Environments (CUDA 12.4 Support)"
echo "=========================================================="

if ! command -v conda &>/dev/null; then
    echo "[ERROR] 'conda' command not found. Please ensure Conda/Miniconda is on your PATH."
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Optional: Set Kargadan PyPI proxy if requested or if direct pypi is restricted
USE_KARGADAN_PIP="${USE_KARGADAN_PIP:-1}"
PIP_EXTRA_FLAGS=()
if [ "${USE_KARGADAN_PIP}" = "1" ]; then
    echo "[INFO] Using Kargadan PyPI mirror for accelerated local package downloads."
    PIP_EXTRA_FLAGS+=(--index-url "https://mirror.kargadan.ir/repository/pypi-group/simple/" --trusted-host "mirror.kargadan.ir")
fi

# ------------------------------------------------------------------
# 1. Codec Environment (nrcomp-codec)
# ------------------------------------------------------------------
echo ""
echo "[1/2] Creating nrcomp-codec environment (Python 3.9 for CompressAI)..."
if conda info --envs | grep -q "nrcomp-codec"; then
    echo "[-] Conda environment 'nrcomp-codec' already exists. Skipping creation."
else
    conda create -n nrcomp-codec python=3.9 -y
fi

echo "Installing PyTorch (CUDA 12.4) into nrcomp-codec..."
conda run -n nrcomp-codec pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 || {
    echo "[!] Wheel install failed, falling back to Conda channel..."
    conda install -n nrcomp-codec pytorch torchvision -y
}

echo "Installing remaining codec dependencies..."
if [ ${#PIP_EXTRA_FLAGS[@]} -gt 0 ]; then
    conda run -n nrcomp-codec pip install "${PIP_EXTRA_FLAGS[@]}" --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements-codec.txt
else
    conda run -n nrcomp-codec pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements-codec.txt
fi

# ------------------------------------------------------------------
# 2. Main Environment (nrcomp-main)
# ------------------------------------------------------------------
echo ""
echo "[2/2] Creating nrcomp-main environment (Python 3.10)..."
if conda info --envs | grep -q "nrcomp-main"; then
    echo "[-] Conda environment 'nrcomp-main' already exists. Skipping creation."
else
    conda create -n nrcomp-main python=3.10 -y
fi

echo "Installing PyTorch (CUDA 12.4) into nrcomp-main..."
conda run -n nrcomp-main pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 || {
    echo "[!] Wheel install failed, falling back to Conda channel..."
    conda install -n nrcomp-main pytorch torchvision -y
}

echo "Installing remaining main dependencies (pyiqa, timm, matplotlib, pandas, etc.)..."
if [ ${#PIP_EXTRA_FLAGS[@]} -gt 0 ]; then
    conda run -n nrcomp-main pip install "${PIP_EXTRA_FLAGS[@]}" --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements-main.txt
else
    conda run -n nrcomp-main pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements-main.txt
fi

echo ""
echo "=========================================================="
echo " Conda environments successfully configured for CUDA 12.4!"
echo " Codec environment:  conda activate nrcomp-codec"
echo " Main environment:   conda activate nrcomp-main"
echo "=========================================================="
