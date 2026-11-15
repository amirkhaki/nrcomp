#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo " Setting up nrcomp Conda Environments via Local Mirrors"
echo " (PyTorch + CUDA via Conda; Kargadan PyPI for extras)"
echo "=========================================================="

if ! command -v conda &>/dev/null; then
    echo "[ERROR] 'conda' command not found. Please ensure Conda/Miniconda is on your PATH."
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Kargadan PyPI mirror (proxies PyPI and pythonhosted.org locally)
KARGADAN_PIP="https://mirror.kargadan.ir/repository/pypi-group/simple/"
TRUSTED_HOST="mirror.kargadan.ir"

# ------------------------------------------------------------------
# 1. Codec Environment (nrcomp-codec)
# ------------------------------------------------------------------
echo ""
echo "[1/2] Creating nrcomp-codec environment (Python 3.9 for CompressAI)..."
if conda info --envs | grep -q "nrcomp-codec"; then
    echo "[-] Conda environment 'nrcomp-codec' already exists."
else
    conda create -n nrcomp-codec python=3.9 -y
fi

echo "[+] Installing PyTorch + CUDA and core tools via Conda..."
conda install -n nrcomp-codec pytorch torchvision pytorch-gpu pyyaml tqdm -y

echo "[+] Installing CompressAI and helper packages via Kargadan PyPI mirror..."
conda run -n nrcomp-codec pip install \
    --index-url "${KARGADAN_PIP}" \
    --trusted-host "${TRUSTED_HOST}" \
    compressai==1.1.5 thop ptflops timm gdown

# ------------------------------------------------------------------
# 2. Main Environment (nrcomp-main)
# ------------------------------------------------------------------
echo ""
echo "[2/2] Creating nrcomp-main environment (Python 3.10)..."
if conda info --envs | grep -q "nrcomp-main"; then
    echo "[-] Conda environment 'nrcomp-main' already exists."
else
    conda create -n nrcomp-main python=3.10 -y
fi

echo "[+] Installing PyTorch + CUDA and scientific stack via Conda..."
conda install -n nrcomp-main pytorch torchvision pytorch-gpu \
    pandas pyarrow scipy scikit-learn matplotlib pyyaml tqdm pytest timm -y

echo "[+] Installing pyiqa via Kargadan PyPI mirror..."
conda run -n nrcomp-main pip install \
    --index-url "${KARGADAN_PIP}" \
    --trusted-host "${TRUSTED_HOST}" \
    pyiqa

echo ""
echo "=========================================================="
echo " Conda environments successfully initialized!"
echo " Codec environment:  conda activate nrcomp-codec"
echo " Main environment:   conda activate nrcomp-main"
echo "=========================================================="
