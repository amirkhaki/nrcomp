#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo " Setting up nrcomp Conda Environments via Local Mirrors"
echo " (PyTorch + CUDA via Conda; Resilient PyPI with Timeout)"
echo "=========================================================="

if ! command -v conda &>/dev/null; then
    echo "[ERROR] 'conda' command not found. Please ensure Conda/Miniconda is on your PATH."
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Helper function to install pip packages with extended timeout and mirror fallback
pip_install_safe() {
    local env_name="$1"
    shift
    local packages=("$@")

    echo "[+] Installing ${packages[*]} into ${env_name}..."

    # Attempt 1: Tsinghua TUNA PyPI mirror (ultra-stable and high bandwidth)
    echo "  -> Trying Tsinghua TUNA PyPI mirror (timeout 120s)..."
    if conda run -n "${env_name}" pip install \
        --default-timeout 120 \
        --retries 3 \
        --index-url "https://pypi.tuna.tsinghua.edu.cn/simple/" \
        --trusted-host "pypi.tuna.tsinghua.edu.cn" \
        "${packages[@]}"; then
        return 0
    fi

    # Attempt 2: Kargadan PyPI proxy
    echo "  -> TUNA failed. Retrying with Kargadan PyPI proxy..."
    if conda run -n "${env_name}" pip install \
        --default-timeout 120 \
        --retries 3 \
        --index-url "https://mirror.kargadan.ir/repository/pypi-group/simple/" \
        --trusted-host "mirror.kargadan.ir" \
        "${packages[@]}"; then
        return 0
    fi

    # Attempt 3: Runflare mirror fallback
    echo "  -> Kargadan failed. Retrying with Runflare PyPI mirror..."
    if conda run -n "${env_name}" pip install \
        --default-timeout 120 \
        --retries 3 \
        --index-url "https://mirror-pypi.runflare.com/simple/" \
        --trusted-host "mirror-pypi.runflare.com" \
        "${packages[@]}"; then
        return 0
    fi

    # Attempt 4: Standard pip fallback
    echo "  -> Mirror fallback: Attempting standard pip install..."
    conda run -n "${env_name}" pip install --default-timeout 120 "${packages[@]}"
}

# ------------------------------------------------------------------
# 1. Codec Environment (nrcomp-codec)
# ------------------------------------------------------------------
echo ""
echo "[1/2] Preparing nrcomp-codec environment (Python 3.9 for CompressAI)..."
if conda info --envs | grep -q "nrcomp-codec"; then
    echo "[-] Conda environment 'nrcomp-codec' already exists."
else
    conda create -n nrcomp-codec python=3.9 -y
fi

echo "[+] Verifying PyTorch + CUDA and core tools via Conda..."
conda install -n nrcomp-codec pytorch torchvision pytorch-gpu pyyaml tqdm -y

echo "[+] Installing CompressAI and helper packages via pip..."
pip_install_safe "nrcomp-codec" compressai==1.1.5 thop ptflops timm gdown

# ------------------------------------------------------------------
# 2. Main Environment (nrcomp-main)
# ------------------------------------------------------------------
echo ""
echo "[2/2] Preparing nrcomp-main environment (Python 3.10)..."
if conda info --envs | grep -q "nrcomp-main"; then
    echo "[-] Conda environment 'nrcomp-main' already exists."
else
    conda create -n nrcomp-main python=3.10 -y
fi

echo "[+] Verifying PyTorch + CUDA and scientific stack via Conda..."
conda install -n nrcomp-main pytorch torchvision pytorch-gpu \
    pandas pyarrow scipy scikit-learn matplotlib pyyaml tqdm pytest timm -y

echo "[+] Installing pyiqa via pip..."
pip_install_safe "nrcomp-main" pyiqa

echo ""
echo "=========================================================="
echo " Conda environments successfully initialized!"
echo " Codec environment:  conda activate nrcomp-codec"
echo " Main environment:   conda activate nrcomp-main"
echo "=========================================================="
