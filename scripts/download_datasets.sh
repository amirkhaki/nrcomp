#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_RAW="${PROJECT_ROOT}/data/raw"
mkdir -p "${DATA_RAW}"

echo "=========================================================="
echo " Downloading Datasets from PyIQA HuggingFace Repository"
echo " Destination: ${DATA_RAW}"
echo "=========================================================="

BASE_URL="https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets/resolve/main"

# 1. Kodak (24 uncompressed images for Stage 1 sanity gate)
if [ -d "${DATA_RAW}/kodak" ] || [ -d "${DATA_RAW}/Kodak" ]; then
    echo "[-] Kodak dataset already exists, skipping."
else
    echo "[+] Downloading Kodak dataset..."
    curl -L -f --retry 3 -o "${DATA_RAW}/kodak.tgz" "${BASE_URL}/kodak.tgz" || {
        echo "[!] Direct download failed. Please check internet connection or provide Kodak manually in ${DATA_RAW}/kodak"
    }
    if [ -f "${DATA_RAW}/kodak.tgz" ]; then
        mkdir -p "${DATA_RAW}/kodak"
        tar -xzf "${DATA_RAW}/kodak.tgz" -C "${DATA_RAW}/kodak" --strip-components=1 2>/dev/null || tar -xzf "${DATA_RAW}/kodak.tgz" -C "${DATA_RAW}/kodak"
        rm -f "${DATA_RAW}/kodak.tgz"
        echo "[+] Kodak extracted to ${DATA_RAW}/kodak"
    fi
fi

# 2. KonIQ-10k
if [ -d "${DATA_RAW}/koniq-10k" ] || [ -d "${DATA_RAW}/koniq10k" ]; then
    echo "[-] KonIQ-10k dataset already exists, skipping."
else
    echo "[+] Downloading KonIQ-10k dataset archive..."
    curl -L -f --retry 3 -o "${DATA_RAW}/koniq10k.tgz" "${BASE_URL}/koniq10k.tgz" || {
        echo "[!] Downloading koniq10k.tgz failed. If Hugging Face is slow, use huggingface-cli."
    }
    if [ -f "${DATA_RAW}/koniq10k.tgz" ]; then
        mkdir -p "${DATA_RAW}/koniq-10k"
        echo "[+] Extracting KonIQ-10k..."
        tar -xzf "${DATA_RAW}/koniq10k.tgz" -C "${DATA_RAW}/koniq-10k"
        rm -f "${DATA_RAW}/koniq10k.tgz"
        echo "[+] KonIQ-10k extracted to ${DATA_RAW}/koniq-10k"
    fi
fi

# 3. LiveC (CLIVE)
if [ -d "${DATA_RAW}/livec" ] || [ -d "${DATA_RAW}/CLIVE" ]; then
    echo "[-] LiveC dataset already exists, skipping."
else
    echo "[+] Downloading LiveC dataset..."
    curl -L -f --retry 3 -o "${DATA_RAW}/livec.tgz" "${BASE_URL}/livec.tgz" || {
        echo "[!] LiveC download failed or not found under livec.tgz."
    }
    if [ -f "${DATA_RAW}/livec.tgz" ]; then
        mkdir -p "${DATA_RAW}/livec"
        echo "[+] Extracting LiveC..."
        tar -xzf "${DATA_RAW}/livec.tgz" -C "${DATA_RAW}/livec"
        rm -f "${DATA_RAW}/livec.tgz"
        echo "[+] LiveC extracted to ${DATA_RAW}/livec"
    fi
fi

echo "=========================================================="
echo " Dataset download completed!"
echo " Raw directory listing:"
ls -la "${DATA_RAW}"
echo "=========================================================="
