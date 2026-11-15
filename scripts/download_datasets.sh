#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_RAW="${PROJECT_ROOT}/data/raw"
mkdir -p "${DATA_RAW}"

echo "=========================================================="
echo " Downloading Datasets from PyIQA HuggingFace Repository"
echo " Destination: ${DATA_RAW}"
echo "=========================================================="

PRIMARY_URL="https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets/resolve/main"
MIRROR_URL="https://hf-mirror.com/datasets/chaofengc/IQA-PyTorch-Datasets/resolve/main"

download_with_fallback() {
    local filename="$1"
    local output_path="$2"

    echo "[+] Downloading ${filename}..."
    if curl -L -f --connect-timeout 10 --retry 2 -o "${output_path}" "${PRIMARY_URL}/${filename}"; then
        echo "[+] Successfully downloaded from huggingface.co"
    else
        echo "[!] Primary Hugging Face endpoint unreachable. Retrying with mirror (hf-mirror.com)..."
        curl -L -f --connect-timeout 15 --retry 3 -o "${output_path}" "${MIRROR_URL}/${filename}" || {
            echo "[ERROR] Failed to download ${filename} from both primary and mirror."
            return 1
        }
    fi
}

# 1. Kodak (24 uncompressed images for Stage 1 sanity gate)
if [ -d "${DATA_RAW}/kodak" ] || [ -d "${DATA_RAW}/Kodak" ]; then
    echo "[-] Kodak dataset already exists, skipping."
else
    download_with_fallback "kodak.tgz" "${DATA_RAW}/kodak.tgz" || true
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
    download_with_fallback "koniq10k.tgz" "${DATA_RAW}/koniq10k.tgz" || true
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
    download_with_fallback "livec.tgz" "${DATA_RAW}/livec.tgz" || true
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
