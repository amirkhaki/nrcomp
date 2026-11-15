#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo " Setting up nrcomp Virtual Environments for Execution Server"
echo "=========================================================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "[1/2] Creating Codec Environment (.venv-codec)..."
python3 -m venv .venv-codec
.venv-codec/bin/pip install --upgrade pip setuptools wheel
.venv-codec/bin/pip install -r requirements-codec.txt

echo "[2/2] Creating Main Environment (.venv-main)..."
python3 -m venv .venv-main
.venv-main/bin/pip install --upgrade pip setuptools wheel
.venv-main/bin/pip install -r requirements-main.txt

echo "=========================================================="
echo " Environments successfully initialized!"
echo " Codec environment:  source .venv-codec/bin/activate"
echo " Main environment:   source .venv-main/bin/activate"
echo "=========================================================="
