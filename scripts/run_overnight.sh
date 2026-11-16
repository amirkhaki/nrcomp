#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p reports data/latents data/recons data/splits

export PYTHONUNBUFFERED=1

LOG_FILE="${PROJECT_ROOT}/reports/overnight.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================================="
echo " Starting Overnight End-to-End Pipeline Run"
echo " Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Working Directory: ${PROJECT_ROOT}"
echo " Log file: ${LOG_FILE}"
echo "=========================================================="

# 1. Download datasets if missing (using Python downloader with PyIQA)
echo ""
echo "=== Step 1: Checking Datasets (Python Downloader) ==="
conda run --no-capture-output -n nrcomp-main python scripts/download_datasets.py || true

# Ensure pandas & pyarrow are in codec env if not present
conda run --no-capture-output -n nrcomp-codec pip install \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    pandas pyarrow || true

# Ensure piq, compressai, thop, ptflops are in main env (for HaarPSI & on-the-fly decoding)
conda run --no-capture-output -n nrcomp-main pip install \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    piq "compressai==1.1.5" thop ptflops || true

# 2. Stage 2: Pilot Data Generation (100 KonIQ images)
echo ""
echo "=== Step 2: Running 100-Image Pilot Data Generation (Stage 2) ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
conda run --no-capture-output -n nrcomp-codec python -m nrcomp.generate --config configs/generate_pilot.yaml

# 3. Stage 3: Pilot Quality Metric Labelling
echo ""
echo "=== Step 3: Computing Quality Metric Labels for Pilot (Stage 3) ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
conda run --no-capture-output -n nrcomp-main python -m nrcomp.label --config configs/label.yaml --manifest data/manifest_pilot.parquet

# 4. Stage 3: Pilot Diagnostic Gates
echo ""
echo "=== Step 4: Generating Stage 3 Pilot Diagnostic Reports ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
conda run --no-capture-output -n nrcomp-main python scripts/label_diagnostics.py --manifest data/manifest_pilot.parquet --output reports/label_diagnostics.md
conda run --no-capture-output -n nrcomp-main python scripts/latent_diagnostics.py --manifest data/manifest_pilot.parquet --output reports/latent_diagnostics.md

echo ""
echo "----------------------------------------------------------"
echo " [SUCCESS] Pilot Pipeline Completed Successfully!"
echo " Reports generated: reports/label_diagnostics.md & reports/latent_diagnostics.md"
echo "----------------------------------------------------------"

# 5. Stage 2: Dataset Data Generation (KonIQ-10k & LiveC)
# Default is configs/generate_subset.yaml (fits safely within 15GB disk budget)
GENERATE_CONFIG="${1:-configs/generate_subset.yaml}"
echo ""
echo "=== Step 5: Running Dataset Generation (${GENERATE_CONFIG}) ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
conda run --no-capture-output -n nrcomp-codec python -m nrcomp.generate --config "${GENERATE_CONFIG}"

# 6. Stage 3: Full Quality Metric Labelling
echo ""
echo "=== Step 6: Computing Quality Metric Labels for Full Dataset ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
conda run --no-capture-output -n nrcomp-main python -m nrcomp.label --config configs/label.yaml --manifest data/manifest.parquet

# 7. Stage 3: Full Dataset Diagnostic Reports
echo ""
echo "=== Step 7: Generating Stage 3 Full Diagnostic Reports ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
conda run --no-capture-output -n nrcomp-main python scripts/label_diagnostics.py --manifest data/manifest.parquet --output reports/label_diagnostics_full.md
conda run --no-capture-output -n nrcomp-main python scripts/latent_diagnostics.py --manifest data/manifest.parquet --output reports/latent_diagnostics_full.md

echo ""
echo "=========================================================="
echo " [ALL COMPLETE] Overnight Pipeline Finished Successfully!"
echo " Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Final Manifest: data/manifest.parquet"
echo "=========================================================="
