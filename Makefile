.PHONY: setup-envs download-datasets download-weights test-unit codec-sanity help

help:
	@echo "nrcomp execution targets:"
	@echo "  make setup-envs         - Create .venv-codec and .venv-main virtualenvs"
	@echo "  make download-datasets  - Download Kodak, KonIQ-10k, and LiveC into data/raw/"
	@echo "  make download-weights   - Download ELIC checkpoints and run updata.py"
	@echo "  make test-unit          - Run offline unit tests with pytest"
	@echo "  make codec-sanity       - Run Stage 1 Codec Sanity Gate on Kodak"

setup-envs:
	bash scripts/setup_envs.sh

download-datasets:
	bash scripts/download_datasets.sh

download-weights:
	.venv-codec/bin/python scripts/download_weights.py --output-dir weights/

test-unit:
	.venv-main/bin/pytest tests/unit -v

codec-sanity:
	.venv-codec/bin/python scripts/codec_sanity.py --config configs/codec_sanity.yaml
