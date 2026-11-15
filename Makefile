.PHONY: setup-conda-envs setup-envs download-datasets download-weights test-unit codec-sanity help

help:
	@echo "nrcomp execution targets:"
	@echo "  make setup-conda-envs   - Create nrcomp-codec and nrcomp-main Conda envs (CUDA 12.4)"
	@echo "  make setup-envs         - Create .venv-codec and .venv-main venvs"
	@echo "  make download-datasets  - Download Kodak, KonIQ-10k, and LiveC into data/raw/"
	@echo "  make download-weights   - Download ELIC checkpoints and run updata.py"
	@echo "  make test-unit          - Run offline unit tests with pytest"
	@echo "  make codec-sanity       - Run Stage 1 Codec Sanity Gate on Kodak"

setup-conda-envs:
	bash scripts/setup_conda_envs.sh

setup-envs:
	bash scripts/setup_envs.sh

download-datasets:
	bash scripts/download_datasets.sh

download-weights:
	@if command -v conda >/dev/null && conda info --envs | grep -q "nrcomp-codec"; then \
		conda run -n nrcomp-codec python scripts/download_weights.py --output-dir weights/; \
	else \
		.venv-codec/bin/python scripts/download_weights.py --output-dir weights/; \
	fi

test-unit:
	@if command -v conda >/dev/null && conda info --envs | grep -q "nrcomp-main"; then \
		conda run -n nrcomp-main pytest tests/unit -v; \
	elif [ -d ".venv-main" ]; then \
		.venv-main/bin/pytest tests/unit -v; \
	else \
		python3 -m unittest discover -s tests/unit -p "test_*.py" -v; \
	fi

codec-sanity:
	@if command -v conda >/dev/null && conda info --envs | grep -q "nrcomp-codec"; then \
		conda run -n nrcomp-codec python scripts/codec_sanity.py --config configs/codec_sanity.yaml; \
	else \
		.venv-codec/bin/python scripts/codec_sanity.py --config configs/codec_sanity.yaml; \
	fi
