.PHONY: setup-conda-envs setup-envs download-datasets download-weights test-unit test-server run-baselines train-blind train-conditioned train-config-b codec-sanity help

help:
	@echo "nrcomp execution targets:"
	@echo "  make setup-conda-envs   - Create nrcomp-codec and nrcomp-main Conda envs (CUDA 12.4)"
	@echo "  make setup-envs         - Create .venv-codec and .venv-main venvs"
	@echo "  make download-datasets  - Download Kodak, KonIQ-10k, and LiveC into data/raw/"
	@echo "  make download-weights   - Download ELIC checkpoints and run updata.py"
	@echo "  make test-unit          - Run offline unit tests (dev & server)"
	@echo "  make test-server        - Run PyTorch server integration tests (server only)"
	@echo "  make run-baselines      - Run E5 shortcut baselines (a, b, c) on manifest"
	@echo "  make train-blind        - Train Config A Blind Student"
	@echo "  make train-conditioned  - Train Config A Conditioned Student"
	@echo "  make train-config-b     - Train Config B (y_hat + z_hat) Student"
	@echo "  make codec-sanity       - Run Stage 1 Codec Sanity Gate on Kodak"

setup-conda-envs:
	bash scripts/setup_conda_envs.sh

setup-envs:
	bash scripts/setup_envs.sh

download-datasets:
	@if command -v conda >/dev/null && conda info --envs | grep -q "nrcomp-main"; then \
		conda run --no-capture-output -n nrcomp-main python scripts/download_datasets.py --data-root data/raw; \
	else \
		bash scripts/download_datasets.sh; \
	fi

download-weights:
	@if command -v conda >/dev/null && conda info --envs | grep -q "nrcomp-codec"; then \
		conda run --no-capture-output -n nrcomp-codec python scripts/download_weights.py --output-dir weights/; \
	else \
		.venv-codec/bin/python scripts/download_weights.py --output-dir weights/; \
	fi

test-unit:
	@if command -v conda >/dev/null && conda info --envs | grep -q "nrcomp-main"; then \
		conda run --no-capture-output -n nrcomp-main pytest tests/unit -v; \
	elif [ -d ".venv-main" ]; then \
		.venv-main/bin/pytest tests/unit -v; \
	else \
		python3 -m unittest discover -s tests/unit -p "test_*.py" -v; \
	fi

test-server:
	conda run --no-capture-output -n nrcomp-main pytest tests/server -v

run-baselines:
	conda run --no-capture-output -n nrcomp-main python scripts/shortcut_baselines.py \
		--manifest data/manifest.parquet \
		--output reports/e5_baselines.md

train-blind:
	conda run --no-capture-output -n nrcomp-main python -m nrcomp.train \
		--config configs/train_blind.yaml

train-conditioned:
	conda run --no-capture-output -n nrcomp-main python -m nrcomp.train \
		--config configs/train_conditioned.yaml

train-config-b:
	conda run --no-capture-output -n nrcomp-main python -m nrcomp.train \
		--config configs/train_config_b.yaml

codec-sanity:
	@if command -v conda >/dev/null && conda info --envs | grep -q "nrcomp-codec"; then \
		conda run --no-capture-output -n nrcomp-codec python scripts/codec_sanity.py --config configs/codec_sanity.yaml; \
	else \
		.venv-codec/bin/python scripts/codec_sanity.py --config configs/codec_sanity.yaml; \
	fi
