.PHONY: setup-conda-envs setup-envs download-datasets download-weights test-unit test-server run-baselines train-blind train-conditioned train-config-b train-unmasked train-no-rank train-pure-rank train-teacher-dists train-loss-l1 train-primary train-ablations train-all train-all-seeds retrain-all eval-main eval-refs codec-sanity help

help:
	@echo "nrcomp execution targets:"
	@echo "  make setup-conda-envs   - Create nrcomp-codec and nrcomp-main Conda envs (CUDA 12.4)"
	@echo "  make setup-envs         - Create .venv-codec and .venv-main venvs"
	@echo "  make download-datasets  - Download Kodak, KonIQ-10k, and LiveC into data/raw/"
	@echo "  make download-weights   - Download ELIC checkpoints and run updata.py"
	@echo "  make test-unit          - Run offline unit tests (dev & server)"
	@echo "  make test-server        - Run PyTorch server integration tests (server only)"
	@echo "  make run-baselines      - Run E5 shortcut baselines (a, b, c) on manifest"
	@echo "  make train-all          - Redo all training with all configurations and ablations (seed 42)"
	@echo "  make train-all-seeds    - Redo all training with all configurations across seeds 42, 43, 44"
	@echo "  make retrain-all        - Alias for make train-all"
	@echo "  make train-primary      - Train the 3 primary models (blind_a, conditioned_a, conditioned_b)"
	@echo "  make train-ablations    - Train all architectural ablations (no-rank, pure-rank, unmasked, dists, l1)"
	@echo "  make train-blind        - Train Config A Blind Student"
	@echo "  make train-conditioned  - Train Config A Conditioned Student"
	@echo "  make train-config-b     - Train Config B (y_hat + z_hat) Student"
	@echo "  make train-unmasked     - Train Config A Unmasked Ablation (A7)"
	@echo "  make train-no-rank      - Train Config A No-Rank-Loss Ablation (A2)"
	@echo "  make train-pure-rank    - Train Config A Pure-Rank-Loss Ablation (A2, alpha=1.0)"
	@echo "  make train-teacher-dists- Train Config B DISTS Teacher Ablation (A5)"
	@echo "  make train-loss-l1      - Train Config A L1 / Huber Loss Ablation (A9)"
	@echo "  make eval-main          - Run Stage 5 Evaluation (E1-E4, E5, E8) -> reports/eval_main.md"
	@echo "  make eval-refs          - Run Stage 6 Pixel References & Ablations -> reports/eval_refs.md"
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
	@if [ "$$CONDA_DEFAULT_ENV" = "nrcomp-main" ]; then \
		python -m nrcomp.train --config configs/train_blind.yaml; \
	elif command -v conda >/dev/null && conda info --envs | grep -q "nrcomp-main"; then \
		conda run --no-capture-output -n nrcomp-main python -m nrcomp.train --config configs/train_blind.yaml; \
	elif [ -d ".venv-main" ]; then \
		.venv-main/bin/python -m nrcomp.train --config configs/train_blind.yaml; \
	else \
		python3 -m nrcomp.train --config configs/train_blind.yaml; \
	fi

train-conditioned:
	@if [ "$$CONDA_DEFAULT_ENV" = "nrcomp-main" ]; then \
		python -m nrcomp.train --config configs/train_conditioned.yaml; \
	elif command -v conda >/dev/null && conda info --envs | grep -q "nrcomp-main"; then \
		conda run --no-capture-output -n nrcomp-main python -m nrcomp.train --config configs/train_conditioned.yaml; \
	elif [ -d ".venv-main" ]; then \
		.venv-main/bin/python -m nrcomp.train --config configs/train_conditioned.yaml; \
	else \
		python3 -m nrcomp.train --config configs/train_conditioned.yaml; \
	fi

train-config-b:
	@if [ "$$CONDA_DEFAULT_ENV" = "nrcomp-main" ]; then \
		python -m nrcomp.train --config configs/train_config_b.yaml; \
	elif command -v conda >/dev/null && conda info --envs | grep -q "nrcomp-main"; then \
		conda run --no-capture-output -n nrcomp-main python -m nrcomp.train --config configs/train_config_b.yaml; \
	elif [ -d ".venv-main" ]; then \
		.venv-main/bin/python -m nrcomp.train --config configs/train_config_b.yaml; \
	else \
		python3 -m nrcomp.train --config configs/train_config_b.yaml; \
	fi

train-unmasked:
	@if [ "$$CONDA_DEFAULT_ENV" = "nrcomp-main" ]; then \
		python -m nrcomp.train --config configs/train_unmasked.yaml; \
	elif command -v conda >/dev/null && conda info --envs | grep -q "nrcomp-main"; then \
		conda run --no-capture-output -n nrcomp-main python -m nrcomp.train --config configs/train_unmasked.yaml; \
	elif [ -d ".venv-main" ]; then \
		.venv-main/bin/python -m nrcomp.train --config configs/train_unmasked.yaml; \
	else \
		python3 -m nrcomp.train --config configs/train_unmasked.yaml; \
	fi

train-no-rank:
	@if [ "$$CONDA_DEFAULT_ENV" = "nrcomp-main" ]; then \
		python -m nrcomp.train --config configs/train_no_rank.yaml; \
	elif command -v conda >/dev/null && conda info --envs | grep -q "nrcomp-main"; then \
		conda run --no-capture-output -n nrcomp-main python -m nrcomp.train --config configs/train_no_rank.yaml; \
	elif [ -d ".venv-main" ]; then \
		.venv-main/bin/python -m nrcomp.train --config configs/train_no_rank.yaml; \
	else \
		python3 -m nrcomp.train --config configs/train_no_rank.yaml; \
	fi

train-pure-rank:
	@if [ "$$CONDA_DEFAULT_ENV" = "nrcomp-main" ]; then \
		python -m nrcomp.train --config configs/train_pure_rank.yaml; \
	elif command -v conda >/dev/null && conda info --envs | grep -q "nrcomp-main"; then \
		conda run --no-capture-output -n nrcomp-main python -m nrcomp.train --config configs/train_pure_rank.yaml; \
	elif [ -d ".venv-main" ]; then \
		.venv-main/bin/python -m nrcomp.train --config configs/train_pure_rank.yaml; \
	else \
		python3 -m nrcomp.train --config configs/train_pure_rank.yaml; \
	fi

train-teacher-dists:
	@if [ "$$CONDA_DEFAULT_ENV" = "nrcomp-main" ]; then \
		python -m nrcomp.train --config configs/train_teacher_dists.yaml; \
	elif command -v conda >/dev/null && conda info --envs | grep -q "nrcomp-main"; then \
		conda run --no-capture-output -n nrcomp-main python -m nrcomp.train --config configs/train_teacher_dists.yaml; \
	elif [ -d ".venv-main" ]; then \
		.venv-main/bin/python -m nrcomp.train --config configs/train_teacher_dists.yaml; \
	else \
		python3 -m nrcomp.train --config configs/train_teacher_dists.yaml; \
	fi

train-loss-l1:
	@if [ "$$CONDA_DEFAULT_ENV" = "nrcomp-main" ]; then \
		python -m nrcomp.train --config configs/train_loss_l1.yaml; \
	elif command -v conda >/dev/null && conda info --envs | grep -q "nrcomp-main"; then \
		conda run --no-capture-output -n nrcomp-main python -m nrcomp.train --config configs/train_loss_l1.yaml; \
	elif [ -d ".venv-main" ]; then \
		.venv-main/bin/python -m nrcomp.train --config configs/train_loss_l1.yaml; \
	else \
		python3 -m nrcomp.train --config configs/train_loss_l1.yaml; \
	fi

train-primary: train-blind train-conditioned train-config-b

train-ablations: train-no-rank train-pure-rank train-unmasked train-teacher-dists train-loss-l1

train-all:
	@if [ "$$CONDA_DEFAULT_ENV" = "nrcomp-main" ]; then \
		python scripts/train_all.py; \
	elif command -v conda >/dev/null && conda info --envs | grep -q "nrcomp-main"; then \
		conda run --no-capture-output -n nrcomp-main python scripts/train_all.py; \
	elif [ -d ".venv-main" ]; then \
		.venv-main/bin/python scripts/train_all.py; \
	else \
		python3 scripts/train_all.py; \
	fi

train-all-seeds:
	@if [ "$$CONDA_DEFAULT_ENV" = "nrcomp-main" ]; then \
		python scripts/train_all.py --seeds 42,43,44; \
	elif command -v conda >/dev/null && conda info --envs | grep -q "nrcomp-main"; then \
		conda run --no-capture-output -n nrcomp-main python scripts/train_all.py --seeds 42,43,44; \
	elif [ -d ".venv-main" ]; then \
		.venv-main/bin/python scripts/train_all.py --seeds 42,43,44; \
	else \
		python3 scripts/train_all.py --seeds 42,43,44; \
	fi

retrain-all: train-all

eval-main:
	conda run --no-capture-output -n nrcomp-main python -m nrcomp.evaluate \
		--config configs/evaluate.yaml

eval-refs:
	conda run --no-capture-output -n nrcomp-main python -m nrcomp.eval_pixel_refs \
		--config configs/eval_refs.yaml

codec-sanity:
	@if command -v conda >/dev/null && conda info --envs | grep -q "nrcomp-codec"; then \
		conda run --no-capture-output -n nrcomp-codec python scripts/codec_sanity.py --config configs/codec_sanity.yaml; \
	else \
		.venv-codec/bin/python scripts/codec_sanity.py --config configs/codec_sanity.yaml; \
	fi
