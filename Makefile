.PHONY: help lint format test test-e2e test-multigpu

help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

lint:  ## Run ruff, mypy, bandit at the CI-pinned versions (via pre-commit)
	pre-commit run --all-files

format:  ## Auto-fix and format with the pinned ruff (same order as pre-commit)
	uvx ruff@0.15.8 check --fix
	uvx ruff@0.15.8 format

test:  ## Run the CPU suite the way CI does (excludes GPU e2e)
	pytest -n4 --dist loadfile --ignore=tests/e2e tests/

test-e2e:  ## Run the single-GPU e2e suite (needs an NVIDIA GPU)
	pytest tests/e2e/ --ignore=tests/e2e/multigpu/

test-multigpu:  ## Run the multi-GPU e2e suite (needs >= 2 GPUs)
	pytest tests/e2e/multigpu/
