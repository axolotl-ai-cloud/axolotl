.PHONY: help lint format test test-e2e test-multigpu

# ruff version, read from the single source of truth so it never drifts from CI.
# Lazy (=, not :=) so the lookup only runs for `format`, not every target.
RUFF_REV = $(shell awk '/ruff-pre-commit/{f=1} f&&/rev:/{sub(/^v/,"",$$2);print $$2;exit}' .pre-commit-config.yaml)

help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

lint:  ## Run ruff, mypy, bandit at the CI-pinned versions (via pre-commit)
	pre-commit run --all-files

format:  ## Auto-fix and format with the pinned ruff (same order as pre-commit)
	uvx ruff@$(RUFF_REV) check --fix
	uvx ruff@$(RUFF_REV) format

test:  ## Run the CPU suite the way CI does (excludes GPU e2e)
	pytest -m "not slow" -n4 --dist loadfile --ignore=tests/e2e tests/

test-e2e:  ## Run the single-GPU e2e suite (needs an NVIDIA GPU)
	pytest tests/e2e/ --ignore=tests/e2e/multigpu/

test-multigpu:  ## Run the multi-GPU e2e suite (needs >= 2 GPUs)
	pytest tests/e2e/multigpu/
