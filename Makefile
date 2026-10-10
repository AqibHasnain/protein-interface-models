PY := .venv/bin/python
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff

.PHONY: help setup test lint data audit report clean

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

setup: ## Create the venv and install the package with dev extras
	uv venv --python 3.12 && uv pip install -e ".[dev]"

lint: ## Lint and format-check
	$(RUFF) check src tests
	$(RUFF) format --check src tests

test: ## Full test suite
	$(PYTEST) tests

data: ## Download and assemble the datasets (idempotent)
	$(PY) -m cla.data.sources
	$(PY) -m cla.data.assemble

audit: ## Phase 1 nuisance-sensitivity audit on all configured models
	$(PY) -m cla.eval.audit --config configs/audit/esm2_35m.yaml
	$(PY) -m cla.eval.audit --config configs/audit/esm2_150m.yaml
	$(PY) -m cla.eval.audit --config configs/audit/esm2_650m.yaml

report: ## Regenerate every figure and table from committed results
	$(PY) -m cla.eval.figures

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
