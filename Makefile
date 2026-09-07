# ZTBAudit — see PLAN.md for the module order.
# Local mode is the default; nothing here needs AWS.

PYTHON      ?= python3.11
VENV        ?= .venv
BIN         := $(VENV)/bin
UV          := $(shell command -v uv 2> /dev/null)

.DEFAULT_GOAL := help
.PHONY: help setup test lint sample dataset train eval ablation latency tamper demo serve clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create .venv and install pinned dependencies
ifeq ($(UV),)
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
else
	uv venv --python $(PYTHON) $(VENV)
	uv pip install --python $(BIN)/python -r requirements.txt
endif
	@echo "Done. Activate with: source $(BIN)/activate"

test: ## Run the test suite
	$(BIN)/pytest -q

lint: ## Lint and format-check
	$(BIN)/ruff check .

sample: ## Validate the pipeline on a 1-month sample -> data/processed/sample/
	$(BIN)/python scripts/build_dataset.py --root data/raw/r4.2 --months 1

dataset: ## Build the full 17-month train/val/test splits -> data/processed/
	$(BIN)/python scripts/build_dataset.py --root data/raw/r4.2

train: ## Train autoencoder + isolation forest over 5 seeds -> models/
	$(BIN)/python scripts/train.py

eval: ## Regenerate every table and figure in results/ from scratch
	$(BIN)/python scripts/evaluate.py
	$(BIN)/python scripts/ablation.py

ablation: ## Table VI only
	$(BIN)/python scripts/ablation.py

latency: ## Fig. 5 — added latency by stage, 50k requests at 200 rps
	$(BIN)/python scripts/latency_bench.py

tamper: ## Table VII — 500 tamper attempts + 10k clean control run
	$(BIN)/python scripts/tamper_test.py

serve: ## Run the PDP locally on :8000
	$(BIN)/uvicorn ztb.pdp.app:app --reload --port 8000

demo: ## Full demo: PDP + ledger, replay 200 events, live tamper catch
	$(BIN)/python scripts/demo.py

clean: ## Remove caches and build artefacts (keeps models/ and results/)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage
