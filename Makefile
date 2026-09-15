# Sentinel — see PLAN.md for the module order.
# Local mode is the default; nothing here needs AWS.

PYTHON      ?= python3.11
# Override on the command line, e.g. make eval DATA=/path/to/splits OUT=/tmp/out
DATA        ?= data/processed
MODELS      ?= models
OUT         ?= results
VENV        ?= .venv
BIN         := $(VENV)/bin
UV          := $(shell command -v uv 2> /dev/null)

# Long runs must survive the laptop's idle-sleep (this Mac sleeps after 1 min;
# sleep cuts USB power and ejects the external data disk). caffeinate -dims
# holds display/idle/disk/system sleep for the duration of the wrapped command.
CAFF        := $(shell command -v caffeinate 2> /dev/null)
KEEPAWAKE   := $(if $(CAFF),caffeinate -dims,)

.DEFAULT_GOAL := help
.PHONY: help setup test lint sample dataset train eval ablation latency tamper demo demo-bootstrap demo-check demo-scenario serve clean

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
	$(KEEPAWAKE) $(BIN)/python scripts/build_dataset.py --root data/raw/r4.2 --months 1

dataset: ## Build the full 17-month train/val/test splits -> data/processed/ (~2 h)
	$(KEEPAWAKE) $(BIN)/python scripts/build_dataset.py --root data/raw/r4.2

train: ## Train autoencoder + isolation forest over 5 seeds -> $(MODELS)
	$(KEEPAWAKE) $(BIN)/python scripts/train.py --data "$(DATA)" --models "$(MODELS)"

eval: ## Regenerate Table V/VI and Figs. 3-4 -> $(OUT)
	$(KEEPAWAKE) $(BIN)/python scripts/evaluate.py --data "$(DATA)" --models "$(MODELS)" --out "$(OUT)"
	$(KEEPAWAKE) $(BIN)/python scripts/ablation.py --data "$(DATA)" --models "$(MODELS)" --out "$(OUT)"

ablation: ## Table VI only -> $(OUT)
	$(KEEPAWAKE) $(BIN)/python scripts/ablation.py --data "$(DATA)" --models "$(MODELS)" --out "$(OUT)"

latency: ## Fig. 5 — added latency by stage, 50k requests at 200 rps
	$(KEEPAWAKE) $(BIN)/python scripts/latency_bench.py

tamper: ## Table VII — 500 tamper attempts + 10k clean control run
	$(KEEPAWAKE) $(BIN)/python scripts/tamper_test.py

serve: ## Run the PDP locally on :8000
	$(BIN)/uvicorn sentinel.pdp.app:app --reload --port 8000

demo: demo-bootstrap ## Split-screen dashboard: Plain IAM vs Sentinel over one insider (Streamlit)
	$(BIN)/streamlit run sentinel/demo/app.py

demo-bootstrap: ## Synthesise models + scenario when the CERT-trained ones are absent (fresh clone)
	$(BIN)/python scripts/demo_bootstrap.py

demo-check: ## Verify Docker, Fabric network, shim, models and scenario before the demo
	$(BIN)/python scripts/demo_check.py

demo-scenario: ## Extract the demo scenario from the test split -> demo/
	$(BIN)/python scripts/demo_scenario.py --data "$(DATA)" --out demo

clean: ## Remove caches and build artefacts (keeps models/ and results/)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage
