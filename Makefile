.PHONY: help install setup test lint format clean build build-hook-image push push-hook-image run dev gen-types-schemas

# Variables
IMAGE_NAME ?= ghcr.io/neuro-inc/apps-huggingface-proxy
IMAGE_TAG ?= latest
FULL_IMAGE_NAME = $(IMAGE_NAME):$(IMAGE_TAG)

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies using poetry
	poetry install

setup: install ## Install dependencies and pre-commit hooks
	poetry run pre-commit install

test: ## Run tests with pytest
	poetry run pytest

lint: ## Run linting with ruff
	poetry run ruff check src/ tests/

format: ## Format code with ruff
	poetry run ruff format src/ tests/
	poetry run ruff check --fix src/ tests/

clean: ## Clean up build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	rm -rf dist/ build/ htmlcov/


.PHONY: build-hook-image
build-hook-image:
	docker build \
		-t $(IMAGE_NAME):latest \
		-f hooks.Dockerfile \
		.;

.PHONY: push-hook-image
push-hook-image:
	docker tag $(IMAGE_NAME):latest ghcr.io/neuro-inc/$(IMAGE_NAME):$(IMAGE_TAG)
	docker push ghcr.io/neuro-inc/$(IMAGE_NAME):$(IMAGE_TAG)

run: ## Run Docker container locally
	docker run --rm -p 8080:8080 --name hf-proxy-instance $(IMAGE_NAME):latest

dev: ## Run development server with hot reload
	poetry run uvicorn src.main:app --host 0.0.0.0 --port 8080 --reload

check: lint test ## Run linting and tests

all: clean install lint test build ## Run all checks and build

gen-types-schemas: ## Generate JSON schemas from Pydantic types
	poetry run app-types dump-types-schema .apolo/src/apolo_apps_hf_proxy HfProxyInputs .apolo/src/apolo_apps_hf_proxy/schemas/HfProxyInputs.json
	poetry run app-types dump-types-schema .apolo/src/apolo_apps_hf_proxy HfProxyOutputs .apolo/src/apolo_apps_hf_proxy/schemas/HfProxyOutputs.json
