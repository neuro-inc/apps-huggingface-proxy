.PHONY: help install test lint format clean build push run dev

# Variables
IMAGE_NAME ?= hf-proxy
IMAGE_TAG ?= latest
REGISTRY ?= docker.io
FULL_IMAGE_NAME = $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies using poetry
	poetry install

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

build: ## Build Docker image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "Image built: $(IMAGE_NAME):$(IMAGE_TAG)"
	@docker images $(IMAGE_NAME):$(IMAGE_TAG)

push: ## Push Docker image to registry
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(FULL_IMAGE_NAME)
	docker push $(FULL_IMAGE_NAME)
	@echo "Image pushed: $(FULL_IMAGE_NAME)"

run: ## Run Docker container locally
	docker run --rm -p 8080:8080 --name hf-proxy-instance $(IMAGE_NAME):$(IMAGE_TAG)

dev: ## Run development server with hot reload
	poetry run uvicorn src.main:app --host 0.0.0.0 --port 8080 --reload

check: lint test ## Run linting and tests

all: clean install lint test build ## Run all checks and build
