.PHONY: help up down test lint typecheck build clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

up: ## Start all services (development)
	docker compose up -d

down: ## Stop all services
	docker compose down

test: ## Run all tests
	cd packages/dana-common && python -m pytest tests/ -v
	cd services/api-gateway && python -m pytest tests/ -v
	cd services/auth-service && python -m pytest tests/ -v
	cd services/inference-router && python -m pytest tests/ -v
	cd services/inference-worker && python -m pytest tests/ -v
	cd services/billing-service && python -m pytest tests/ -v
	cd services/analytics-service && python -m pytest tests/ -v
	cd services/model-registry && python -m pytest tests/ -v

lint: ## Run linter
	ruff check packages/ services/

typecheck: ## Run type checker
	mypy packages/dana-common/src services/*/src --ignore-missing-imports

build: ## Build all Docker images
	docker compose build

clean: ## Remove all containers, volumes, and images
	docker compose down -v --rmi local

setup-dev: ## Set up development environment
	bash scripts/setup-dev.sh
