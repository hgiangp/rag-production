DOCKER_COMPOSE ?= docker-compose
ENV ?= development

# ─── Development ─────────────────────────────────────────────────────────────
.PHONY: install dev ui

install:
	pip install uv
	uv sync --all-extras

dev:
	@echo "Starting RAG API in development mode..."
	@bash -c "APP_ENV=development uv run uvicorn app.main:app --reload --port 8000 --loop uvloop"

ui:
	@echo "Starting Streamlit UI..."
	@bash -c "APP_ENV=$(ENV) uv run streamlit run ui/streamlit/app.py --server.port 8501"

# ─── Testing ─────────────────────────────────────────────────────────────────
.PHONY: test test-integration test-cov

test:
	@echo "Running unit tests..."
	uv run pytest tests/ -m "not integration" -v

test-integration:
	@echo "Running integration tests (requires Docker services)..."
	uv run pytest tests/ -m "integration" -v

test-cov:
	uv run pytest tests/ --cov=app --cov-report=html --cov-report=term-missing

# ─── Evaluation ──────────────────────────────────────────────────────────────
.PHONY: eval eval-batch eval-quick

eval:
	@echo "Running interactive evaluation..."
	@bash -c "APP_ENV=$(ENV) uv run python -m evals.run_eval --interactive"

eval-batch:
	@echo "Running batch evaluation against golden dataset..."
	@bash -c "APP_ENV=$(ENV) uv run python -m evals.run_eval --batch --dataset evals/golden_dataset.json"

eval-quick:
	@echo "Running quick evaluation (first 5 samples)..."
	@bash -c "APP_ENV=$(ENV) uv run python -m evals.run_eval --batch --dataset evals/golden_dataset.json --limit 5"

# ─── Code Quality ────────────────────────────────────────────────────────────
.PHONY: lint format check

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run isort .

check: lint
	uv run ruff format --check .

# ─── Docker ──────────────────────────────────────────────────────────────────
.PHONY: docker-up docker-down docker-logs docker-build

docker-up:
	@echo "Starting all services (ENV=$(ENV))..."
	@ENV_FILE=.env.$(ENV); \
	if [ ! -f $$ENV_FILE ]; then \
		echo "Missing $$ENV_FILE — copy from .env.example"; exit 1; \
	fi; \
	APP_ENV=$(ENV) $(DOCKER_COMPOSE) --env-file $$ENV_FILE up -d

docker-down:
	@APP_ENV=$(ENV) $(DOCKER_COMPOSE) down

docker-logs:
	@APP_ENV=$(ENV) $(DOCKER_COMPOSE) logs -f app

docker-build:
	@APP_ENV=$(ENV) $(DOCKER_COMPOSE) build app

docker-services:
	@echo "Starting only infra services (db, qdrant, prometheus, grafana)..."
	@ENV_FILE=.env.$(ENV); \
	APP_ENV=$(ENV) $(DOCKER_COMPOSE) --env-file $$ENV_FILE up -d db qdrant prometheus grafana

# ─── Database ────────────────────────────────────────────────────────────────
.PHONY: db-migrate seed-db

db-migrate:
	@echo "Running schema migrations..."
	@bash -c "PGPASSWORD=$$POSTGRES_PASSWORD psql -h $$POSTGRES_HOST -U $$POSTGRES_USER -d $$POSTGRES_DB -f schema.sql"

seed-db:
	@echo "Seeding database with test data..."
	@bash -c "APP_ENV=$(ENV) uv run python scripts/seed_db.py"

# ─── Sample Data ─────────────────────────────────────────────────────────────
.PHONY: ingest-sample

ingest-sample:
	@echo "Ingesting sample documents..."
	@bash -c "APP_ENV=$(ENV) uv run python scripts/ingest_sample.py"

# ─── Cleanup ─────────────────────────────────────────────────────────────────
.PHONY: clean

clean:
	rm -rf .venv __pycache__ .pytest_cache .ruff_cache htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# ─── Help ────────────────────────────────────────────────────────────────────
.PHONY: help

help:
	@echo ""
	@echo "Production RAG Chatbot — Make Targets"
	@echo "======================================"
	@echo "  install           Install all dependencies via uv"
	@echo "  dev               Start FastAPI dev server (hot reload)"
	@echo "  ui                Start Streamlit UI"
	@echo ""
	@echo "  test              Run unit tests"
	@echo "  test-integration  Run integration tests (Docker required)"
	@echo "  test-cov          Run tests with coverage report"
	@echo ""
	@echo "  eval              Interactive eval CLI"
	@echo "  eval-batch        Batch eval against golden_dataset.json"
	@echo "  eval-quick        Quick eval (first 5 samples)"
	@echo ""
	@echo "  lint              Ruff lint check"
	@echo "  format            Ruff format + isort"
	@echo ""
	@echo "  docker-up         Start full stack (ENV=development)"
	@echo "  docker-down       Stop all services"
	@echo "  docker-services   Start only infra (db, qdrant, prometheus, grafana)"
	@echo "  docker-logs       Tail app logs"
	@echo ""
	@echo "  db-migrate        Apply schema.sql"
	@echo "  seed-db           Seed test data"
	@echo "  ingest-sample     Ingest sample documents"
	@echo ""
	@echo "Override ENV: make docker-up ENV=staging"
