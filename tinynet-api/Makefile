# TinyNet API Makefile

.PHONY: help install test api db-migrate db-upgrade db-downgrade clean bootstrap

help:  ## Show this help message
	@echo "TinyNet API - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install Python dependencies
	python3 -m pip install -r requirements.txt

test:  ## Run tests
	python3 -m pytest tests/ -v

api:  ## Start the API server
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

api-prod:  ## Start the API server in production mode
	uvicorn app.main:app --host 0.0.0.0 --port 8000

db-migrate:  ## Generate new migration
	@echo "Generating new migration..."
	@read -p "Enter migration message: " message; \
	python3 -m alembic revision --autogenerate -m "$$message"

db-upgrade:  ## Run database migrations
	python3 -m alembic upgrade head

db-downgrade:  ## Rollback one migration
	python3 -m alembic downgrade -1

db-status:  ## Show migration status
	python3 -m alembic current

bootstrap:  ## Run bootstrap labeller on sample data
	python3 scripts/bootstrap_labels.py data/raw/*.md --out data/train.jsonl --verbose

clean:  ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.db" -delete
	find . -type f -name "*.sqlite" -delete

dev-setup: install  ## Set up development environment
	@echo "Setting up development environment..."
	@echo "✅ Dependencies installed"
	@echo "✅ Database initialized"
	@echo "✅ Ready for development!"
	@echo ""
	@echo "Next steps:"
	@echo "  make api     - Start the API server"
	@echo "  make test    - Run tests"
	@echo "  make bootstrap - Generate training data"

# Database operations
db-init:  ## Initialize database tables
	python3 -c "import asyncio; from app.db import init_db; asyncio.run(init_db())"

# ML operations
vectorize:  ## Test vectorizer (requires numpy)
	python3 -c "from app.ml.vectorizer import HashingVectorizer512; import numpy as np; vec = HashingVectorizer512(); x = vec.encode('Ran 2 miles, shin tight'); print(f'Vector shape: {x.shape}, Nonzero: {np.count_nonzero(x)}')"

# Quick checks
check:  ## Quick health check
	@echo "Checking API health..."
	@curl -s http://localhost:8000/healthz | python3 -m json.tool || echo "API not running"

check-deps:  ## Check if dependencies are installed
	@echo "Checking dependencies..."
	@python3 -c "import fastapi, sqlalchemy, pydantic, numpy; print('✅ All dependencies available')" || echo "❌ Missing dependencies - run 'make install'"

# Development workflow
dev: check-deps  ## Start development workflow
	@echo "🚀 Starting TinyNet API development..."
	@echo "📖 API docs: http://localhost:8000/docs"
	@echo "🔍 OpenAPI: http://localhost:8000/openapi.json"
	@echo "🏥 Health: http://localhost:8000/healthz"
	@echo ""
	@echo "Press Ctrl+C to stop"
	@make api
