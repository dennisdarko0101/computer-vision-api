.PHONY: install dev lint format test test-unit test-integration test-cov docker run clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

docker-build:
	docker build -f docker/Dockerfile --target cpu -t cv-api:latest .

docker-build-gpu:
	docker build -f docker/Dockerfile --target gpu -t cv-api:gpu .

docker-run:
	docker compose -f docker/docker-compose.yml up -d

docker-stop:
	docker compose -f docker/docker-compose.yml down

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

benchmark:
	python -c "from src.evaluation.evaluator import CVEvaluator; print('Evaluator loaded')"

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
