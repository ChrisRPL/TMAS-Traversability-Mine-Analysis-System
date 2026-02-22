.PHONY: help install install-dev test lint format clean

help:
	@echo "TMAS Development Commands"
	@echo "========================="
	@echo "install        Install package in editable mode"
	@echo "install-dev    Install with development dependencies"
	@echo "test           Run all tests"
	@echo "test-cov       Run tests with coverage"
	@echo "lint           Run linting (ruff + mypy)"
	@echo "format         Format code with black"
	@echo "clean          Remove build artifacts and cache"
	@echo "pre-commit     Run pre-commit hooks"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,tracking]"
	pre-commit install

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src/tmas --cov-report=html --cov-report=term

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/ scripts/
	ruff check --fix src/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

pre-commit:
	pre-commit run --all-files
