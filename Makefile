# LLM Fine-Tuning Framework Makefile

.PHONY: help setup install clean train preprocess evaluate test lint format

# Default target
help:
	@echo "LLM Fine-Tuning Framework - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  setup          - Initialize project environment"
	@echo "  install        - Install dependencies"
	@echo "  clean          - Clean generated files"
	@echo ""
	@echo "Data:"
	@echo "  preprocess     - Preprocess sample data"
	@echo "  create-data    - Create sample data"
	@echo ""
	@echo "Training:"
	@echo "  train          - Train model with default config"
	@echo "  train-test     - Train with test config (smaller model)"
	@echo ""
	@echo "Evaluation:"
	@echo "  evaluate       - Evaluate trained model"
	@echo ""
	@echo "Development:"
	@echo "  lint           - Run code linting"
	@echo "  format         - Format code"
	@echo "  test           - Run tests"

# Setup
setup:
	@echo "Setting up LLM fine-tuning framework..."
	python scripts/setup.py

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

clean:
	@echo "Cleaning generated files..."
	rm -rf data/processed/*
	rm -rf experiments/outputs/*
	rm -rf logs/*
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/*/__pycache__
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Data
preprocess:
	@echo "Preprocessing data..."
	python scripts/preprocess.py --input-file data/raw/sample_data.jsonl

create-data:
	@echo "Creating sample data..."
	python -c "from src.data.preprocessor import create_sample_data; create_sample_data('data/raw/sample_data.jsonl')"

# Training
train:
	@echo "Training model with default configuration..."
	python scripts/train.py --config configs/default_config.yaml --create-sample-data

train-test:
	@echo "Training model with test configuration..."
	python scripts/train.py --config configs/test_config.yaml --create-sample-data

# Evaluation
evaluate:
	@echo "Evaluating model..."
	python -c "from src.utils.evaluation import evaluate_model; results = evaluate_model('mistralai/Mistral-7B-Instruct-v0.2', 'experiments/outputs'); print(results)"

# Development
lint:
	@echo "Running code linting..."
	flake8 src/ scripts/ --max-line-length=100 --ignore=E203,W503

format:
	@echo "Formatting code..."
	black src/ scripts/ --line-length=100
	isort src/ scripts/

test:
	@echo "Running tests..."
	python -m pytest tests/ -v

# Quick start
quick-start: setup create-data preprocess train

# Full pipeline
pipeline: setup create-data preprocess train evaluate