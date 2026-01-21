# Trading System Makefile
# Simple commands for common operations

.PHONY: help install install-dev install-all clean test format lint run-backtest train docs

# Colors for output
BLUE=\033[0;34m
GREEN=\033[0;32m
RED=\033[0;31m
NC=\033[0m # No Color

help:
	@echo "$(BLUE)Trading System - Available Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  make install          - Install package with core dependencies"
	@echo "  make install-dev      - Install with development tools"
	@echo "  make install-all      - Install everything (dev + mt5)"
	@echo "  make venv            - Create virtual environment with UV"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  make format          - Format code with black"
	@echo "  make lint            - Lint code with ruff"
	@echo "  make test            - Run tests with pytest"
	@echo "  make clean           - Remove build artifacts"
	@echo ""
	@echo "$(GREEN)Trading Operations:$(NC)"
	@echo "  make train           - Train ML models"
	@echo "  make predict         - Make predictions"
	@echo "  make sentiment       - Analyze sentiment"
	@echo "  make backtest        - Run backtest with visualization"
	@echo ""
	@echo "$(GREEN)Examples:$(NC)"
	@echo "  make SYMBOL=GBPUSD train           - Train GBPUSD model"
	@echo "  make SYMBOL=GBPUSD DAYS=60 backtest - Backtest GBPUSD 60 days"

# Setup commands
venv:
	@echo "$(BLUE)Creating virtual environment with UV...$(NC)"
	uv venv
	@echo "$(GREEN)✓ Virtual environment created!$(NC)"
	@echo "Activate with: source .venv/bin/activate"

install:
	@echo "$(BLUE)Installing package...$(NC)"
	uv pip install -e .
	@echo "$(GREEN)✓ Installation complete!$(NC)"

install-dev:
	@echo "$(BLUE)Installing with dev dependencies...$(NC)"
	uv pip install -e ".[dev]"
	@echo "$(GREEN)✓ Installation complete!$(NC)"

install-all:
	@echo "$(BLUE)Installing all dependencies...$(NC)"
	uv pip install -e ".[all]"
	@echo "$(GREEN)✓ Installation complete!$(NC)"

# Development commands
format:
	@echo "$(BLUE)Formatting code with black...$(NC)"
	black trading_system/
	@echo "$(GREEN)✓ Code formatted!$(NC)"

lint:
	@echo "$(BLUE)Linting code with ruff...$(NC)"
	ruff check trading_system/
	@echo "$(GREEN)✓ Linting complete!$(NC)"

lint-fix:
	@echo "$(BLUE)Fixing lint issues...$(NC)"
	ruff check trading_system/ --fix
	@echo "$(GREEN)✓ Fixes applied!$(NC)"

test:
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v
	@echo "$(GREEN)✓ Tests complete!$(NC)"

test-cov:
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest tests/ -v --cov=trading_system --cov-report=html
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

clean:
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .coverage htmlcov/
	rm -rf **/__pycache__
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Cleaned!$(NC)"

# Trading commands with default values
SYMBOL ?= EURUSD
EPOCHS ?= 100
DAYS ?= 90
BALANCE ?= 10000

train:
	@echo "$(BLUE)Training models for $(SYMBOL)...$(NC)"
	python -m trading_system train --symbols $(SYMBOL) --epochs $(EPOCHS)
	@echo "$(GREEN)✓ Training complete!$(NC)"

predict:
	@echo "$(BLUE)Making prediction for $(SYMBOL)...$(NC)"
	python -m trading_system predict --symbol $(SYMBOL)

sentiment:
	@echo "$(BLUE)Analyzing sentiment for $(SYMBOL)...$(NC)"
	python -m trading_system sentiment --pair $(SYMBOL)

backtest:
	@echo "$(BLUE)Running backtest for $(SYMBOL)...$(NC)"
	python -m trading_system backtest --symbol $(SYMBOL) --days $(DAYS) \
		--balance $(BALANCE) --visualize --save-charts
	@echo "$(GREEN)✓ Backtest complete! Results in ./backtest_results/$(NC)"

# Run all examples
examples:
	@echo "$(BLUE)Running all examples...$(NC)"
	@echo "\n=== Training Models ===\n"
	python trading_system/examples/train_models.py
	@echo "\n=== Making Predictions ===\n"
	python trading_system/examples/predict_forex.py
	@echo "\n=== Running Backtest ===\n"
	python trading_system/examples/run_backtest.py
	@echo "$(GREEN)✓ All examples complete!$(NC)"

# Quick start
quick-start: venv install-dev train backtest
	@echo "$(GREEN)✓ Quick start complete!$(NC)"
	@echo "Check ./backtest_results/ for visualizations"

# Build distribution
build:
	@echo "$(BLUE)Building distribution...$(NC)"
	python -m build
	@echo "$(GREEN)✓ Build complete!$(NC)"

# Install from distribution
install-dist:
	uv pip install dist/*.whl

# Full workflow
full-workflow: clean install-dev train backtest
	@echo "$(GREEN)✓ Full workflow complete!$(NC)"
