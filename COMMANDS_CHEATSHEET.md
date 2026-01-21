# Trading System Commands Cheatsheet

Quick reference for all commands and operations.

## 🚀 Installation

```bash
# Quick install (UV + Makefile)
make venv && source .venv/bin/activate && make install-dev

# Or step by step
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## 📋 Common Operations

### Using Makefile (Easiest)

```bash
# View all commands
make help

# Setup
make install-dev          # Install with dev tools
make install-all          # Install everything

# Training
make train                # Train EURUSD (default)
make train SYMBOL=GBPUSD EPOCHS=200

# Backtesting
make backtest                        # Backtest EURUSD 90 days
make backtest SYMBOL=GBPUSD DAYS=60  # Backtest GBPUSD 60 days

# Development
make format               # Format code with black
make lint                # Check code with ruff
make test                # Run tests
make clean               # Clean build artifacts
```

### Using CLI

```bash
# Train models
trading-system train --symbols EURUSD,GBPUSD --epochs 300

# Or
python -m trading_system train --symbols EURUSD --epochs 100

# Make predictions
trading-system predict --symbol EURUSD

# Analyze sentiment
trading-system sentiment --pair EURUSD --lookback-hours 24

# Run backtest
trading-system backtest --symbol EURUSD --days 90 --visualize --save-charts

# Specify output directory
trading-system backtest --symbol EURUSD --output-dir ./my_results --save-charts
```

### Using Python Scripts

```bash
# Train models
python trading_system/examples/train_models.py

# Make predictions
python trading_system/examples/predict_forex.py

# Analyze sentiment
python trading_system/examples/analyze_sentiment.py

# Run complete strategy
python trading_system/examples/complete_trading_strategy.py

# Run backtest
python trading_system/examples/run_backtest.py

# Run combined strategy backtest
python trading_system/examples/backtest_combined_strategy.py
```

### Using Python API

```python
# Training
from trading_system import ModelTrainer
trainer = ModelTrainer(history_size=120, training_days=120)
results = trainer.train_multiple_symbols(["EURUSD"], epochs=300)

# Prediction
from trading_system import ForexPredictor
predictor = ForexPredictor("./models/model.EURUSD.H1.120.onnx")
prediction, price, last = predictor.predict_symbol("EURUSD")

# Backtesting
from trading_system import BacktestEngine, BacktestVisualizer
engine = BacktestEngine(initial_balance=10000)
data = engine.load_data("EURUSD", "H1", days=90)
result = engine.run_backtest(data, strategy_func, "EURUSD")
viz = BacktestVisualizer()
viz.plot_all(result, data)
```

## 📊 Visualization

```bash
# Display charts interactively
trading-system backtest --symbol EURUSD --visualize

# Save charts to files
trading-system backtest --symbol EURUSD --save-charts

# Custom output directory
trading-system backtest --symbol EURUSD --save-charts --output-dir ./results
```

**Output files:**
- `equity_curve.png` - Account balance over time
- `drawdown.png` - Maximum drawdown chart
- `trade_distribution.png` - Histogram of profits/losses
- `win_loss_analysis.png` - Win/loss breakdown
- `trade_timeline.png` - Price chart with trades

## 🔧 Development Commands

### UV Package Manager

```bash
# Install package
uv pip install -e .
uv pip install -e ".[dev]"
uv pip install -e ".[all]"

# Install specific package
uv pip install pandas

# Create virtual environment
uv venv
uv venv --python 3.11

# Generate lock file
uv pip compile pyproject.toml -o uv.lock

# Sync from lock file
uv pip sync uv.lock
```

### Code Quality

```bash
# Format code
make format
# or
black trading_system/

# Lint code
make lint
# or
ruff check trading_system/

# Fix lint issues
make lint-fix
# or
ruff check trading_system/ --fix

# Type checking
mypy trading_system/
```

### Testing

```bash
# Run tests
make test
# or
pytest tests/

# Run with coverage
make test-cov
# or
pytest tests/ --cov=trading_system --cov-report=html

# Run specific test
pytest tests/test_predictor.py -v
```

### Cleanup

```bash
# Clean build artifacts
make clean

# Remove virtual environment
rm -rf .venv

# Remove all generated files
make clean && rm -rf models/ backtest_results/
```

## 📦 Package Management

### Install Dependencies

```bash
# Core dependencies
uv pip install -e .

# With development tools
uv pip install -e ".[dev]"

# With MetaTrader 5 (Windows)
uv pip install -e ".[mt5]"

# Everything
uv pip install -e ".[all]"
```

### Update Dependencies

```bash
# Update all packages
uv pip install --upgrade -e ".[dev]"

# Update specific package
uv pip install --upgrade pandas
```

### List Installed Packages

```bash
uv pip list
# or
pip list
```

## 🎯 Quick Workflows

### Complete Workflow (Training → Backtesting)

```bash
# Method 1: Makefile
make full-workflow

# Method 2: Manual
trading-system train --symbols EURUSD --epochs 100
trading-system backtest --symbol EURUSD --days 90 --visualize --save-charts
```

### Development Workflow

```bash
# 1. Setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Make changes
# ... edit code ...

# 3. Format and lint
make format
make lint-fix

# 4. Test
make test

# 5. Run
trading-system backtest --symbol EURUSD --visualize
```

### Research Workflow

```bash
# 1. Train multiple models
trading-system train --symbols EURUSD,GBPUSD,USDJPY --epochs 300

# 2. Backtest each
for symbol in EURUSD GBPUSD USDJPY; do
    trading-system backtest --symbol $symbol --days 90 --save-charts \
        --output-dir ./results_$symbol
done

# 3. Compare results
ls -la results_*/equity_curve.png
```

## 🔍 Debugging

### Verbose Output

```bash
# Add --verbose flag
trading-system train --symbols EURUSD --epochs 100 --verbose

# Or set log level
export LOG_LEVEL=DEBUG
trading-system backtest --symbol EURUSD
```

### Check Installation

```bash
# Verify package
python -c "import trading_system; print(trading_system.__version__)"

# Verify CLI
trading-system --help

# Check dependencies
uv pip list | grep trading-system
```

### Troubleshooting

```bash
# Reinstall package
uv pip install -e . --force-reinstall

# Clear cache
uv cache clean
pip cache purge

# Check Python version
python --version  # Should be 3.8+
```

## 📝 Configuration

### Environment Variables

```bash
# Create .env file
cat > .env << EOF
NEWSAPI_KEY=your_newsapi_key
FINNHUB_KEY=your_finnhub_key
LOG_LEVEL=INFO
EOF
```

### API Keys

Get free API keys:
- NewsAPI: https://newsapi.org
- Finnhub: https://finnhub.io

### Python Version

```bash
# Set preferred Python version
echo "3.11" > .python-version

# Create venv with specific version
uv venv --python 3.11
```

## 📊 Result Files

After backtesting, find results in:

```
./backtest_results/
├── equity_curve.png          # Account balance over time
├── drawdown.png             # Drawdown analysis
├── trade_distribution.png   # Trade histogram
├── win_loss_analysis.png    # Detailed breakdown
└── trade_timeline.png       # Trades on price chart
```

## 🚀 Production Deployment

```bash
# 1. Train production models
trading-system train --symbols EURUSD,GBPUSD --epochs 500

# 2. Validate with backtest
trading-system backtest --symbol EURUSD --days 180 --save-charts

# 3. Build package
python -m build

# 4. Install on production server
uv pip install dist/trading_system-1.0.0-py3-none-any.whl
```

## 📖 Documentation

- [README.md](README.md) - Full documentation
- [QUICKSTART.md](QUICKSTART.md) - 5-minute quick start
- [UV_GUIDE.md](UV_GUIDE.md) - UV package manager guide
- [INSTALL.md](INSTALL.md) - Detailed installation
- [ENTRY_POINTS.md](ENTRY_POINTS.md) - All entry points

## 💡 Pro Tips

```bash
# Use aliases for common commands
alias ts='trading-system'
alias tsb='trading-system backtest --visualize'
alias tst='trading-system train'

# Now you can use:
ts backtest --symbol EURUSD
tsb --symbol GBPUSD --days 60
tst --symbols EURUSD --epochs 100

# Combine commands
tst --symbols EURUSD --epochs 50 && tsb --symbol EURUSD
```

## 🎯 Most Used Commands

```bash
# 1. Quick setup
make venv && source .venv/bin/activate && make install-dev

# 2. Train model
make train SYMBOL=EURUSD EPOCHS=100

# 3. Run backtest
make backtest SYMBOL=EURUSD DAYS=90

# 4. View results
open backtest_results/equity_curve.png
```

That's it! Bookmark this for quick reference. 🔖
