# Installation Guide

Complete installation instructions for the Trading System.

## Quick Install (Recommended)

### Using UV (Fastest ⚡)

```bash
# 1. Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone repository
git clone https://github.com/yourusername/trading-system.git
cd trading-system

# 3. Setup in one command
make venv
source .venv/bin/activate
make install-dev
```

That's it! Skip to [Verify Installation](#verify-installation)

## Detailed Installation

### Method 1: UV (Recommended)

UV is 10-100x faster than pip and handles dependencies more reliably.

```bash
# Step 1: Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Step 2: Clone repository
git clone https://github.com/yourusername/trading-system.git
cd trading-system

# Step 3: Create virtual environment
uv venv

# Step 4: Activate virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Step 5: Install package
uv pip install -e ".[dev]"
```

### Method 2: Makefile (Easy)

```bash
# Clone repository
git clone https://github.com/yourusername/trading-system.git
cd trading-system

# One-command setup
make venv
source .venv/bin/activate
make install-dev
```

### Method 3: Traditional pip

```bash
# Clone repository
git clone https://github.com/yourusername/trading-system.git
cd trading-system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Installation Options

### Core Installation (Minimal)

```bash
# With UV
uv pip install -e .

# With pip
pip install -e .
```

Includes: numpy, pandas, tensorflow, matplotlib, etc.

### With Development Tools

```bash
# With UV
uv pip install -e ".[dev]"

# With pip
pip install -e ".[dev]"

# With Makefile
make install-dev
```

Includes: pytest, black, ruff, mypy

### With MetaTrader 5 (Windows only)

```bash
# With UV
uv pip install -e ".[mt5]"

# With pip
pip install -e ".[mt5]"
```

### Everything

```bash
# With UV
uv pip install -e ".[all]"

# With pip
pip install -e ".[all]"

# With Makefile
make install-all
```

## Verify Installation

```bash
# Check if package is installed
python -c "import trading_system; print(trading_system.__version__)"

# Should output: 1.0.0

# Check CLI is available
trading-system --help

# Or
python -m trading_system --help
```

## Platform-Specific Notes

### Linux/macOS

```bash
# May need Python dev packages
sudo apt-get install python3-dev  # Ubuntu/Debian
brew install python               # macOS

# Activate venv
source .venv/bin/activate
```

### Windows

```bash
# Activate venv
.venv\Scripts\activate

# If you get execution policy errors
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### MetaTrader 5 (Windows only)

```bash
# Install MT5 package
uv pip install MetaTrader5

# Or
pip install MetaTrader5
```

## Troubleshooting

### UV not found

```bash
# Add UV to PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Verify
uv --version
```

### TensorFlow installation issues

```bash
# CPU-only version (lighter)
uv pip install tensorflow-cpu

# Or specify index
uv pip install tensorflow --index-url https://pypi.org/simple
```

### Matplotlib backend issues

```bash
# If charts don't display
export MPLBACKEND=TkAgg  # Linux/macOS

# Or install backend
sudo apt-get install python3-tk  # Ubuntu/Debian
```

### ImportError after installation

```bash
# Reinstall in editable mode
uv pip install -e . --force-reinstall

# Or clear cache
uv cache clean
pip cache purge
```

## Quick Start After Installation

```bash
# 1. Train models
make train SYMBOL=EURUSD EPOCHS=100

# Or
trading-system train --symbols EURUSD --epochs 100

# 2. Run backtest
make backtest SYMBOL=EURUSD DAYS=90

# Or
trading-system backtest --symbol EURUSD --days 90 --visualize

# 3. View results
ls backtest_results/
```

## Dependencies Overview

### Core Dependencies
- **numpy** - Numerical computing
- **pandas** - Data manipulation
- **scikit-learn** - ML utilities
- **tensorflow** - Deep learning
- **onnx/onnxruntime** - Model deployment
- **matplotlib/seaborn** - Visualization
- **requests** - API calls

### Optional Dependencies
- **MetaTrader5** - Live trading (Windows only)
- **pytest** - Testing
- **black/ruff** - Code formatting
- **mypy** - Type checking

## Environment Setup

### API Keys (Optional)

```bash
# Create .env file
cat > .env << EOF
NEWSAPI_KEY=your_newsapi_key_here
FINNHUB_KEY=your_finnhub_key_here
EOF
```

Get free keys:
- NewsAPI: https://newsapi.org
- Finnhub: https://finnhub.io

### Python Version

```bash
# Check Python version
python --version

# Should be 3.8 or higher
# Recommended: 3.11
```

## Upgrading

```bash
# Pull latest code
git pull

# Reinstall
uv pip install -e ".[dev]" --force-reinstall

# Or with pip
pip install -e ".[dev]" --force-reinstall
```

## Uninstallation

```bash
# Remove package
pip uninstall trading-system

# Remove virtual environment
rm -rf .venv

# Remove build artifacts
make clean
```

## Next Steps

After installation:

1. **Train models**: [QUICKSTART.md](QUICKSTART.md)
2. **Run backtests**: [UV_GUIDE.md](UV_GUIDE.md)
3. **Read docs**: [README.md](README.md)

## Support

Issues? Check:
- [UV_GUIDE.md](UV_GUIDE.md) - UV-specific help
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- GitHub Issues - Report bugs

---

**Ready to trade!** 🚀
