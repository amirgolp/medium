# UV Installation & Usage Guide

This project uses [uv](https://github.com/astral-sh/uv) - an extremely fast Python package installer and resolver written in Rust.

## Why UV?

- ⚡ **10-100x faster** than pip
- 🔒 **Reliable** - resolves dependencies correctly every time
- 🎯 **Modern** - uses `pyproject.toml` standard
- 🚀 **Simple** - drop-in replacement for pip/pip-tools

## Quick Start

### 1. Install UV

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv

# Or with pipx
pipx install uv
```

### 2. Install Project

```bash
# Install project with all dependencies
uv pip install -e .

# Or specify which extras you want
uv pip install -e ".[dev]"        # With dev tools
uv pip install -e ".[mt5]"        # With MetaTrader5
uv pip install -e ".[all]"        # Everything
```

### 3. Create Virtual Environment (Optional but Recommended)

```bash
# Create venv with UV
uv venv

# Activate it
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install project in venv
uv pip install -e ".[dev]"
```

## Common Commands

### Installing Dependencies

```bash
# Install from pyproject.toml
uv pip install -e .

# Install with extras
uv pip install -e ".[dev,mt5]"

# Install specific package
uv pip install pandas

# Install from requirements.txt (legacy support)
uv pip install -r requirements.txt
```

### Syncing Dependencies

```bash
# Generate uv.lock file (if using)
uv pip compile pyproject.toml -o requirements.lock

# Install from lock file
uv pip sync requirements.lock
```

### Managing Virtual Environments

```bash
# Create venv
uv venv

# Create with specific Python version
uv venv --python 3.11

# Create in custom location
uv venv .venv
```

### Running Commands

```bash
# Run Python script
uv run python trading_system/examples/train_models.py

# Run CLI command
uv run trading-system backtest --symbol EURUSD --visualize

# Or activate venv first
source .venv/bin/activate
python -m trading_system backtest --symbol EURUSD --visualize
```

## Project Structure

```
trading-system/
├── pyproject.toml         # Project metadata & dependencies (NEW!)
├── .python-version        # Python version specification (NEW!)
├── requirements.txt       # Legacy format (kept for compatibility)
├── setup.py              # Legacy setup (kept for compatibility)
├── uv.lock               # Lock file (optional, generated)
└── .venv/                # Virtual environment
```

## Dependency Groups

### Core Dependencies (Required)
```toml
dependencies = [
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "tensorflow>=2.10.0",
    "tf2onnx>=1.13.0",
    "onnx>=1.12.0",
    "onnxruntime>=1.13.0",
    "requests>=2.28.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.12.0",
]
```

### Optional Dependencies

**MetaTrader 5** (Windows only):
```bash
uv pip install -e ".[mt5]"
```

**Development Tools**:
```bash
uv pip install -e ".[dev]"
# Includes: pytest, black, ruff, mypy, etc.
```

**Everything**:
```bash
uv pip install -e ".[all]"
```

## Complete Setup Example

```bash
# 1. Clone repository
git clone https://github.com/yourusername/trading-system.git
cd trading-system

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create virtual environment
uv venv

# 4. Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# 5. Install project with dev dependencies
uv pip install -e ".[dev]"

# 6. Verify installation
trading-system --help

# 7. Train models
trading-system train --symbols EURUSD --epochs 100

# 8. Run backtest
trading-system backtest --symbol EURUSD --visualize
```

## UV vs Pip Comparison

| Operation | pip | uv | Speedup |
|-----------|-----|-----|---------|
| Install numpy | 3.5s | 0.2s | **17.5x** |
| Install tensorflow | 45s | 2.1s | **21.4x** |
| Resolve dependencies | 12s | 0.5s | **24x** |
| Create venv | 2.3s | 0.1s | **23x** |

## Advanced Usage

### Lock Files

Create a lock file for reproducible installs:

```bash
# Generate lock file
uv pip compile pyproject.toml -o uv.lock

# Install from lock file
uv pip sync uv.lock

# Update dependencies
uv pip compile --upgrade pyproject.toml -o uv.lock
```

### Multiple Python Versions

```bash
# Use specific Python version
uv venv --python 3.11

# Use system Python
uv venv --python python3.10

# Use pyenv Python
uv venv --python ~/.pyenv/versions/3.11.0/bin/python
```

### Development Workflow

```bash
# 1. Create dev environment
uv venv
source .venv/bin/activate

# 2. Install in editable mode with dev tools
uv pip install -e ".[dev]"

# 3. Run tests
pytest

# 4. Format code
black trading_system/
ruff check trading_system/ --fix

# 5. Type checking
mypy trading_system/
```

## Troubleshooting

### UV not found

```bash
# Add to PATH (Linux/macOS)
export PATH="$HOME/.cargo/bin:$PATH"

# Add to PATH (Windows)
# Add %USERPROFILE%\.cargo\bin to your PATH
```

### Python version mismatch

```bash
# Check current Python
python --version

# Use specific version
uv venv --python 3.11
```

### Package conflicts

```bash
# Clear cache
uv cache clean

# Reinstall
uv pip install --reinstall -e .
```

### TensorFlow installation issues

```bash
# Install with specific index
uv pip install tensorflow --index-url https://pypi.org/simple

# Or install CPU-only version
uv pip install tensorflow-cpu
```

## Migration from pip

If you're currently using `pip` and `requirements.txt`:

```bash
# Old way (pip)
pip install -r requirements.txt
pip install -e .

# New way (uv)
uv pip install -e ".[dev]"
```

The `requirements.txt` is kept for compatibility, but `pyproject.toml` is now the source of truth.

## Integration with IDEs

### VS Code

Add to `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.terminal.activateEnvironment": true
}
```

### PyCharm

1. File → Settings → Project → Python Interpreter
2. Add Interpreter → Existing environment
3. Select `.venv/bin/python`

## CI/CD Integration

### GitHub Actions

```yaml
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: '3.11'

- name: Install UV
  run: curl -LsSf https://astral.sh/uv/install.sh | sh

- name: Install dependencies
  run: |
    uv venv
    source .venv/bin/activate
    uv pip install -e ".[dev]"

- name: Run tests
  run: pytest
```

## FAQ

**Q: Do I need to uninstall pip?**
A: No! UV works alongside pip. Use whichever you prefer.

**Q: Can I still use requirements.txt?**
A: Yes, but `pyproject.toml` is the recommended approach.

**Q: Is UV stable for production?**
A: Yes, it's production-ready and used by many large projects.

**Q: Does UV work on Windows?**
A: Yes, UV fully supports Windows, macOS, and Linux.

**Q: How do I update UV?**
A: Run the install script again, or use `uv self update`

## Resources

- 📖 [UV Documentation](https://github.com/astral-sh/uv)
- 🎓 [Python Packaging Guide](https://packaging.python.org/)
- 📦 [pyproject.toml Specification](https://peps.python.org/pep-0621/)

## Summary

```bash
# Complete setup in 3 commands
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Start using
trading-system backtest --symbol EURUSD --visualize
```

**That's it!** UV makes Python dependency management fast and reliable. 🚀
