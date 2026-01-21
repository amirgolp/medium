# 👋 START HERE

Welcome to your **Trading System**! This is your entry point.

## 🎯 What Is This?

A complete automated trading system with:
- 🤖 Machine Learning predictions (CNN-LSTM)
- 📰 News sentiment analysis
- 💼 Risk management
- 📊 Backtesting with visualization
- 📈 5 professional chart types

## ⚡ Quick Start (60 seconds)

```bash
# 1. Install UV (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Setup (one command)
make venv && source .venv/bin/activate && make install-dev

# 3. Train a model
make train SYMBOL=EURUSD EPOCHS=100

# 4. Run backtest
make backtest SYMBOL=EURUSD DAYS=90

# 5. View results
open backtest_results/equity_curve.png
```

## 📚 Documentation Index

Choose your path:

### 🚀 I want to start immediately
→ [QUICKSTART.md](QUICKSTART.md) - 5-minute quick start

### 💻 I want to install the system
→ [INSTALL.md](INSTALL.md) - Detailed installation guide

### 📖 I want complete documentation
→ [README.md](README.md) - Full documentation

### 🎮 I want command reference
→ [COMMANDS_CHEATSHEET.md](COMMANDS_CHEATSHEET.md) - All commands

### 🔧 I want to use UV package manager
→ [UV_GUIDE.md](UV_GUIDE.md) - UV complete guide

### 🎯 I want to know all entry points
→ [ENTRY_POINTS.md](ENTRY_POINTS.md) - CLI, API, scripts

### 📊 I want project overview
→ [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Complete overview

## 🎯 Your Entry Points

### 1️⃣ **Makefile** (Easiest)
```bash
make train                # Train models
make backtest             # Run backtest with charts
make help                 # See all commands
```

### 2️⃣ **CLI** (Most flexible)
```bash
python -m trading_system train --symbols EURUSD --epochs 300
python -m trading_system backtest --symbol EURUSD --visualize
```

### 3️⃣ **Python API** (Most powerful)
```python
from trading_system import BacktestEngine, BacktestVisualizer
engine = BacktestEngine(initial_balance=10000)
result = engine.run_backtest(data, strategy_func, "EURUSD")
```

### 4️⃣ **Example Scripts** (Learning)
```bash
python trading_system/examples/run_backtest.py
```

## 📊 What You Get

After backtesting, you'll have **5 professional charts**:

```
backtest_results/
├── equity_curve.png          # 📈 Account balance over time
├── drawdown.png             # 📉 Risk exposure
├── trade_distribution.png   # 📊 P&L histogram
├── win_loss_analysis.png    # 🎯 Detailed breakdown
└── trade_timeline.png       # 🗓️ Trades on price chart
```

## 🗂️ Project Structure

```
trading_system/
├── ml_predictor/           # 🤖 ML models
├── sentiment_analyzer/     # 📰 News analysis
├── risk_management/        # 💼 Risk control
├── backtest/              # 📊 Backtesting
└── examples/              # 📚 Example scripts

Documentation/
├── README_FIRST.md        # 👈 You are here
├── QUICKSTART.md         # ⚡ 5-min start
├── INSTALL.md            # 💻 Installation
├── README.md             # 📖 Full docs
├── UV_GUIDE.md           # 🔧 UV guide
└── COMMANDS_CHEATSHEET.md # 🎮 Commands
```

## 🎓 Learning Path

### Beginner
1. [QUICKSTART.md](QUICKSTART.md) - Get running in 5 minutes
2. [COMMANDS_CHEATSHEET.md](COMMANDS_CHEATSHEET.md) - Learn basic commands
3. Run: `make backtest SYMBOL=EURUSD`

### Intermediate
1. [README.md](README.md) - Understand the system
2. [ENTRY_POINTS.md](ENTRY_POINTS.md) - Learn all interfaces
3. Study: `trading_system/examples/`

### Advanced
1. [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - System architecture
2. [UV_GUIDE.md](UV_GUIDE.md) - Dependency management
3. Customize: Write your own strategies

## 🔥 Most Common Commands

```bash
# Training
make train SYMBOL=EURUSD EPOCHS=100

# Backtesting
make backtest SYMBOL=EURUSD DAYS=90

# With different symbols
make backtest SYMBOL=GBPUSD DAYS=60

# Format code
make format

# Run tests
make test

# Clean up
make clean
```

## 🎯 Common Tasks

### Train a Model
```bash
make train SYMBOL=EURUSD EPOCHS=100
# or
python -m trading_system train --symbols EURUSD --epochs 100
```

### Run Backtest
```bash
make backtest SYMBOL=EURUSD DAYS=90
# or
python -m trading_system backtest --symbol EURUSD --days 90 --visualize
```

### View Results
```bash
open backtest_results/
# Charts are automatically saved as PNG files
```

## 💡 Pro Tips

1. **Start with QUICKSTART.md** - Get running immediately
2. **Use Makefile** - Simplest interface
3. **Check examples/** - Learn from working code
4. **Read COMMANDS_CHEATSHEET.md** - Quick reference

## 🆘 Need Help?

1. **Installation issues?** → [INSTALL.md](INSTALL.md)
2. **Don't understand commands?** → [COMMANDS_CHEATSHEET.md](COMMANDS_CHEATSHEET.md)
3. **Want examples?** → `trading_system/examples/`
4. **Need complete docs?** → [README.md](README.md)

## 🚀 Next Steps

1. Pick a documentation file above
2. Follow the installation guide
3. Run your first backtest
4. Start customizing

## 📞 Support

- 📖 Documentation: Multiple guides included
- 💻 Examples: See `trading_system/examples/`
- 🐛 Issues: Report on GitHub
- 💬 Questions: GitHub discussions

---

**Ready to start?**

→ Go to [QUICKSTART.md](QUICKSTART.md) for 5-minute setup  
→ Or run: `make venv && source .venv/bin/activate && make install-dev`

**Happy Trading!** 📈
