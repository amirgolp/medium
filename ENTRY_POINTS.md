# Trading System Entry Points

This document explains all the ways you can use the trading system.

## 🎯 Main Entry Points

### 1. Command-Line Interface (Recommended)

The **primary entry point** for all operations:

```bash
# As a module
python -m trading_system <command> [options]

# Or after installation
trading-system <command> [options]
```

**Available commands:**
- `train` - Train ML models
- `predict` - Make price predictions
- `sentiment` - Analyze news sentiment
- `backtest` - Run historical backtests

**Examples:**
```bash
# Train models
python -m trading_system train --symbols EURUSD,GBPUSD --epochs 300

# Make predictions
python -m trading_system predict --symbol EURUSD

# Analyze sentiment
python -m trading_system sentiment --pair EURUSD

# Run backtest with visualization
python -m trading_system backtest --symbol EURUSD --days 90 --visualize --save-charts
```

### 2. Example Scripts

Located in `trading_system/examples/`:

```bash
# ML Model Training
python trading_system/examples/train_models.py

# Price Prediction
python trading_system/examples/predict_forex.py

# Sentiment Analysis
python trading_system/examples/analyze_sentiment.py

# Complete Trading Strategy
python trading_system/examples/complete_trading_strategy.py

# Backtesting
python trading_system/examples/run_backtest.py

# Combined Strategy Backtest
python trading_system/examples/backtest_combined_strategy.py
```

### 3. Python API

Import and use in your own code:

```python
from trading_system import (
    # ML Prediction
    ModelTrainer,
    ForexPredictor,
    
    # Sentiment Analysis
    SentimentAnalyzer,
    
    # Risk Management
    PositionSizer,
    RiskManager,
    ATRCalculator,
    
    # Backtesting
    BacktestEngine,
    BacktestVisualizer,
    Trade,
    BacktestResult
)
```

## 📊 Visualization Entry Points

### Display Charts Interactively

```python
from trading_system.backtest import BacktestEngine, BacktestVisualizer

# Run backtest
engine = BacktestEngine(initial_balance=10000)
data = engine.load_data("EURUSD", "H1", days=90)
result = engine.run_backtest(data, strategy_func, "EURUSD")

# Visualize
visualizer = BacktestVisualizer()
visualizer.plot_equity_curve(result)
visualizer.plot_drawdown(result)
visualizer.plot_trade_distribution(result)
visualizer.plot_win_loss_analysis(result)
visualizer.plot_trade_timeline(result, data)
```

### Save Charts to Files

```python
# Save individual charts
visualizer.plot_equity_curve(result, save_path="equity.png")
visualizer.plot_drawdown(result, save_path="drawdown.png")

# Or save full report
visualizer.create_full_report(result, data, output_dir="./results")
```

### Via CLI

```bash
# Display charts interactively
python -m trading_system backtest --symbol EURUSD --visualize

# Save charts to files
python -m trading_system backtest --symbol EURUSD --save-charts --output-dir ./my_results
```

## 🔧 Installation Entry Points

### Development Mode

```bash
# Install in editable mode
pip install -e .

# Now you can use:
python -m trading_system <command>
trading-system <command>
```

### Production Mode

```bash
# Install as package
pip install .

# Use the trading-system command
trading-system train --symbols EURUSD
trading-system backtest --symbol EURUSD --visualize
```

## 📁 File Structure

```
trading_system/
├── __init__.py          # Package imports
├── __main__.py          # Module entry point (python -m trading_system)
├── cli.py               # CLI entry point (trading-system command)
│
├── ml_predictor/        # ML prediction module
├── sentiment_analyzer/  # Sentiment analysis module
├── risk_management/     # Risk management module
├── backtest/            # Backtesting module
│   ├── engine.py        # Backtest engine
│   └── visualizer.py    # Chart visualization
│
└── examples/            # Example scripts
    ├── train_models.py
    ├── predict_forex.py
    ├── analyze_sentiment.py
    ├── complete_trading_strategy.py
    ├── run_backtest.py
    └── backtest_combined_strategy.py
```

## 🎨 Visualization Outputs

When running backtests with visualization, you get:

### 1. Equity Curve (`equity_curve.png`)
- Account balance over time
- Shows cumulative performance
- P&L displayed in corner

### 2. Drawdown Chart (`drawdown.png`)
- Shows risk exposure
- Maximum drawdown highlighted
- Red shaded area = underwater periods

### 3. Trade Distribution (`trade_distribution.png`)
- Histogram of profits/losses
- Box plot showing quartiles
- Visual representation of trade outcomes

### 4. Win/Loss Analysis (`win_loss_analysis.png`)
- Win/loss ratio pie chart
- Cumulative P&L by trade number
- Exit reason distribution
- Monthly returns bar chart

### 5. Trade Timeline (`trade_timeline.png`)
- Price chart with all trades
- Green ^ = Buy entries
- Red v = Sell entries
- X markers = Exits
- Lines connecting entry to exit

## 💡 Quick Start Workflow

### 1. Train Models
```bash
python -m trading_system train --symbols EURUSD --epochs 100
```

### 2. Run Backtest with Visualization
```bash
python -m trading_system backtest --symbol EURUSD --days 90 --visualize --save-charts
```

### 3. View Results
```bash
# Charts saved to ./backtest_results/
ls backtest_results/
# equity_curve.png
# drawdown.png
# trade_distribution.png
# win_loss_analysis.png
# trade_timeline.png
```

## 🐍 Python API Examples

### Simple Backtest

```python
from trading_system import BacktestEngine, BacktestVisualizer
from datetime import timedelta

# Initialize
engine = BacktestEngine(initial_balance=10000)

# Load data
data = engine.load_data("EURUSD", "H1", days=90)

# Define strategy
def my_strategy(data, i):
    # Your strategy logic
    return {'signal': 'BUY', 'stop_loss': 1.0950, 'take_profit': 1.1050, 'lot_size': 0.1}

# Run backtest
result = engine.run_backtest(data, my_strategy, "EURUSD")

# Show results
result.print_summary()

# Visualize
viz = BacktestVisualizer()
viz.plot_all(result, data)
```

### Access Trade Details

```python
# Get all trades
for trade in result.trades:
    print(f"{trade.direction} @ {trade.entry_price:.5f}")
    print(f"  P&L: ${trade.profit_loss:.2f}")
    print(f"  Duration: {trade.duration_hours:.1f} hours")
    print(f"  Exit: {trade.exit_reason}")

# Filter winning trades
winners = [t for t in result.trades if t.profit_loss > 0]

# Calculate custom metrics
avg_winner = sum(t.profit_loss for t in winners) / len(winners)
```

## 📈 MetaTrader 5 Integration

When MetaTrader 5 is available, the system uses real market data:

```python
from trading_system import ForexPredictor

# This will fetch real data from MT5
predictor = ForexPredictor("./models/model.EURUSD.H1.120.onnx")
predictor.update_min_max("EURUSD")  # Uses MT5 data
prediction, price, last = predictor.predict_symbol("EURUSD")  # Real-time prediction
```

Without MT5, it uses mock data for development and testing.

## 🎯 Summary

**Primary Entry Point:**
```bash
python -m trading_system <command>
```

**Quick Backtest:**
```bash
python -m trading_system backtest --symbol EURUSD --days 90 --visualize --save-charts
```

**View Results:**
```bash
open backtest_results/equity_curve.png
```

That's it! Start with these commands and explore from there.
