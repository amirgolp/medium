# Quick Start Guide

Get started with the Trading System in 5 minutes!

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install the package
pip install -e .
```

## Entry Points

The trading system provides **multiple entry points**:

### 1️⃣ Command-Line Interface (CLI)

The main entry point for all operations:

```bash
# View available commands
python -m trading_system.cli --help

# Train models
python -m trading_system.cli train --symbols EURUSD,GBPUSD --epochs 300

# Make predictions
python -m trading_system.cli predict --symbol EURUSD

# Analyze sentiment
python -m trading_system.cli sentiment --pair EURUSD

# Run backtest
python -m trading_system.cli backtest --symbol EURUSD --days 90 --visualize
```

### 2️⃣ Python Scripts

Run individual example scripts:

```bash
# Train ML models
python trading_system/examples/train_models.py

# Make predictions
python trading_system/examples/predict_forex.py

# Analyze sentiment
python trading_system/examples/analyze_sentiment.py

# Complete strategy
python trading_system/examples/complete_trading_strategy.py

# Run backtest
python trading_system/examples/run_backtest.py

# Backtest combined strategy
python trading_system/examples/backtest_combined_strategy.py
```

### 3️⃣ Python API

Import and use in your own code:

```python
from trading_system import (
    ModelTrainer,
    ForexPredictor,
    SentimentAnalyzer,
    BacktestEngine,
    BacktestVisualizer
)

# Your custom trading logic here
```

## Quick Workflow

### Step 1: Train Models

```bash
python -m trading_system.cli train --symbols EURUSD --epochs 100
```

This creates `./models/model.EURUSD.H1.120.onnx`

### Step 2: Run Backtest

```bash
python -m trading_system.cli backtest --symbol EURUSD --days 90 --visualize --save-charts
```

This creates:
- `./backtest_results/equity_curve.png`
- `./backtest_results/drawdown.png`
- `./backtest_results/trade_distribution.png`
- `./backtest_results/win_loss_analysis.png`
- `./backtest_results/trade_timeline.png`

### Step 3: Visualize Results

Charts are automatically displayed and saved to `./backtest_results/`

Open the PNG files to view:
- 📈 **Equity Curve**: Account balance over time
- 📉 **Drawdown**: Maximum loss from peak
- 📊 **Trade Distribution**: Histogram of profits/losses
- 🎯 **Win/Loss Analysis**: Pie charts, cumulative P&L, exit reasons
- 🗓️ **Trade Timeline**: Price chart with entry/exit markers

## Example: Complete Backtest

```python
from trading_system.backtest import BacktestEngine, BacktestVisualizer
from trading_system import ForexPredictor
from datetime import datetime, timedelta

# Load data
engine = BacktestEngine(initial_balance=10000)
data = engine.load_data("EURUSD", "H1", days=90)

# Load model
predictor = ForexPredictor("./models/model.EURUSD.H1.120.onnx")
predictor.min_price = data['close'].min()
predictor.max_price = data['close'].max()

# Define strategy
def ml_strategy(data, i, predictor, symbol):
    if i < 120:
        return None

    prices = data.iloc[i-120:i]['close'].values
    prediction, _, _ = predictor.predict(prices)
    signal = predictor.get_signal_name(prediction)

    if signal == "NEUTRAL":
        return None

    return {
        'signal': signal,
        'stop_loss': data.iloc[i]['close'] * (0.995 if signal == "BUY" else 1.005),
        'take_profit': data.iloc[i]['close'] * (1.01 if signal == "BUY" else 0.99),
        'lot_size': 0.1
    }

# Run backtest
result = engine.run_backtest(data, ml_strategy, "EURUSD", predictor=predictor)

# Print results
result.print_summary()

# Visualize
visualizer = BacktestVisualizer()
visualizer.plot_equity_curve(result)
visualizer.plot_drawdown(result)
visualizer.plot_win_loss_analysis(result)
visualizer.plot_trade_timeline(result, data)
```

## Understanding Backtest Results

### Key Metrics Explained

**Win Rate**: Percentage of profitable trades
- Target: >50% is good, >60% is excellent

**Profit Factor**: Total profit / Total loss
- Target: >1.5 is good, >2.0 is excellent

**Max Drawdown**: Largest peak-to-trough decline
- Target: <20% is good, <10% is excellent

**Sharpe Ratio**: Risk-adjusted returns
- Target: >1.0 is good, >2.0 is excellent

### Charts Explained

1. **Equity Curve**: Shows account growth over time
   - Upward slope = profitable strategy
   - Smooth curve = consistent performance
   - Sharp drops = large losses

2. **Drawdown Chart**: Shows risk exposure
   - Red area = underwater periods
   - Deeper drawdowns = higher risk
   - Longer drawdowns = slower recovery

3. **Trade Distribution**: Shows profit/loss spread
   - Centered on positive = profitable bias
   - Wide spread = high volatility
   - Outliers = exceptional wins/losses

4. **Win/Loss Analysis**: Detailed performance breakdown
   - Pie chart: Win/loss ratio
   - Cumulative P&L: Progressive performance
   - Exit reasons: How trades closed (TP, SL, Signal)
   - Monthly returns: Time-based performance

5. **Trade Timeline**: Visual trade history
   - Green ^ = Buy entry
   - Red v = Sell entry
   - X markers = Exits
   - Lines connect entry to exit

## Next Steps

1. **Optimize Parameters**: Experiment with different ATR multipliers, risk percentages
2. **Add Features**: Incorporate sentiment analysis, multiple timeframes
3. **Live Trading**: Deploy to MetaTrader 5 for real-time trading
4. **Paper Trading**: Test with demo account first

## CLI Command Reference

```bash
# Train models
trading-system train --symbols EURUSD,GBPUSD --epochs 300 --batch-size 32

# Make predictions
trading-system predict --symbol EURUSD --model-dir ./models

# Analyze sentiment
trading-system sentiment --pair EURUSD --lookback-hours 24

# Run backtest
trading-system backtest --symbol EURUSD --days 90 --balance 10000 --visualize --save-charts
```

## Environment Variables

Create a `.env` file:

```bash
NEWSAPI_KEY=your_newsapi_key_here
FINNHUB_KEY=your_finnhub_key_here
```

Get free API keys:
- NewsAPI: https://newsapi.org
- Finnhub: https://finnhub.io

## Troubleshooting

**No module named 'MetaTrader5'**
- Expected if not on Windows or MT5 not installed
- System will use mock data for development

**Matplotlib not available**
- Install with: `pip install matplotlib seaborn`

**Model not found**
- Run training first: `python trading_system/examples/train_models.py`

**No data retrieved**
- Check MT5 connection
- Verify symbol name (EURUSD not EUR/USD)
- System will use mock data as fallback

## Support

- Documentation: [README.md](README.md)
- Examples: `trading_system/examples/`
- Issues: Report bugs on GitHub

---

**Happy Trading! 📈**
