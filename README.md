# Trading System

A complete automated trading system with ML predictions, sentiment analysis, risk management, and backtesting.

## 🚀 Quick Start

```bash
# 1. Install UV (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Setup
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# 3. See all commands
poe

# 4. Train a model
poe train --symbol EURUSD --epochs 100

# 5. Run backtest with visualization
poe backtest --symbol EURUSD --days 90

# 6. View results
open backtest_results/equity_curve.png
```

## 📋 Common Commands

```bash
# List all tasks
poe

# Setup
poe install-dev              # Install with dev tools

# Training
poe train                    # Train EURUSD (default)
poe train --symbol GBPUSD --epochs 200

# Backtesting
poe backtest                 # Backtest EURUSD 90 days (default)
poe backtest --symbol GBPUSD --days 60 --balance 20000

# Development
poe format                   # Format code
poe lint-fix                 # Fix lint issues
poe test                     # Run tests
poe check                    # Run all checks (format+lint+test)
poe clean                    # Clean build artifacts

# Workflows
poe quick-start              # Install + train + backtest
poe full-workflow            # Clean + install + train + backtest

# Examples
poe example-backtest         # Run backtest example
poe examples                 # Run all examples
```

## 🎯 What You Get

### 1. ML-Powered Predictions
- CNN-LSTM neural network for forex predictions
- Weekly retraining with fresh data
- ONNX model export for production

### 2. Sentiment Analysis
- Multi-source news aggregation (NewsAPI, Finnhub, MT5 Calendar)
- Keyword-based sentiment scoring
- Currency-specific signal generation

### 3. Risk Management
- ATR-based dynamic stops
- Multiple position sizing modes (fixed, risk%, balance%)
- Portfolio risk controls

### 4. Backtesting & Visualization
- Historical data simulation
- **5 professional charts automatically generated:**
  - `equity_curve.png` - Account balance over time
  - `drawdown.png` - Risk exposure
  - `trade_distribution.png` - P&L histogram
  - `win_loss_analysis.png` - Detailed breakdown
  - `trade_timeline.png` - Trades on price chart

## 📊 Usage Examples

### Train Models

```python
from trading_system import ModelTrainer

trainer = ModelTrainer(history_size=120, training_days=120)
results = trainer.train_multiple_symbols(["EURUSD"], epochs=300)
```

Or use the CLI:
```bash
poe train --symbol EURUSD --epochs 100
```

### Make Predictions

```python
from trading_system import ForexPredictor

predictor = ForexPredictor("./models/model.EURUSD.H1.120.onnx")
prediction, price, last = predictor.predict_symbol("EURUSD")
signal = predictor.get_signal_name(prediction)
```

Or use the CLI:
```bash
poe predict --symbol EURUSD
```

### Run Backtest

```python
from trading_system import BacktestEngine, BacktestVisualizer

# Run backtest
engine = BacktestEngine(initial_balance=10000)
data = engine.load_data("EURUSD", "H1", days=90)
result = engine.run_backtest(data, strategy_func, "EURUSD")

# Visualize
viz = BacktestVisualizer()
viz.plot_equity_curve(result)
viz.plot_drawdown(result)
viz.create_full_report(result, data)
```

Or use the CLI:
```bash
poe backtest --symbol EURUSD --days 90
```

### Analyze Sentiment

```python
from trading_system import SentimentAnalyzer
from trading_system.sentiment_analyzer import NewsAPIClient

analyzer = SentimentAnalyzer(lookback_hours=24, min_sentiment_score=10.0)

# Fetch and analyze news
news_client = NewsAPIClient("your_api_key")
articles = news_client.fetch_news()
sentiments = news_client.parse_to_sentiment(articles, analyzer)

for s in sentiments:
    analyzer.add_sentiment(s)

# Generate signal
signal, confidence, details = analyzer.generate_signal("EUR", "USD")
```

Or use the CLI:
```bash
poe sentiment --pair EURUSD
```

## 🎨 Complete Example

```python
from trading_system import (
    ForexPredictor,
    SentimentAnalyzer,
    BacktestEngine,
    BacktestVisualizer,
    ATRCalculator,
    PositionSizer
)
from trading_system.risk_management import LotMode

# Load ML predictor
predictor = ForexPredictor("./models/model.EURUSD.H1.120.onnx")

# Setup sentiment analyzer
analyzer = SentimentAnalyzer(lookback_hours=24)

# Risk management
atr_calc = ATRCalculator(period=14)
sizer = PositionSizer(mode=LotMode.RISK_BASED, risk_percent=3.0)

# Run backtest
engine = BacktestEngine(initial_balance=10000)
data = engine.load_data("EURUSD", "H1", days=90)

def strategy(data, i, predictor, atr_calc):
    if i < 120:
        return None

    # Get ML signal
    prices = data.iloc[i-120:i]['close'].values
    prediction, _, _ = predictor.predict(prices)
    signal = predictor.get_signal_name(prediction)

    if signal == "NEUTRAL":
        return None

    # Calculate stops
    highs = data.iloc[i-14:i]['high'].values
    lows = data.iloc[i-14:i]['low'].values
    closes = data.iloc[i-14:i]['close'].values
    atr = atr_calc.calculate_atr_from_data(highs, lows, closes)

    current = data.iloc[i]['close']
    if signal == "BUY":
        sl = current - (atr * 2.5)
        tp = current + (atr * 3.0)
    else:
        sl = current + (atr * 2.5)
        tp = current - (atr * 3.0)

    return {
        'signal': signal,
        'stop_loss': sl,
        'take_profit': tp,
        'lot_size': 0.1
    }

result = engine.run_backtest(data, strategy, "EURUSD",
                              predictor=predictor, atr_calc=atr_calc)

# Print results
result.print_summary()

# Visualize
viz = BacktestVisualizer()
viz.create_full_report(result, data, output_dir="./backtest_results")
```

## 📦 Installation Options

### Install Core
```bash
uv pip install -e .
```

### Install with Dev Tools
```bash
uv pip install -e ".[dev]"
```

### Install Everything (including MT5)
```bash
uv pip install -e ".[all]"
```

## 🔧 Project Structure

```
trading_system/
├── ml_predictor/           # ML models (CNN-LSTM)
│   ├── trainer.py         # Model training
│   └── predictor.py       # Predictions
├── sentiment_analyzer/     # News sentiment analysis
│   ├── analyzer.py        # Core logic
│   └── news_sources.py    # API clients
├── risk_management/        # Risk controls
│   ├── position_sizer.py  # Position sizing
│   ├── risk_manager.py    # Portfolio risk
│   └── atr_calculator.py  # ATR calculations
├── backtest/              # Backtesting
│   ├── engine.py         # Simulation engine
│   └── visualizer.py     # Chart generation
└── examples/              # Example scripts
    ├── train_models.py
    ├── run_backtest.py
    └── ...
```

## ⚙️ Configuration

### API Keys (Optional)

Create `.env` file:
```bash
NEWSAPI_KEY=your_newsapi_key
FINNHUB_KEY=your_finnhub_key
```

Get free keys:
- NewsAPI: https://newsapi.org
- Finnhub: https://finnhub.io

### Current Parameters (Optimized for Jan 2026)

```python
# ML Model
history_size = 120          # Hours of history
training_days = 120         # Days of training data
early_stop_patience = 20    # Early stopping

# Risk Management
risk_percent = 3.0          # Risk per trade
atr_period = 14             # ATR period
sl_multiplier = 2.5         # SL = ATR * 2.5
tp_multiplier = 3.0         # TP = ATR * 3.0
trailing_stop = 1.0         # Trailing = ATR * 1.0

# Trading Hours
trade_start = 11            # 11:00 GMT
trade_end = 22              # 22:00 GMT
```

## 🛠️ Development

### Code Quality
```bash
poe format              # Format with black
poe lint-fix            # Fix with ruff
poe type-check          # Check with mypy
poe test                # Run pytest
poe check               # All of the above
```

### Running Examples
```bash
python trading_system/examples/train_models.py
python trading_system/examples/run_backtest.py
python trading_system/examples/backtest_combined_strategy.py
```

## 🎯 Entry Points

### 1. Poe Tasks (Recommended)
```bash
poe train
poe backtest
poe check
```

### 2. CLI
```bash
python -m trading_system train --symbols EURUSD --epochs 100
python -m trading_system backtest --symbol EURUSD --visualize
```

### 3. Python API
```python
from trading_system import ModelTrainer, BacktestEngine
```

### 4. Example Scripts
```bash
python trading_system/examples/run_backtest.py
```

## 📊 Backtest Metrics

Your backtests calculate:
- **Performance**: Win rate, profit factor, net profit
- **Risk**: Max drawdown, Sharpe ratio
- **Trade Stats**: Average profit/loss, duration
- **Exit Analysis**: TP, SL, signal-based exits

## 💡 Pro Tips

### Create Aliases
```bash
alias p='poe'
alias pt='poe train'
alias pb='poe backtest'

# Usage
p                    # List tasks
pt --symbol GBPUSD   # Train
pb --days 60         # Backtest
```

### Quick Workflows
```bash
# One-command setup + train + backtest
poe quick-start

# Full clean workflow
poe full-workflow
```

### Multiple Symbols
```bash
# Train multiple (use CLI directly for comma-separated)
python -m trading_system train --symbols EURUSD,GBPUSD,USDJPY --epochs 300

# Or train individually
poe train --symbol EURUSD --epochs 100
poe train --symbol GBPUSD --epochs 100
```

## 🔍 Troubleshooting

### Installation Issues
```bash
# Clear cache
uv cache clean

# Reinstall
uv pip install -e ".[dev]" --force-reinstall
```

### MT5 Not Available
The system works without MetaTrader 5 - it uses mock data for development and testing.

### Matplotlib Issues
```bash
# Install backend
sudo apt-get install python3-tk  # Ubuntu/Debian
brew install python-tk            # macOS

# Or set backend
export MPLBACKEND=TkAgg
```

## 📚 All Available Tasks

Run `poe` to see full list. Main tasks:

**Setup:** `install`, `install-dev`, `install-all`
**Trading:** `train`, `predict`, `sentiment`, `backtest`
**Dev:** `format`, `lint`, `lint-fix`, `type-check`, `test`, `test-cov`, `clean`
**Examples:** `example-train`, `example-backtest`, `examples`
**Workflows:** `check`, `quick-start`, `full-workflow`
**Utility:** `info`, `version`, `docs`

## 📄 License

MIT License - Free for commercial and personal use.

## ⚠️ Disclaimer

This software is for educational purposes only. Trading carries risk. Past performance does not guarantee future results. Always test with paper trading first.

---

**Quick Start:** `uv venv && source .venv/bin/activate && poe install-dev && poe quick-start`

**Happy Trading!** 📈
