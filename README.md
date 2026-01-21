# Trading System

A complete automated trading system with ML predictions, sentiment analysis, risk management, and backtesting.

## 🚀 Quick Start

### Option 1: Web Dashboard (Easiest!)

```bash
# 1. Install UV (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Setup
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 3. Launch web dashboard
poe app

# Browser opens at http://localhost:8501
# Use the GUI for everything: training, signals, backtesting!
```

### Option 2: Command Line

```bash
# 1-2. Same setup as above

# 3. See all commands
poe

# 4. Train a model (each pair needs its own model)
poe train --symbol EURUSD --epochs 200  # Initial training
poe train --symbol GBPUSD --epochs 200  # Optional: train more pairs

# 5. Get live trading signal
poe signal --symbol EURUSD

# 6. Run backtest
poe backtest --symbol EURUSD --days 90

# 7. Compare strategies
poe compare --symbol EURUSD --days 90
```

### ⚠️ Important: Model Training

- **One model per currency pair** - EURUSD, GBPUSD, USDJPY each need separate models
- **Retrain weekly** - Run `poe validate` to check if models need updates
- **Initial training**: 200 epochs (~5-10 min per pair)
- **Regular retraining**: 100-150 epochs when performance drops

```bash
# Check if models need retraining
poe validate

# Retrain all major pairs
poe train-all
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

# Strategy Comparison
poe compare                  # Compare all strategies on EURUSD
poe compare --symbol GBPUSD --days 60

# Live Trading Signals
poe signal                   # Get signal for EURUSD (hybrid strategy)
poe signal --symbol XAUUSD --strategy ml  # Gold with ML strategy
poe monitor --symbols EURUSD,GBPUSD,XAUUSD  # Monitor continuously

# Web Dashboard (NEW!)
poe app                      # Launch interactive Streamlit dashboard

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
poe example-compare          # Run strategy comparison example
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

### 3. Pattern Recognition (Nobel Prize-Winning)
- HLC (Higher-Lower-Close) 8-pattern classification system
- Fibonacci retracement zone analysis
- Adaptive stop loss based on pattern strength
- Multi-factor validation (pattern + zone + R/R + volatility)

### 4. Multiple Strategy Framework
- **ML Strategy**: Optimized CNN-LSTM with trend filter
- **Pattern Strategy**: Pure HLC pattern recognition
- **Hybrid Strategy**: Combined ML + Pattern signals
- **Sentiment Strategy**: ML + News sentiment alignment
- Extensible base class for custom strategies

### 5. Risk Management
- ATR-based dynamic stops
- Multiple position sizing modes (fixed, risk%, balance%)
- Portfolio risk controls
- Trailing stops and adaptive risk based on confidence

### 6. Strategy Comparison & Backtesting
- Run multiple strategies on same data
- Side-by-side performance metrics
- **Professional visualizations:**
  - Equity curves comparison
  - Metrics comparison bar charts
  - Trade distribution histograms
  - Drawdown comparison
  - Win/Loss analysis
  - Summary dashboard
- Export results to charts automatically

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
├── pattern_analyzer/       # Nobel Prize pattern recognition (NEW!)
│   └── hlc_patterns.py    # HLC 8-pattern system
├── strategies/            # Strategy framework (NEW!)
│   ├── base_strategy.py   # Abstract base class
│   ├── ml_strategy.py     # ML-based strategy
│   ├── pattern_strategy.py # Pattern-based strategy
│   ├── hybrid_strategy.py  # Combined ML+Pattern
│   └── sentiment_strategy.py # ML+Sentiment
├── comparison/            # Strategy comparison tools (NEW!)
│   ├── comparator.py      # Compare multiple strategies
│   └── visualizer.py      # Comparison charts
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
    ├── compare_strategies.py (NEW!)
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

## 🎓 Model Training Guide

### Do I Need a Model Per Symbol?

**Yes!** Each currency pair needs its own model because they have unique characteristics:

```bash
# Train separate models for each pair
poe train --symbol EURUSD --epochs 150
poe train --symbol GBPUSD --epochs 150
poe train --symbol USDJPY --epochs 150
```

**Why?**
- Different volatility patterns
- Unique correlation structures
- Pair-specific time-of-day behaviors
- Distinct economic drivers (EUR/USD vs GBP/USD vs USD/JPY)

### Do I Need to Retrain Regularly?

**Yes, but strategically.** Here's a practical approach:

#### Recommended Retraining Schedule

| Schedule | When | Why |
|----------|------|-----|
| **Weekly** | Every Monday | Captures recent market regime changes (recommended for active trading) |
| **Monthly** | First of month | Good balance for stable markets |
| **Event-driven** | After major events | Central bank decisions, economic crises, policy changes |

```bash
# Weekly retraining (recommended)
poe train --symbol EURUSD --epochs 100

# Or validate first, then retrain only what needs it
poe validate  # Check all models
poe train --symbol EURUSD --epochs 150  # Retrain specific pair
```

#### When to Retrain (Signals)

Retrain if ANY of these are true:
- ✗ Sharpe ratio < 0.5 on recent data (30 days)
- ✗ Profit factor < 1.0 (losing money)
- ✗ Win rate < 40%
- ✗ Model is > 30 days old
- ✗ Major market regime change (bull → bear)

#### When NOT to Retrain

- ✓ Model performing well in recent backtests
- ✓ Market conditions stable
- ✓ Less than 1-2 weeks since last training

### Quick Commands

```bash
# Validate all models (check if retraining needed)
poe validate

# Train all major pairs at once
poe train-all

# Train specific pair
poe train --symbol EURUSD --epochs 150

# Quick validation: Run 30-day backtest
poe backtest --symbol EURUSD --days 30
```

### Training Best Practices

1. **Initial Training** (First time):
   ```bash
   # Use more epochs for initial training
   poe train --symbol EURUSD --epochs 200
   ```

2. **Regular Retraining**:
   ```bash
   # Use fewer epochs (model already has good weights)
   poe train --symbol EURUSD --epochs 100
   ```

3. **After Market Crisis**:
   ```bash
   # Retrain immediately with more epochs
   poe train --symbol EURUSD --epochs 200
   ```

4. **Automated Weekly Retraining** (crontab):
   ```bash
   # Add to crontab for automatic weekly retraining (Mondays at 2 AM)
   0 2 * * 1 cd /path/to/project && source .venv/bin/activate && poe train-all
   ```

### Model Validation Workflow

```bash
# 1. Check all models
poe validate

# Output example:
# EURUSD:
#   • Model age: 45 days
#   • Sharpe ratio: 0.3
#   ⚠️  NEEDS RETRAINING
#      - Low Sharpe ratio (0.30)
#      - Model is 45 days old (>30 days)

# 2. Retrain flagged models
poe train --symbol EURUSD --epochs 150

# 3. Verify improvement
poe backtest --symbol EURUSD --days 30
```

### Storage Requirements

Each model is ~1MB. For 10 pairs:
```
./models/
├── model.EURUSD.H1.120.onnx  (~887KB)
├── model.GBPUSD.H1.120.onnx  (~887KB)
├── model.USDJPY.H1.120.onnx  (~887KB)
...
Total: ~10MB for 10 pairs
```

### Pro Tips

1. **Keep Old Models**: Don't delete immediately - compare new vs old
   ```bash
   mv models/model.EURUSD.H1.120.onnx models/model.EURUSD.old.onnx
   poe train --symbol EURUSD --epochs 150
   # Now compare performance
   ```

2. **Version Control Models**: Use Git LFS for model versioning
   ```bash
   git lfs track "*.onnx"
   git add models/*.onnx
   git commit -m "Update EURUSD model - 2026-01-21"
   ```

3. **A/B Testing**: Compare old vs new model before deploying
   ```bash
   poe backtest --symbol EURUSD --days 90  # Test new model
   # Then swap in old model and test again
   ```

### Commodities Support

**Can this work for gold, silver, oil, wheat?**

**Yes**, but requires modifications. The current system is optimized for forex pairs. To support commodities:

#### What Needs Changing:

1. **Pip/Tick Calculations** ([engine.py:60-70](trading_system/backtest/engine.py))
   - Forex uses pips (0.0001)
   - Commodities use different tick sizes:
     - Gold (XAU): $0.01 per oz
     - Silver (XAG): $0.001 per oz
     - Oil (WTI/Brent): $0.01 per barrel
     - Wheat: $0.25 per bushel

2. **Contract Specifications**
   - Forex: Standard lots, mini lots
   - Commodities: Futures contracts (100 oz gold, 1000 barrels oil)
   - Different margin requirements

3. **Symbol Naming**
   - Forex: EURUSD, GBPUSD
   - Commodities: XAUUSD (gold), XAGUSD (silver), CL (crude oil)

#### Quick Adaptation:

```python
# In engine.py, modify close_trade():
def close_trade(self, exit_time, exit_price, reason):
    # ... existing code ...

    # Commodity-aware calculations
    if self.symbol.startswith('XAU'):  # Gold
        tick_size = 0.01
        tick_value = 1.0  # $1 per 0.01 move for 1 oz
        contract_size = 100  # 100 oz per contract
    elif self.symbol.startswith('XAG'):  # Silver
        tick_size = 0.001
        tick_value = 5.0
        contract_size = 5000
    elif self.symbol in ['CL', 'WTI', 'BRENT']:  # Crude oil
        tick_size = 0.01
        tick_value = 10.0
        contract_size = 1000
    else:  # Forex (existing logic)
        tick_size = 0.0001 if "JPY" not in self.symbol else 0.01
        tick_value = 10.0
        contract_size = 100000

    # Calculate P&L
    price_change = exit_price - entry_price if direction == "BUY" else entry_price - exit_price
    ticks = price_change / tick_size
    self.profit_loss = ticks * tick_value * self.lot_size
```

#### Training for Commodities:

Same process - one model per commodity:

```bash
# Train commodity models
poe train --symbol XAUUSD --epochs 200  # Gold
poe train --symbol XAGUSD --epochs 200  # Silver
poe train --symbol CL --epochs 200      # Crude oil

# Works the same way as forex
poe backtest --symbol XAUUSD --days 90
poe compare --symbol XAUUSD --days 90
```

**Key Differences:**
- ✅ ML architecture works the same (CNN-LSTM learns patterns)
- ✅ Pattern recognition works (HLC patterns universal)
- ✅ Risk management principles apply
- ⚠️ Need different contract size calculations
- ⚠️ Different volatility characteristics (gold more volatile than EUR)
- ⚠️ Consider storage costs, rollover for futures

### Advanced Model Improvements

#### 1. Cross-Pair Correlations (Multi-Asset Features)

**Your idea is excellent!** Correlated assets can improve predictions:

**Forex Correlations:**
- EUR/USD ↔ GBP/USD (often move together - both vs USD)
- EUR/USD ↔ USD/CHF (inverse - CHF is safe haven)
- Gold ↔ USD (inverse - gold denominated in USD)
- Oil ↔ CAD (positive - Canada is oil exporter)

**Implementation Approach:**

```python
# In trainer.py, enhance feature engineering:

def create_features_with_correlations(self, primary_symbol, correlation_pairs):
    """
    Add correlated asset features to improve predictions.

    Args:
        primary_symbol: Main pair (e.g., "EURUSD")
        correlation_pairs: Related symbols (e.g., ["GBPUSD", "USDJPY", "DXY"])
    """
    # Primary asset features (existing)
    primary_prices = load_data(primary_symbol)

    # Correlated asset features (new)
    for corr_symbol in correlation_pairs:
        corr_prices = load_data(corr_symbol)

        # Add as additional input channels
        features.append({
            'primary_price': primary_prices,
            'primary_returns': calculate_returns(primary_prices),

            # Correlation features
            f'{corr_symbol}_price': corr_prices,
            f'{corr_symbol}_returns': calculate_returns(corr_prices),
            f'{corr_symbol}_correlation': rolling_correlation(primary_prices, corr_prices)
        })

    return features
```

**Enhanced Model Architecture:**

```python
# Multi-input CNN-LSTM for cross-pair learning

from tensorflow.keras.layers import Input, Concatenate, Dense

# Input 1: Primary pair (EURUSD)
primary_input = Input(shape=(120, 1), name='primary')
primary_cnn = Conv1D(256, 2, activation='relu')(primary_input)
primary_lstm = LSTM(128)(primary_cnn)

# Input 2: Correlated pair 1 (GBPUSD)
corr1_input = Input(shape=(120, 1), name='corr1')
corr1_cnn = Conv1D(128, 2, activation='relu')(corr1_input)
corr1_lstm = LSTM(64)(corr1_cnn)

# Input 3: Correlated pair 2 (DXY - Dollar Index)
corr2_input = Input(shape=(120, 1), name='corr2')
corr2_cnn = Conv1D(128, 2, activation='relu')(corr2_input)
corr2_lstm = LSTM(64)(corr2_cnn)

# Combine all inputs
combined = Concatenate()([primary_lstm, corr1_lstm, corr2_lstm])
dense = Dense(256, activation='relu')(combined)
output = Dense(3, activation='softmax')(dense)  # BUY/SELL/NEUTRAL

model = Model(inputs=[primary_input, corr1_input, corr2_input], outputs=output)
```

**Benefits:**
- ✅ Captures inter-market dynamics
- ✅ Better predictions during divergences
- ✅ Learns regime changes (risk-on vs risk-off)
- ✅ Reduces false signals

**Example Correlation Groups:**

```python
CORRELATION_MAP = {
    'EURUSD': ['GBPUSD', 'AUDUSD', 'DXY'],  # USD majors + dollar index
    'GBPUSD': ['EURUSD', 'EURGBP', 'DXY'],
    'USDJPY': ['AUDJPY', 'DXY', 'US10Y'],   # JPY pairs + bond yields
    'XAUUSD': ['DXY', 'US10Y', 'EURUSD'],   # Gold vs USD, yields
    'XAGUSD': ['XAUUSD', 'COPPER', 'DXY'],  # Silver follows gold + industrial metals
    'CL': ['USDCAD', 'XAUUSD', 'DXY']       # Oil affects CAD
}
```

#### 2. Additional Feature Engineering

Beyond correlations, consider adding:

**Technical Indicators:**
```python
features = {
    'price': prices,
    'returns': log_returns,
    'volatility': rolling_std(returns, 14),

    # Momentum
    'rsi': calculate_rsi(prices, 14),
    'macd': calculate_macd(prices),

    # Trend
    'ma_20': moving_average(prices, 20),
    'ma_50': moving_average(prices, 50),
    'ma_200': moving_average(prices, 200),

    # Volatility
    'atr': calculate_atr(high, low, close, 14),
    'bollinger_width': bollinger_bands_width(prices),

    # Volume (if available)
    'volume': volume_data,
    'volume_ma': moving_average(volume_data, 20)
}
```

**Time-Based Features:**
```python
# Market microstructure
features['hour_of_day'] = data.index.hour  # London open, NY open patterns
features['day_of_week'] = data.index.dayofweek  # Monday vs Friday behavior
features['is_session_overlap'] = (hour >= 13) & (hour <= 16)  # London-NY overlap

# Economic calendar (if available)
features['days_to_fed'] = days_until_next_fed_meeting()
features['days_to_nfp'] = days_until_next_nonfarm_payrolls()
```

**Sentiment Features:**
```python
# Already have sentiment analyzer - integrate it!
features['news_sentiment'] = sentiment_analyzer.get_score()
features['sentiment_momentum'] = sentiment_change_rate()
```

#### 3. Implementation Priority

| Enhancement | Complexity | Impact | Recommendation |
|-------------|-----------|--------|----------------|
| **Commodities support** | Medium | High (if trading commodities) | Implement if needed |
| **Cross-pair correlations** | High | Very High | **Highly recommended** |
| **Technical indicators** | Low | Medium | Easy wins, add incrementally |
| **Time features** | Low | Medium | Quick improvement |
| **Sentiment integration** | Low | Medium | Already have infrastructure |

**Start Here:**
1. Add technical indicators (easy, good impact)
2. Add time-based features (easy, captures session patterns)
3. Implement cross-pair correlations (complex, highest impact)
4. Add commodities support if needed

## 🖥️ Web Dashboard (Interactive GUI)

**The easiest way to use the system!** A complete Streamlit web interface with all functionality.

### Launch Dashboard

```bash
poe app

# Opens browser at http://localhost:8501
```

### Features

**🏠 Home**
- System overview
- Quick stats
- Recent signals
- System status

**📈 Live Signals**
- Get real-time signals for any symbol
- Choose strategy (ML, Pattern, Hybrid)
- Interactive signal display with:
  - Entry price, SL, TP
  - Risk/reward ratio
  - Confidence score
  - Execution instructions
- Monitor multiple symbols continuously

**🔬 Backtest**
- Run backtests with custom parameters
- Interactive equity curve chart
- Detailed performance metrics
- Trade statistics

**🏆 Compare Strategies**
- Side-by-side strategy comparison
- Visual performance charts
- Identify best strategy

**🎓 Train Models**
- Train models for any symbol
- Set custom epochs
- Batch training for multiple symbols
- View existing models

**✅ Validate Models**
- Check model health
- Get retraining recommendations
- Model age tracking

**📊 Analytics**
- Performance over time
- Win rate comparisons
- Visual insights

**⚙️ Settings**
- Configure default parameters
- Set favorite symbols
- Auto-retrain settings

### Screenshots

```
┌─────────────────────────────────────┐
│  📈 Trading System Dashboard        │
├─────────────────────────────────────┤
│                                     │
│  Models: 5  │ Win Rate: 53.8%      │
│  Sharpe: 7.12 │ Strategies: 4      │
│                                     │
│  ⚡ Quick Actions                   │
│  [Get Signal] [Backtest] [Train]   │
│                                     │
│  📊 Recent Activity                 │
│  EURUSD - BUY - 75.5% confidence   │
│  XAUUSD - SELL - 68.2% confidence  │
│                                     │
└─────────────────────────────────────┘
```

### Why Use the Dashboard?

- ✅ **Visual Interface** - No command line needed
- ✅ **Interactive Charts** - Plotly visualizations
- ✅ **Real-time Signals** - Get signals with one click
- ✅ **All Features** - Complete system access
- ✅ **Easy to Use** - Intuitive navigation
- ✅ **Professional** - Production-ready UI

### Usage Example

1. **Launch dashboard:**
   ```bash
   poe app
   ```

2. **Navigate to "Live Signals"**

3. **Select symbol** (e.g., EURUSD)

4. **Choose strategy** (Hybrid recommended)

5. **Click "Generate Signal"**

6. **Get complete trading signal:**
   - Entry: 1.04523
   - Stop Loss: 1.04323
   - Take Profit: 1.04923
   - Risk/Reward: 1:2.00

7. **Execute in your broker!**

### Running 24/7

Deploy on a server for continuous monitoring:

```bash
# Install screen/tmux
sudo apt-get install screen

# Start in background
screen -S trading
poe app
# Detach: Ctrl+A, D

# Or use Docker
docker build -t trading-system .
docker run -p 8501:8501 trading-system
```

## 📱 Live Trading Signals (Without MetaTrader!)

Get real-time trading signals with entry price, stop loss, and take profit - **works with ANY broker!**

### Quick Start

```bash
# Get signal for EURUSD
poe signal

# Get signal for Gold
poe signal --symbol XAUUSD

# Get signal for Crude Oil
poe signal --symbol CL --strategy ml

# Monitor multiple symbols continuously
poe monitor --symbols EURUSD,GBPUSD,XAUUSD
```

### Example Output

```
================================================================================
                           🔔 TRADING SIGNAL
================================================================================

Symbol:              EURUSD
Time:                2026-01-21 19:30:15
Strategy:            Hybrid (ML+Pattern)
Signal:              📈 BUY

Current Price:       1.04523
Entry Price:         1.04523
Stop Loss:           1.04323
Take Profit:         1.04923

Risk/Reward:         1:2.00
Risk Amount:         $20.00
Lot Size:            0.1
Confidence:          75.5%
Pattern:             HHL

================================================================================

📝 HOW TO USE THIS SIGNAL:

1. Open your broker platform (any broker)
2. Find EURUSD
3. Place a BUY order:
   - Entry: Market order at ~1.04523
   - Stop Loss: 1.04323
   - Take Profit: 1.04923
4. Position size: 0.1 lots
5. Risk: $20.00
```

### Using with Your Broker

**Works with ALL brokers!** No MetaTrader required.

Supported brokers:
- ✅ Interactive Brokers
- ✅ TD Ameritrade
- ✅ E*TRADE
- ✅ Robinhood (forex/crypto)
- ✅ Coinbase (crypto)
- ✅ Any broker that lets you manually place trades

### Step-by-Step Trading Workflow

#### 1. Train Model (One Time)

```bash
# Train models for your favorite pairs
poe train --symbol EURUSD --epochs 200
poe train --symbol XAUUSD --epochs 200  # Gold
poe train --symbol CL --epochs 200      # Crude Oil
```

#### 2. Get Live Signal

```bash
# Get current signal
poe signal --symbol EURUSD --strategy hybrid
```

**Signal Types:**
- `📈 BUY` - Go long
- `📉 SELL` - Go short
- `⏸️ NEUTRAL` - No trade opportunity

#### 3. Execute Trade in Your Broker

When you get a BUY signal:

1. **Open your broker app/platform**
2. **Find the instrument** (EURUSD, XAUUSD, etc.)
3. **Place market order:**
   - Direction: BUY (or SELL)
   - Size: Use the lot size shown
   - Entry: Current market price
4. **Set stop loss** (SL price shown in signal)
5. **Set take profit** (TP price shown in signal)

#### 4. Monitor Position

```bash
# Check if you should close early or adjust
poe signal --symbol EURUSD

# If signal changes from BUY to SELL or NEUTRAL,
# consider closing your position
```

### Advanced: Continuous Monitoring

Monitor multiple symbols and get alerted when signals appear:

```bash
# Monitor forex majors
poe monitor --symbols EURUSD,GBPUSD,USDJPY --interval 60

# Monitor commodities
poe monitor --symbols XAUUSD,XAGUSD,CL --interval 30

# Monitor everything
poe monitor --symbols EURUSD,GBPUSD,XAUUSD,CL,WHEAT
```

**Output:**
```
🤖 Live Signal Monitor Started
Symbols: EURUSD, GBPUSD, XAUUSD
Strategy: hybrid
Interval: 60 minutes
Press Ctrl+C to stop

================================================================================
                           🔔 TRADING SIGNAL
================================================================================
Symbol:              XAUUSD
Signal:              📈 BUY
Entry Price:         2045.30
Stop Loss:           2040.50
Take Profit:         2054.90
...

Next check in 60 minutes...
```

### Commodities Trading

**Now supports commodities directly!**

```bash
# Gold
poe train --symbol XAUUSD --epochs 200
poe signal --symbol XAUUSD

# Silver
poe train --symbol XAGUSD --epochs 200
poe signal --symbol XAGUSD

# Crude Oil (WTI)
poe train --symbol CL --epochs 200
poe signal --symbol CL

# Natural Gas
poe train --symbol NATGAS --epochs 200
poe signal --symbol NATGAS

# Agricultural
poe train --symbol WHEAT --epochs 200
poe signal --symbol WHEAT
```

**Supported Commodities:**
- **Precious Metals**: XAUUSD (Gold), XAGUSD (Silver), XPTUSD (Platinum), XPDUSD (Palladium)
- **Energy**: CL (Crude Oil), BRENT (Brent Oil), NATGAS (Natural Gas)
- **Metals**: COPPER, ALUMINUM, ZINC
- **Agricultural**: WHEAT, CORN, SOYBEAN, COFFEE, SUGAR, COTTON

### Strategy Selection

Choose the best strategy for your trading style:

```bash
# ML Strategy - More trades, trending markets
poe signal --symbol EURUSD --strategy ml

# Pattern Strategy - Fewer trades, high quality
poe signal --symbol EURUSD --strategy pattern

# Hybrid Strategy - Best of both (recommended)
poe signal --symbol EURUSD --strategy hybrid
```

### Python API

For programmatic access:

```python
from trading_system.live_trading import LiveSignalGenerator

# Create generator
generator = LiveSignalGenerator(strategy_type="hybrid")

# Get signal
signal = generator.generate_signal("EURUSD")

# Print formatted signal
signal.print_signal()

# Access signal data
if signal.signal == "BUY":
    print(f"Enter BUY at {signal.entry_price}")
    print(f"Stop Loss: {signal.stop_loss}")
    print(f"Take Profit: {signal.take_profit}")
    print(f"Risk: ${signal.risk_amount:.2f}")

# Convert to dict (for JSON/API)
signal_data = signal.to_dict()
```

### Automation Options

#### Option 1: Cron Job (Check Hourly)

```bash
# Edit crontab
crontab -e

# Add this line - check every hour
0 * * * * cd /path/to/trading-system && source .venv/bin/activate && poe signal --symbol EURUSD >> signals.log
```

#### Option 2: Continuous Monitor

```bash
# Run in tmux/screen session
tmux new -s trading
poe monitor --symbols EURUSD,GBPUSD,XAUUSD --interval 60
# Detach: Ctrl+b, d
```

#### Option 3: Telegram Bot

```python
# Custom integration (example)
from trading_system.live_trading import LiveSignalGenerator
import telegram

bot = telegram.Bot(token="YOUR_TOKEN")
generator = LiveSignalGenerator(strategy_type="hybrid")

signal = generator.generate_signal("EURUSD")

if signal.signal != "NEUTRAL":
    message = f"""
🔔 Trading Signal: {signal.signal}

Symbol: {signal.symbol}
Entry: {signal.entry_price}
SL: {signal.stop_loss}
TP: {signal.take_profit}
Confidence: {signal.confidence}%
    """
    bot.send_message(chat_id="YOUR_CHAT_ID", text=message)
```

### Important Notes

1. **Data Source**: Uses Yahoo Finance (free, no API key needed)
2. **Updates**: Hourly data (suitable for H1 strategy)
3. **No MetaTrader Required**: Works with any broker
4. **Manual Execution**: You place trades yourself in your broker
5. **Risk Management**: Always use the SL/TP provided
6. **Paper Trading**: Test with demo account first!

### Troubleshooting

**"Model not found"**
```bash
# Train the model first
poe train --symbol EURUSD --epochs 200
```

**"No data received"**
```bash
# Check symbol format
poe signal --symbol EURUSD  # Correct
poe signal --symbol EUR/USD  # Wrong
```

**"Symbol not supported"**
```bash
# Check available symbols in correlations.py
# Or add your own commodity spec
```

## 🏆 Strategy Comparison

Compare multiple strategies on the same historical data to find the best performer.

### Quick Start

```bash
# Compare all strategies on EURUSD for 90 days
poe compare --symbol EURUSD --days 90

# View results
open strategy_comparison/summary_dashboard.png
```

### Available Strategies

1. **ML Optimized** - CNN-LSTM with 200-MA trend filter, time filter, trailing stop
2. **Pattern (HLC)** - Pure Nobel Prize pattern recognition with adaptive stops
3. **Hybrid (ML+Pattern)** - Only trades when both ML and pattern agree
4. **ML Aggressive** - Tighter stops (1.5 ATR), higher targets (5.0 ATR)

### Python API

```python
from trading_system.comparison import StrategyComparator
from trading_system.strategies import MLStrategy, PatternStrategy, HybridStrategy
from trading_system import ForexPredictor
from trading_system.backtest import BacktestEngine
from datetime import datetime, timedelta

# Load data
engine = BacktestEngine(initial_balance=10000)
end_date = datetime.now()
start_date = end_date - timedelta(days=90)
data = engine.load_data("EURUSD", "H1", start_date, end_date)

# Load ML model
predictor = ForexPredictor("./models/model.EURUSD.H1.120.onnx", history_size=120)
predictor.min_price = float(data['close'].min())
predictor.max_price = float(data['close'].max())

# Create comparator
comparator = StrategyComparator(initial_balance=10000, slippage_pips=1.0)

# Add strategies
ml_strategy = MLStrategy(predictor, sl_atr_mult=2.0, tp_atr_mult=4.0)
comparator.add_strategy(ml_strategy, "ML Optimized")

pattern_strategy = PatternStrategy(min_strength=70, min_rr=2.0)
comparator.add_strategy(pattern_strategy, "Pattern (HLC)")

hybrid_strategy = HybridStrategy(predictor, min_pattern_strength=60)
comparator.add_strategy(hybrid_strategy, "Hybrid")

# Run comparison with visualizations
result = comparator.run_and_visualize(
    data=data,
    symbol="EURUSD",
    output_dir="./strategy_comparison"
)

# Print results
result.print_summary()

# Get best strategy
best = result.get_winner('sharpe_ratio')
print(f"Best strategy: {best}")
```

### Comparison Outputs

The comparison generates 6 professional charts in `./strategy_comparison/`:

1. **equity_curves.png** - All strategies' equity curves on one chart
2. **metrics_comparison.png** - 6-panel comparison (profit, win rate, PF, Sharpe, DD, trades)
3. **trade_distributions.png** - P&L distribution histogram for each strategy
4. **drawdown_comparison.png** - Drawdown curves comparison
5. **win_loss_analysis.png** - Average and largest wins/losses
6. **summary_dashboard.png** - Complete dashboard with metrics table and charts

### Metrics Compared

- **Total Trades** - Number of trades executed
- **Win Rate** - Percentage of profitable trades
- **Profit Factor** - Gross profit / Gross loss (need >1.0)
- **Sharpe Ratio** - Risk-adjusted returns (higher is better)
- **Net Profit** - Total profit/loss in dollars and percentage
- **Max Drawdown** - Largest peak-to-trough decline
- **Average Win/Loss** - Mean profit per winning/losing trade
- **Largest Win/Loss** - Best and worst single trade

### Creating Custom Strategies

Extend the `BaseStrategy` class:

```python
from trading_system.strategies import BaseStrategy, StrategySignal
import pandas as pd

class MyCustomStrategy(BaseStrategy):
    def __init__(self, my_param=10):
        super().__init__()
        self.my_param = my_param

    def get_name(self) -> str:
        return "My Custom Strategy"

    def get_description(self) -> str:
        return f"Custom strategy with param={self.my_param}"

    def generate_signal(self, data: pd.DataFrame, index: int):
        # Your strategy logic here
        if index < 100:
            return None

        # Example: Simple MA crossover
        ma_fast = self.calculate_ma(data, index, 20)
        ma_slow = self.calculate_ma(data, index, 50)
        current_price = data.iloc[index]['close']
        atr = self.calculate_atr(data, index, 14)

        if ma_fast > ma_slow:  # Bullish
            return StrategySignal(
                signal="BUY",
                entry_price=current_price,
                stop_loss=current_price - (atr * 2.0),
                take_profit=current_price + (atr * 3.0),
                lot_size=0.1,
                confidence=75.0,
                metadata={'ma_fast': ma_fast, 'ma_slow': ma_slow}
            )

        return None

# Use it
comparator = StrategyComparator()
comparator.add_strategy(MyCustomStrategy(my_param=15), "My Strategy")
result = comparator.run(data, "EURUSD")
```

### Tips for Strategy Development

1. **Start with BaseStrategy** - Inherit and implement `generate_signal()` and `get_name()`
2. **Use helper methods** - `calculate_atr()`, `calculate_ma()` are built-in
3. **Return StrategySignal** - Structured signal with all trade parameters
4. **Validate signals** - `is_valid_signal()` checks risk/reward automatically
5. **Add metadata** - Track custom metrics in `metadata` dict for analysis
6. **Test extensively** - Use comparison tool to validate against other strategies

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
**Training:** `train`, `train-all`, `validate`
**Trading:** `predict`, `sentiment`, `backtest`, `compare`
**Live Signals:** `signal`, `monitor`
**Web Dashboard:** `app` (launch interactive GUI - easiest way to use!)
**Dev:** `format`, `lint`, `lint-fix`, `type-check`, `test`, `test-cov`, `clean`
**Examples:** `example-train`, `example-backtest`, `example-compare`, `examples`
**Workflows:** `check`, `quick-start`, `full-workflow`
**Utility:** `info`, `version`, `docs`

## 📄 License

MIT License - Free for commercial and personal use.

## ⚠️ Disclaimer

This software is for educational purposes only. Trading carries risk. Past performance does not guarantee future results. Always test with paper trading first.

---

**Quick Start:** `uv venv && source .venv/bin/activate && poe install-dev && poe quick-start`

**Happy Trading!** 📈
