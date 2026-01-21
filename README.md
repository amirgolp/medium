# Trading System

A sophisticated three-pillar automated trading system combining machine learning predictions, sentiment-based news analysis, and advanced risk management for forex trading.

## Features

### 🤖 Pillar 1: ML-Powered Forex Prediction
- **CNN-LSTM Hybrid Neural Network** for price prediction
- Weekly model retraining with fresh market data
- Supports multiple currency pairs (EURUSD, GBPUSD, etc.)
- ONNX model export for production deployment
- Automated data collection from MetaTrader 5

**Architecture:**
```
Input (120h) → Conv1D (256) → MaxPooling → LSTM (100) → Dropout →
LSTM (100) → Dropout → Dense (sigmoid) → Prediction
```

### 📰 Pillar 2: Sentiment-Based News Analysis
- Multi-source news aggregation (NewsAPI, Finnhub, MT5 Calendar)
- Natural language sentiment analysis
- Economic event impact assessment
- Currency-specific sentiment scoring (0-100 scale)
- Automated signal generation based on sentiment divergence

### 💼 Pillar 3: Risk Management
- **ATR-Based Dynamic Stops**: Automatically adjust to market volatility
- **Multiple Position Sizing Modes**:
  - Fixed lot size
  - Risk-based (% of account per trade)
  - Balance proportion
- **Portfolio Controls**:
  - Maximum position limits
  - Daily loss limits
  - Trading hour restrictions
  - Position holding time limits

## Installation

### Prerequisites
- Python 3.8 or higher
- MetaTrader 5 (for live trading, optional for development)
- API keys for NewsAPI and Finnhub (optional)

### Install Package

```bash
# Clone the repository
git clone https://github.com/yourusername/trading-system.git
cd trading-system

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Optional Dependencies

```bash
# For MetaTrader 5 integration (Windows only)
pip install MetaTrader5

# For development
pip install -e ".[dev]"
```

## Quick Start

### 1. Train ML Models

```python
from trading_system import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(
    history_size=120,
    training_days=120,
    early_stop_patience=20
)

# Train models for multiple symbols
symbols = ["EURUSD", "GBPUSD", "USDCHF"]
results = trainer.train_multiple_symbols(symbols, epochs=300)

# Save models
for symbol, data in results.items():
    model = data['model']
    trainer.save_model_onnx(model, symbol, output_dir="./models")
```

### 2. Make Predictions

```python
from trading_system import ForexPredictor

# Load trained model
predictor = ForexPredictor(
    model_path="./models/model.EURUSD.H1.120.onnx",
    history_size=120
)

# Update normalization parameters
predictor.update_min_max("EURUSD", lookback_hours=2880)

# Make prediction
prediction, predicted_price, last_close = predictor.predict_symbol("EURUSD")

signal = predictor.get_signal_name(prediction)
print(f"Signal: {signal}")
print(f"Last Close: {last_close:.5f}")
print(f"Predicted: {predicted_price:.5f}")
```

### 3. Analyze Sentiment

```python
from trading_system import SentimentAnalyzer
from trading_system.sentiment_analyzer import NewsAPIClient, FinnhubClient

# Initialize analyzer
analyzer = SentimentAnalyzer(
    lookback_hours=24,
    min_sentiment_score=10.0
)

# Fetch news from APIs
news_api = NewsAPIClient(api_key="YOUR_NEWSAPI_KEY")
articles = news_api.fetch_news()
sentiments = news_api.parse_to_sentiment(articles, analyzer)

# Add to analyzer
for sentiment in sentiments:
    analyzer.add_sentiment(sentiment)

# Generate trading signal
signal, confidence, details = analyzer.generate_signal("EUR", "USD")

if signal:
    print(f"Signal: {signal} EURUSD (Confidence: {confidence:.1f})")
```

### 4. Calculate Position Size

```python
from trading_system import PositionSizer, ATRCalculator
from trading_system.risk_management import LotMode

# Initialize position sizer
sizer = PositionSizer(
    mode=LotMode.RISK_BASED,
    risk_percent=3.0,
    max_lot=10.0
)

# Calculate ATR-based stop loss distance
atr_calc = ATRCalculator(period=14)
sl_pips = atr_calc.get_sl_distance_pips("EURUSD", sl_multiplier=2.5)

# Calculate lot size
lot = sizer.calculate_lot("EURUSD", sl_distance_pips=sl_pips)

print(f"Position Size: {lot:.2f} lots")
print(f"Stop Loss: {sl_pips:.1f} pips")
```

### 5. Manage Risk

```python
from trading_system import RiskManager

# Initialize risk manager
risk_mgr = RiskManager(
    max_positions=1,
    max_daily_loss_pct=5.0,
    holding_period_hours=24,
    magic_number=123456
)

# Check if we can trade
can_trade, reason = risk_mgr.can_open_position("EURUSD")

if can_trade:
    print("Ready to trade")
else:
    print(f"Cannot trade: {reason}")

# Get portfolio status
status = risk_mgr.get_portfolio_status()
print(f"Positions: {status['position_count']}/{status['max_positions']}")
print(f"Daily PnL: ${status['daily_pnl']:.2f}")
```

## Complete Trading Example

```python
from trading_system import (
    ForexPredictor,
    SentimentAnalyzer,
    PositionSizer,
    RiskManager,
    ATRCalculator
)
from trading_system.risk_management import LotMode

# 1. Check risk management
risk_mgr = RiskManager(max_positions=1, max_daily_loss_pct=5.0)
can_trade, reason = risk_mgr.can_open_position("EURUSD")

if not can_trade:
    print(f"Trading not allowed: {reason}")
    exit()

# 2. Get ML prediction
predictor = ForexPredictor("./models/model.EURUSD.H1.120.onnx", history_size=120)
predictor.update_min_max("EURUSD")
ml_signal, pred_price, last_close = predictor.predict_symbol("EURUSD")

# 3. Get sentiment signal
analyzer = SentimentAnalyzer(lookback_hours=24, min_sentiment_score=10.0)
# ... load news data into analyzer ...
sent_signal, confidence, details = analyzer.generate_signal("EUR", "USD")

# 4. Combine signals
if predictor.get_signal_name(ml_signal) == sent_signal:
    print(f"✓ Signals aligned: {sent_signal}")

    # 5. Calculate position size
    atr_calc = ATRCalculator(period=14)
    sl_price, tp_price, atr = atr_calc.calculate_sl_tp("EURUSD", sent_signal)

    sizer = PositionSizer(mode=LotMode.RISK_BASED, risk_percent=3.0)
    sl_pips = atr_calc.get_sl_distance_pips("EURUSD", sl_multiplier=2.5)
    lot = sizer.calculate_lot("EURUSD", sl_distance_pips=sl_pips)

    print(f"Signal: {sent_signal}")
    print(f"Lot Size: {lot:.2f}")
    print(f"SL: {sl_price:.5f}, TP: {tp_price:.5f}")

    # Execute trade via MT5 or your broker's API
else:
    print("Signals not aligned, waiting...")
```

## Configuration

### API Keys

Create a `.env` file in your project root:

```env
NEWSAPI_KEY=your_newsapi_key_here
FINNHUB_KEY=your_finnhub_key_here
```

Get free API keys:
- NewsAPI: https://newsapi.org
- Finnhub: https://finnhub.io

### Model Parameters

Edit training parameters in your training script:

```python
trainer = ModelTrainer(
    history_size=120,          # Hours of history for input
    training_days=120,         # Days of historical data
    test_split=0.2,           # Train/test split ratio
    early_stop_patience=20    # Early stopping patience
)
```

### Risk Parameters

Configure risk management:

```python
risk_mgr = RiskManager(
    max_positions=1,              # Max simultaneous positions
    max_daily_loss_pct=5.0,      # Max daily loss (%)
    holding_period_hours=24,      # Max position holding time
    magic_number=123456          # EA identifier
)

sizer = PositionSizer(
    mode=LotMode.RISK_BASED,     # Position sizing mode
    risk_percent=3.0,            # Risk per trade (%)
    max_lot=10.0                 # Maximum lot size
)

atr_calc = ATRCalculator(
    period=14                    # ATR period
)
```

## Architecture

```
trading_system/
├── ml_predictor/           # ML prediction module
│   ├── trainer.py         # Model training
│   └── predictor.py       # Inference engine
├── sentiment_analyzer/     # Sentiment analysis
│   ├── analyzer.py        # Core sentiment logic
│   └── news_sources.py    # API clients
├── risk_management/        # Risk management
│   ├── position_sizer.py  # Position sizing
│   ├── risk_manager.py    # Portfolio risk control
│   └── atr_calculator.py  # ATR calculations
└── utils/                  # Utilities
    └── helpers.py         # Helper functions
```

## Current Optimized Parameters (Jan 16, 2026)

### ML Predictor
- History size: 120 hours
- Training days: 120
- Early stop patience: 20 epochs
- Architecture: Conv1D(256) → MaxPool → LSTM(100) → LSTM(100)

### Sentiment Analyzer
- Lookback hours: 24
- Min sentiment score: 10.0
- High impact only: True

### Risk Management
- Risk per trade: 3%
- ATR period: 14
- SL multiplier: 2.5x ATR
- TP multiplier: 3.0x ATR
- Trailing stop: 1.0x ATR
- Trading hours: 11:00-22:00 GMT

## Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=trading_system tests/
```

## License

MIT License - see LICENSE file for details

## Disclaimer

This software is for educational purposes only. Trading financial instruments carries risk.
Past performance does not guarantee future results. Always test thoroughly with paper trading
before using real capital.

## Support

- Issues: https://github.com/yourusername/trading-system/issues
- Documentation: https://github.com/yourusername/trading-system/wiki

## Contributing

Contributions are welcome! Please read CONTRIBUTING.md for guidelines.

---

**Built with Python, TensorFlow, and ONNX**

*Happy Trading! 📈*
