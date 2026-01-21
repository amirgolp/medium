# Trading Strategy Optimization Guide

## Current Issues (Based on Backtest Results)

- **Win Rate: 44.4%** - Below break-even (need >50% with current risk/reward)
- **Profit Factor: 0.79** - Losing $0.79 for every $1 won (need >1.0)
- **Negative Sharpe: -1.52** - Strategy loses money consistently
- **Issue**: Current SL (2.5 ATR) ≈ TP (3.0 ATR) gives poor risk/reward

---

## 1. Immediate Optimizations (No Code Changes)

### A. Adjust Risk/Reward Ratio
Current: SL=2.5 ATR, TP=3.0 ATR (1:1.2 ratio)
**Recommended**:
- Conservative: SL=2.0 ATR, TP=4.0 ATR (1:2 ratio)
- Aggressive: SL=1.5 ATR, TP=4.5 ATR (1:3 ratio)

**Why**: With 44% win rate, you need larger wins to offset losses.

### B. Add Trailing Stop
- Activate after profit reaches 1.5 ATR
- Trail by 1.0 ATR (lock in profits)

### C. Filter Trading Hours
- Only trade 11:00-22:00 GMT (high liquidity)
- Avoid news event times (first Friday of month)

---

## 2. Model Optimization

### A. Retrain with More Data
```bash
# Train with more historical data
poe train --symbol EURUSD --epochs 200

# Current: 120 days training
# Recommended: 365+ days for better patterns
```

### B. Optimize Model Architecture
- Increase history size: 120 → 240 hours
- Add more features: RSI, MACD, volume
- Experiment with learning rate

### C. Ensemble Multiple Models
- Train 3 models with different parameters
- Only trade when 2/3 agree

---

## 3. Signal Filtering (Reduce False Signals)

### A. Add Trend Filter
```python
# Only trade in direction of longer trend
ma_200 = data['close'].rolling(200).mean()
current_price = data['close'].iloc[i]

if signal == "BUY" and current_price < ma_200:
    return None  # Don't buy in downtrend
```

### B. Add Volatility Filter
```python
# Skip low volatility periods
if atr < threshold:
    return None  # Market too quiet
```

### C. Use Combined Strategy
- Require both ML + Sentiment agreement
- Reduces trades but increases quality

---

## 4. Position Sizing Optimization

### A. Risk-Based Position Sizing
```python
# Instead of fixed 0.1 lots
account_risk = 0.03  # Risk 3% per trade
sl_distance = abs(entry - stop_loss)
lot_size = (balance * account_risk) / sl_distance
```

### B. Kelly Criterion
```python
# Optimal position size based on win rate
win_rate = 0.444
avg_win = 4.68
avg_loss = 4.75
kelly = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win
position_size = balance * kelly * 0.5  # Use 50% Kelly for safety
```

---

## 5. Parameter Optimization (Grid Search)

### Test Different Combinations:
```python
# Parameters to test
sl_multipliers = [1.5, 2.0, 2.5, 3.0]
tp_multipliers = [3.0, 4.0, 5.0, 6.0]
atr_periods = [10, 14, 20]

# Run backtest for each combination
# Find best Sharpe ratio
```

### Optimize via Walk-Forward Analysis
1. Train on 6 months of data
2. Test on next 3 months
3. Roll forward, repeat
4. Find parameters that work consistently

---

## 6. Advanced Optimizations

### A. Machine Learning Improvements
- **Add more features**: RSI, MACD, Bollinger Bands, Volume
- **Use attention mechanism**: Let model focus on important patterns
- **Train on multiple timeframes**: H1, H4, D1 simultaneously

### B. Sentiment Integration
- Use real news APIs (currently simulated)
- Weight by news source reliability
- Consider sentiment momentum (improving vs declining)

### C. Multi-Symbol Portfolio
```bash
# Train models for multiple pairs
poe train --symbol EURUSD --epochs 200
poe train --symbol GBPUSD --epochs 200
poe train --symbol USDJPY --epochs 200

# Diversify risk across pairs
```

---

## 7. Quick Optimization Script

Create `optimize_strategy.py`:

```python
from trading_system import BacktestEngine
import numpy as np

# Test different SL/TP combinations
results = []
for sl_mult in [1.5, 2.0, 2.5, 3.0]:
    for tp_mult in [3.0, 4.0, 5.0]:
        if tp_mult <= sl_mult:
            continue

        # Run backtest with these parameters
        result = run_backtest(sl_mult, tp_mult)

        results.append({
            'sl': sl_mult,
            'tp': tp_mult,
            'profit_factor': result.profit_factor,
            'sharpe': result.sharpe_ratio,
            'net_profit': result.net_profit
        })

# Find best parameters
best = max(results, key=lambda x: x['sharpe'])
print(f"Best SL: {best['sl']}, TP: {best['tp']}")
```

---

## 8. Recommended Action Plan

### Phase 1: Quick Wins (Today)
1. ✅ Run combined strategy (filters bad signals)
2. ✅ Change SL=2.0 ATR, TP=4.0 ATR
3. ✅ Add time filter (11:00-22:00 GMT)

### Phase 2: Data & Training (This Week)
1. ⏳ Collect more historical data (365+ days)
2. ⏳ Retrain with larger dataset
3. ⏳ Add trend filter (200 MA)

### Phase 3: Advanced (Next Week)
1. ⏳ Implement risk-based position sizing
2. ⏳ Run grid search optimization
3. ⏳ Add more technical indicators

---

## 9. Expected Improvements

| Optimization | Expected Improvement |
|-------------|---------------------|
| Better SL/TP ratio (1:2) | Profit factor: 0.79 → 1.2+ |
| Combined strategy | Win rate: 44% → 55%+ |
| Trend filter | Reduce losing trades by 30% |
| Risk-based sizing | Better risk management |
| More training data | More accurate predictions |

---

## 10. Monitoring & Validation

### After Each Change:
```bash
# Run backtest
poe backtest --symbol EURUSD --days 90

# Compare metrics:
# - Profit Factor (target: >1.3)
# - Sharpe Ratio (target: >0.5)
# - Max Drawdown (target: <10%)
# - Win Rate (target: >50%)
```

### Walk-Forward Testing:
1. Optimize on Jan-Jun 2025
2. Test on Jul-Sep 2025
3. If still profitable → good parameters
4. If not → overfit, try different approach

---

## Key Takeaway

**The current strategy loses money because:**
1. Risk/reward is poor (1:1.2 instead of 1:2+)
2. No trend filter (trading against major trends)
3. Too many false signals (44% win rate)

**Quick fix**: Use combined strategy + better SL/TP ratios
**Long-term**: More data, better features, parameter optimization
