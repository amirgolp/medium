"""
Example: Run Backtest
======================

Backtest ML prediction strategy on historical data.
"""

from trading_system.backtest import BacktestEngine, BacktestVisualizer
from trading_system import ForexPredictor
from trading_system.risk_management import ATRCalculator
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def ml_strategy(data, i, predictor, atr_calc, symbol):
    """
    ML-based strategy using trained predictor.

    Args:
        data: Price data DataFrame
        i: Current bar index
        predictor: ForexPredictor instance
        atr_calc: ATRCalculator instance
        symbol: Trading symbol

    Returns:
        Signal dictionary or None
    """
    # Need enough history
    if i < 120:
        return None

    # Get recent prices
    recent_prices = data.iloc[i-120:i]['close'].values

    try:
        # Make prediction
        prediction, pred_price, last_close = predictor.predict(recent_prices)
        signal = predictor.get_signal_name(prediction)

        if signal == "NEUTRAL":
            return None

        # Calculate ATR from recent bars
        highs = data.iloc[i-14:i]['high'].values
        lows = data.iloc[i-14:i]['low'].values
        closes = data.iloc[i-14:i]['close'].values

        atr = atr_calc.calculate_atr_from_data(highs, lows, closes)

        # Get current price
        current_price = data.iloc[i]['close']

        # Calculate stops based on ATR
        if signal == "BUY":
            sl = current_price - (atr * 2.5)
            tp = current_price + (atr * 3.0)
        else:  # SELL
            sl = current_price + (atr * 2.5)
            tp = current_price - (atr * 3.0)

        return {
            'signal': signal,
            'stop_loss': sl,
            'take_profit': tp,
            'lot_size': 0.1,
            'ml_signal': signal
        }

    except Exception as e:
        logging.warning(f"Strategy error at bar {i}: {e}")
        return None


def main():
    """Run backtest example."""

    symbol = "EURUSD"
    days = 90
    initial_balance = 10000.0

    print("="*70)
    print("BACKTEST: ML PREDICTION STRATEGY")
    print("="*70)
    print(f"Symbol: {symbol}")
    print(f"Period: {days} days")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Strategy: CNN-LSTM predictions + ATR stops")
    print("="*70 + "\n")

    # Initialize backtest engine
    engine = BacktestEngine(
        initial_balance=initial_balance,
        slippage_pips=1.0
    )

    # Load historical data
    print("Loading historical data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    data = engine.load_data(symbol, "H1", start_date, end_date)
    print(f"✓ Loaded {len(data)} bars\n")

    # Load trained model
    print("Loading ML model...")
    try:
        model_path = f"./models/model.{symbol}.H1.120.onnx"
        predictor = ForexPredictor(model_path, history_size=120)

        # Set min/max from data
        predictor.min_price = float(data['close'].min())
        predictor.max_price = float(data['close'].max())

        print(f"✓ Model loaded: {model_path}")
        print(f"  Min/Max: {predictor.min_price:.5f} / {predictor.max_price:.5f}\n")

    except Exception as e:
        print(f"✗ Could not load model: {e}")
        print("Run 'python trading_system/examples/train_models.py' first\n")
        return

    # Initialize ATR calculator
    atr_calc = ATRCalculator(period=14)

    # Run backtest
    print("Running backtest...")
    print("This may take a moment...\n")

    result = engine.run_backtest(
        data=data,
        strategy_func=ml_strategy,
        symbol=symbol,
        predictor=predictor,
        atr_calc=atr_calc
    )

    # Print results
    result.print_summary()

    # Show trade details
    if result.trades:
        print("\n" + "="*70)
        print("RECENT TRADES (Last 10)")
        print("="*70)

        for trade in result.trades[-10:]:
            profit_emoji = "✅" if trade.profit_loss > 0 else "❌"
            print(f"\n{profit_emoji} {trade.direction} @ {trade.entry_price:.5f}")
            print(f"   Entry: {trade.entry_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"   Exit:  {trade.exit_time.strftime('%Y-%m-%d %H:%M')} ({trade.exit_reason})")
            print(f"   P&L:   ${trade.profit_loss:+.2f} ({trade.pips:+.1f} pips)")
            print(f"   Duration: {trade.duration_hours:.1f} hours")

    # Visualize results
    print("\n" + "="*70)
    print("VISUALIZATION")
    print("="*70)
    print("\nGenerating charts...")

    try:
        visualizer = BacktestVisualizer()

        # Create full report
        visualizer.create_full_report(result, data, output_dir="./backtest_results")

        print("\n✓ Charts saved to ./backtest_results/")
        print("\nGenerated files:")
        print("  - equity_curve.png")
        print("  - drawdown.png")
        print("  - trade_distribution.png")
        print("  - win_loss_analysis.png")
        print("  - trade_timeline.png")

    except ImportError:
        print("\n⚠️  Matplotlib not installed. Install with:")
        print("   pip install matplotlib seaborn")

    print("\n" + "="*70)
    print("BACKTEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
