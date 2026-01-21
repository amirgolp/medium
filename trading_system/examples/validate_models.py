"""
Model Validation Script
========================

Check if models need retraining by comparing recent backtest performance.
"""

import os
from datetime import datetime, timedelta
from trading_system.backtest import BacktestEngine
from trading_system import ForexPredictor
from trading_system.strategies import MLStrategy
import logging

logging.basicConfig(level=logging.WARNING)


def validate_model(symbol: str, days: int = 30) -> dict:
    """
    Validate model performance on recent data.

    Args:
        symbol: Currency pair
        days: Days to test

    Returns:
        Dict with validation metrics
    """
    model_path = f"./models/model.{symbol}.H1.120.onnx"

    if not os.path.exists(model_path):
        return {
            'symbol': symbol,
            'exists': False,
            'message': 'Model not found - needs initial training'
        }

    # Load model
    try:
        predictor = ForexPredictor(model_path, history_size=120)
    except Exception as e:
        return {
            'symbol': symbol,
            'exists': True,
            'error': str(e),
            'message': 'Model loading failed'
        }

    # Load recent data
    engine = BacktestEngine(initial_balance=10000)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    data = engine.load_data(symbol, "H1", start_date, end_date)

    # Set price normalization
    predictor.min_price = float(data['close'].min())
    predictor.max_price = float(data['close'].max())

    # Create strategy
    strategy = MLStrategy(predictor, history_size=120)

    # Run backtest
    result = engine.run_backtest(data, strategy, symbol)

    # Check model age
    model_age_days = (datetime.now() - datetime.fromtimestamp(
        os.path.getmtime(model_path)
    )).days

    # Determine if retraining needed
    needs_retrain = False
    reasons = []

    if result.sharpe_ratio < 0.5:
        needs_retrain = True
        reasons.append(f"Low Sharpe ratio ({result.sharpe_ratio:.2f})")

    if result.profit_factor < 1.0:
        needs_retrain = True
        reasons.append(f"Profit factor below 1.0 ({result.profit_factor:.2f})")

    if result.win_rate < 40:
        needs_retrain = True
        reasons.append(f"Low win rate ({result.win_rate:.1f}%)")

    if model_age_days > 30:
        needs_retrain = True
        reasons.append(f"Model is {model_age_days} days old (>30 days)")

    return {
        'symbol': symbol,
        'exists': True,
        'model_age_days': model_age_days,
        'trades': result.total_trades,
        'win_rate': result.win_rate,
        'profit_factor': result.profit_factor,
        'sharpe_ratio': result.sharpe_ratio,
        'net_profit': result.net_profit,
        'needs_retrain': needs_retrain,
        'reasons': reasons,
        'message': 'Model performing well' if not needs_retrain else 'Model needs retraining'
    }


def main():
    """Validate all models."""
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
    days = 30  # Test on last 30 days

    print("="*80)
    print("MODEL VALIDATION REPORT")
    print("="*80)
    print(f"Testing on last {days} days of data\n")

    results = []
    for symbol in symbols:
        print(f"Validating {symbol}...", end=" ", flush=True)
        result = validate_model(symbol, days)
        results.append(result)
        print("✓" if result.get('exists') else "✗")

    # Print summary
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80 + "\n")

    needs_retraining = []

    for result in results:
        symbol = result['symbol']
        exists = result.get('exists', False)

        print(f"{symbol}:")

        if not exists:
            print(f"  ✗ {result['message']}")
            needs_retraining.append(symbol)
        elif 'error' in result:
            print(f"  ✗ {result['message']}: {result['error']}")
            needs_retraining.append(symbol)
        else:
            age = result['model_age_days']
            print(f"  • Model age: {age} days")
            print(f"  • Trades: {result['trades']}")
            print(f"  • Win rate: {result['win_rate']:.1f}%")
            print(f"  • Profit factor: {result['profit_factor']:.2f}")
            print(f"  • Sharpe ratio: {result['sharpe_ratio']:.2f}")
            print(f"  • Net profit: ${result['net_profit']:.2f}")

            if result['needs_retrain']:
                print(f"  ⚠️  NEEDS RETRAINING")
                for reason in result['reasons']:
                    print(f"     - {reason}")
                needs_retraining.append(symbol)
            else:
                print(f"  ✓ {result['message']}")

        print()

    # Recommendations
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80 + "\n")

    if needs_retraining:
        print(f"⚠️  {len(needs_retraining)} model(s) need retraining:\n")
        for symbol in needs_retraining:
            print(f"  poe train --symbol {symbol} --epochs 150")

        print("\nOr train all at once:")
        print(f"  ./scripts/train_all_pairs.sh")
    else:
        print("✓ All models are performing well!")
        print("  No retraining needed at this time.")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
