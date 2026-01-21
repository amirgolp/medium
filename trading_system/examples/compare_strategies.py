"""
Example: Compare Multiple Strategies
=====================================

Compare ML, Pattern, and Hybrid strategies on the same historical data.
"""

from trading_system.comparison import StrategyComparator
from trading_system.strategies import MLStrategy, PatternStrategy, HybridStrategy
from trading_system import ForexPredictor
from trading_system.backtest import BacktestEngine
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    """Compare multiple strategies."""

    symbol = "EURUSD"
    days = 90
    initial_balance = 10000.0

    print("="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    print(f"Symbol: {symbol}")
    print(f"Period: {days} days")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print("="*80 + "\n")

    # Load historical data
    print("Loading historical data...")
    engine = BacktestEngine(initial_balance=initial_balance)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    data = engine.load_data(symbol, "H1", start_date, end_date)
    print(f"✓ Loaded {len(data)} bars\n")

    # Load ML model
    print("Loading ML model...")
    try:
        model_path = f"./models/model.{symbol}.H1.120.onnx"
        predictor = ForexPredictor(model_path, history_size=120)
        predictor.min_price = float(data['close'].min())
        predictor.max_price = float(data['close'].max())
        print(f"✓ Model loaded\n")
    except Exception as e:
        print(f"✗ Could not load model: {e}")
        print("Run 'poe train --symbol EURUSD --epochs 100' first\n")
        return

    # Create comparator
    comparator = StrategyComparator(
        initial_balance=initial_balance,
        slippage_pips=1.0
    )

    # Add strategies
    print("Preparing strategies...")

    # 1. ML Strategy (Optimized)
    ml_strategy = MLStrategy(
        predictor=predictor,
        history_size=120,
        sl_atr_mult=2.0,
        tp_atr_mult=4.0,
        trailing_atr_mult=1.0,
        min_rr=2.0
    )
    comparator.add_strategy(ml_strategy, "ML Optimized")

    # 2. Pattern Strategy
    pattern_strategy = PatternStrategy(
        swing_lookback=20,
        atr_period=14,
        min_strength=70,
        min_rr=2.0,
        max_volatility_pct=5.0
    )
    comparator.add_strategy(pattern_strategy, "Pattern (HLC)")

    # 3. Hybrid Strategy
    hybrid_strategy = HybridStrategy(
        predictor=predictor,
        history_size=120,
        swing_lookback=20,
        atr_period=14,
        min_pattern_strength=60,
        min_rr=2.0
    )
    comparator.add_strategy(hybrid_strategy, "Hybrid (ML+Pattern)")

    # 4. ML Strategy - Aggressive (lower SL, higher TP)
    ml_aggressive = MLStrategy(
        predictor=predictor,
        history_size=120,
        sl_atr_mult=1.5,
        tp_atr_mult=5.0,
        trailing_atr_mult=1.0,
        min_rr=2.5
    )
    comparator.add_strategy(ml_aggressive, "ML Aggressive")

    print(f"✓ Added {len(comparator.strategies)} strategies\n")

    # Run comparison with visualizations
    result = comparator.run_and_visualize(
        data=data,
        symbol=symbol,
        output_dir="./strategy_comparison",
        verbose=True
    )

    # Print summary
    result.print_summary()

    # Print detailed winner analysis
    print("\n" + "="*80)
    print("WINNER ANALYSIS")
    print("="*80)

    metrics = ['sharpe_ratio', 'profit_factor', 'net_profit', 'win_rate']
    for metric in metrics:
        winner = result.get_winner(metric)
        value = result.comparison_metrics.loc[winner, metric]

        if metric == 'net_profit':
            print(f"  Best {metric.replace('_', ' ').title()}: {winner} (${value:,.2f})")
        elif metric == 'win_rate':
            print(f"  Best {metric.replace('_', ' ').title()}: {winner} ({value:.1f}%)")
        else:
            print(f"  Best {metric.replace('_', ' ').title()}: {winner} ({value:.2f})")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    # Overall recommendation (based on Sharpe ratio)
    best_overall = result.get_winner('sharpe_ratio')
    best_result = result.results[result.strategy_names.index(best_overall)]

    # Calculate net profit percentage
    net_profit_pct = (best_result.net_profit / best_result.initial_balance) * 100

    print(f"\n✨ Best Overall Strategy: {best_overall}")
    print(f"\n  Reasons:")
    print(f"    • Highest Sharpe Ratio: {best_result.sharpe_ratio:.2f}")
    print(f"    • Win Rate: {best_result.win_rate:.1f}%")
    print(f"    • Profit Factor: {best_result.profit_factor:.2f}")
    print(f"    • Net Profit: ${best_result.net_profit:,.2f} ({net_profit_pct:+.2f}%)")
    print(f"    • Max Drawdown: {best_result.max_drawdown_pct:.2f}%")

    print(f"\n  This strategy provides the best risk-adjusted returns.")
    print(f"  Consider using it for live trading after further validation.")

    print("\n" + "="*80)
    print("\n✓ Comparison complete! Check ./strategy_comparison/ for charts.\n")


if __name__ == "__main__":
    main()
