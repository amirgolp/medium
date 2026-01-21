"""
Example: Backtest Combined Strategy
====================================

Backtest strategy combining ML predictions and sentiment analysis.
"""

from trading_system.backtest import BacktestEngine, BacktestVisualizer
from trading_system import ForexPredictor, SentimentAnalyzer
from trading_system.risk_management import ATRCalculator
from trading_system.sentiment_analyzer import SentimentScore
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class CombinedStrategy:
    """Combined ML + Sentiment strategy."""

    def __init__(self, predictor, atr_calc, sentiment_analyzer, symbol):
        self.predictor = predictor
        self.atr_calc = atr_calc
        self.sentiment_analyzer = sentiment_analyzer
        self.symbol = symbol

        # Simulate sentiment data (in real backtest, load from historical news)
        self._simulate_sentiment_data()

    def _simulate_sentiment_data(self):
        """
        Simulate historical sentiment data.

        In production, this would load actual historical news.
        """
        # Generate random sentiment scores for demonstration
        np.random.seed(42)

        for i in range(100):
            time = datetime.now() - timedelta(hours=i)

            # EUR sentiment
            eur_sentiment = SentimentScore(
                time=time,
                currency="EUR",
                event_name="Economic Data",
                score=50 + np.random.normal(0, 15),
                source="Historical",
                impact="Medium"
            )
            self.sentiment_analyzer.add_sentiment(eur_sentiment)

            # USD sentiment
            usd_sentiment = SentimentScore(
                time=time,
                currency="USD",
                event_name="Economic Data",
                score=50 + np.random.normal(0, 15),
                source="Historical",
                impact="Medium"
            )
            self.sentiment_analyzer.add_sentiment(usd_sentiment)

    def __call__(self, data, i):
        """
        Generate trading signal.

        Args:
            data: Price data
            i: Current bar index

        Returns:
            Signal dictionary or None
        """
        if i < 120:
            return None

        # Get recent prices
        recent_prices = data.iloc[i-120:i]['close'].values

        try:
            # 1. Get ML prediction
            prediction, pred_price, last_close = self.predictor.predict(recent_prices)
            ml_signal = self.predictor.get_signal_name(prediction)

            if ml_signal == "NEUTRAL":
                return None

            # 2. Get sentiment signal
            sent_signal, confidence, details = self.sentiment_analyzer.generate_signal("EUR", "USD")

            # 3. Combine signals - only trade if aligned
            if not sent_signal or ml_signal != sent_signal:
                return None

            # Both signals agree - proceed with trade
            logging.info(f"Bar {i}: ML={ml_signal}, Sentiment={sent_signal} (conf={confidence:.1f})")

            # Calculate ATR
            highs = data.iloc[i-14:i]['high'].values
            lows = data.iloc[i-14:i]['low'].values
            closes = data.iloc[i-14:i]['close'].values
            atr = self.atr_calc.calculate_atr_from_data(highs, lows, closes)

            # Get current price
            current_price = data.iloc[i]['close']

            # Calculate stops
            if ml_signal == "BUY":
                sl = current_price - (atr * 2.5)
                tp = current_price + (atr * 3.0)
            else:
                sl = current_price + (atr * 2.5)
                tp = current_price - (atr * 3.0)

            return {
                'signal': ml_signal,
                'stop_loss': sl,
                'take_profit': tp,
                'lot_size': 0.1,
                'ml_signal': ml_signal,
                'sentiment_signal': sent_signal,
                'sentiment_score': confidence
            }

        except Exception as e:
            logging.warning(f"Strategy error at bar {i}: {e}")
            return None


def main():
    """Run combined strategy backtest."""

    symbol = "EURUSD"
    days = 90
    initial_balance = 10000.0

    print("="*70)
    print("BACKTEST: COMBINED ML + SENTIMENT STRATEGY")
    print("="*70)
    print(f"Symbol: {symbol}")
    print(f"Period: {days} days")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Strategy: ML predictions + Sentiment analysis (aligned signals only)")
    print("="*70 + "\n")

    # Initialize components
    engine = BacktestEngine(initial_balance=initial_balance, slippage_pips=1.0)

    # Load data
    print("Loading historical data...")
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
        print("Run 'python trading_system/examples/train_models.py' first\n")
        return

    # Initialize sentiment analyzer
    print("Initializing sentiment analyzer...")
    sentiment_analyzer = SentimentAnalyzer(
        lookback_hours=24,
        min_sentiment_score=10.0
    )
    print(f"✓ Sentiment analyzer ready\n")

    # Initialize ATR calculator
    atr_calc = ATRCalculator(period=14)

    # Create combined strategy
    strategy = CombinedStrategy(predictor, atr_calc, sentiment_analyzer, symbol)

    # Run backtest
    print("Running backtest...")
    print("This may take a moment...\n")

    result = engine.run_backtest(
        data=data,
        strategy_func=strategy,
        symbol=symbol
    )

    # Print results
    result.print_summary()

    # Additional analysis
    if result.trades:
        print("\n" + "="*70)
        print("SIGNAL ANALYSIS")
        print("="*70)

        ml_only = sum(1 for t in result.trades if t.ml_signal and not t.sentiment_signal)
        sentiment_only = sum(1 for t in result.trades if t.sentiment_signal and not t.ml_signal)
        combined = sum(1 for t in result.trades if t.ml_signal and t.sentiment_signal)

        print(f"\nTrades by signal type:")
        print(f"  ML only:        {ml_only}")
        print(f"  Sentiment only: {sentiment_only}")
        print(f"  Combined:       {combined}")

        if combined > 0:
            combined_trades = [t for t in result.trades if t.ml_signal and t.sentiment_signal]
            combined_pnl = sum(t.profit_loss for t in combined_trades)
            combined_win_rate = sum(1 for t in combined_trades if t.profit_loss > 0) / len(combined_trades) * 100

            print(f"\nCombined signal performance:")
            print(f"  Trades: {len(combined_trades)}")
            print(f"  Win Rate: {combined_win_rate:.1f}%")
            print(f"  Total P&L: ${combined_pnl:,.2f}")

    # Visualize
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    try:
        visualizer = BacktestVisualizer()
        visualizer.create_full_report(result, data, output_dir="./backtest_results_combined")
        print("\n✓ Charts saved to ./backtest_results_combined/")
    except ImportError:
        print("\n⚠️  Matplotlib not installed")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
