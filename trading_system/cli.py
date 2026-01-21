#!/usr/bin/env python3
"""
Trading System CLI
==================

Command-line interface for the trading system.

Commands:
    train       - Train ML models
    predict     - Make predictions
    sentiment   - Analyze sentiment
    backtest    - Run backtest
    optimize    - Optimize parameters
"""

import argparse
import sys
import logging
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_command(args):
    """Train ML models."""
    from trading_system import ModelTrainer

    print("="*60)
    print("TRAINING ML MODELS")
    print("="*60)

    symbols = args.symbols.split(',')
    print(f"Symbols: {', '.join(symbols)}")
    print(f"History size: {args.history_size}")
    print(f"Training days: {args.training_days}")
    print(f"Epochs: {args.epochs}")
    print("="*60 + "\n")

    trainer = ModelTrainer(
        history_size=args.history_size,
        training_days=args.training_days,
        early_stop_patience=args.patience
    )

    results = trainer.train_multiple_symbols(
        symbols=symbols,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1 if args.verbose else 0
    )

    # Save models
    for symbol, data in results.items():
        if data['model']:
            trainer.save_model_onnx(
                data['model'],
                symbol,
                output_dir=args.output_dir
            )

    print(f"\n✅ Training complete! Models saved to {args.output_dir}/")


def predict_command(args):
    """Make predictions."""
    from trading_system import ForexPredictor

    print("="*60)
    print("MAKING PREDICTION")
    print("="*60)

    model_path = f"{args.model_dir}/model.{args.symbol}.H1.{args.history_size}.onnx"

    predictor = ForexPredictor(model_path, history_size=args.history_size)
    predictor.update_min_max(args.symbol, lookback_hours=args.lookback_hours)

    prediction, predicted_price, last_close = predictor.predict_symbol(args.symbol)
    signal = predictor.get_signal_name(prediction)

    change = predicted_price - last_close
    change_pct = (change / last_close) * 100

    print(f"\nSymbol: {args.symbol}")
    print(f"Current: {last_close:.5f}")
    print(f"Predicted: {predicted_price:.5f}")
    print(f"Change: {change:+.5f} ({change_pct:+.2f}%)")
    print(f"Signal: {signal}")
    print("="*60)


def sentiment_command(args):
    """Analyze sentiment."""
    from trading_system import SentimentAnalyzer
    from trading_system.sentiment_analyzer import NewsAPIClient, FinnhubClient
    import os

    print("="*60)
    print("SENTIMENT ANALYSIS")
    print("="*60)

    analyzer = SentimentAnalyzer(
        lookback_hours=args.lookback_hours,
        min_sentiment_score=args.min_score
    )

    # Fetch from NewsAPI
    newsapi_key = os.getenv("NEWSAPI_KEY", args.newsapi_key)
    if newsapi_key:
        client = NewsAPIClient(newsapi_key)
        articles = client.fetch_news(hours_back=args.lookback_hours)
        sentiments = client.parse_to_sentiment(articles, analyzer)
        for s in sentiments:
            analyzer.add_sentiment(s)

    # Fetch from Finnhub
    finnhub_key = os.getenv("FINNHUB_KEY", args.finnhub_key)
    if finnhub_key:
        client = FinnhubClient(finnhub_key)
        news = client.fetch_forex_news()
        sentiments = client.parse_to_sentiment(news, analyzer)
        for s in sentiments:
            analyzer.add_sentiment(s)

    # Generate signal
    base, quote = args.pair[:3], args.pair[3:]
    signal, confidence, details = analyzer.generate_signal(base, quote)

    print(f"\nPair: {args.pair}")
    if signal:
        print(f"Signal: {signal}")
        print(f"Confidence: {confidence:.1f}")
    else:
        print("No signal")

    print("="*60)


def backtest_command(args):
    """Run backtest."""
    from trading_system.backtest import BacktestEngine, BacktestVisualizer
    from trading_system import ForexPredictor
    import numpy as np

    print("="*60)
    print("RUNNING BACKTEST")
    print("="*60)
    print(f"Symbol: {args.symbol}")
    print(f"Days: {args.days}")
    print(f"Initial balance: ${args.balance:,.2f}")
    print("="*60 + "\n")

    # Initialize engine
    engine = BacktestEngine(initial_balance=args.balance)

    # Load data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    data = engine.load_data(args.symbol, "H1", start_date, end_date)

    print(f"Loaded {len(data)} bars\n")

    # Load predictor if model exists
    predictor = None
    try:
        model_path = f"{args.model_dir}/model.{args.symbol}.H1.120.onnx"
        predictor = ForexPredictor(model_path, history_size=120)
        predictor.min_price = data['close'].min()
        predictor.max_price = data['close'].max()
        print("✅ Loaded ML predictor")
    except:
        print("⚠️  ML predictor not available, using simple strategy")

    # Simple strategy function
    def simple_strategy(data, i, predictor=None, symbol=None):
        if i < 120:
            return None

        # Get recent prices
        recent = data.iloc[i-120:i]['close'].values

        if predictor:
            try:
                # Use ML prediction
                prediction, pred_price, last_close = predictor.predict(recent)
                signal = predictor.get_signal_name(prediction)

                if signal == "NEUTRAL":
                    return None

                # Calculate ATR for stops
                highs = data.iloc[i-14:i]['high'].values
                lows = data.iloc[i-14:i]['low'].values
                closes = data.iloc[i-14:i]['close'].values

                atr = np.mean(np.maximum(
                    highs - lows,
                    np.maximum(
                        np.abs(highs - np.roll(closes, 1)),
                        np.abs(lows - np.roll(closes, 1))
                    )
                ))

                current_price = data.iloc[i]['close']

                if signal == "BUY":
                    sl = current_price - (atr * 2.5)
                    tp = current_price + (atr * 3.0)
                else:
                    sl = current_price + (atr * 2.5)
                    tp = current_price - (atr * 3.0)

                return {
                    'signal': signal,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'lot_size': 0.1,
                    'ml_signal': signal
                }
            except:
                pass

        # Fallback: Simple MA crossover
        ma_short = np.mean(recent[-20:])
        ma_long = np.mean(recent[-50:])

        if ma_short > ma_long:
            return {
                'signal': 'BUY',
                'stop_loss': data.iloc[i]['close'] * 0.995,
                'take_profit': data.iloc[i]['close'] * 1.01,
                'lot_size': 0.1
            }
        elif ma_short < ma_long:
            return {
                'signal': 'SELL',
                'stop_loss': data.iloc[i]['close'] * 1.005,
                'take_profit': data.iloc[i]['close'] * 0.99,
                'lot_size': 0.1
            }

        return None

    # Run backtest
    result = engine.run_backtest(
        data=data,
        strategy_func=simple_strategy,
        symbol=args.symbol,
        predictor=predictor
    )

    # Print results
    result.print_summary()

    # Visualize if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        visualizer = BacktestVisualizer()

        if args.save_charts:
            visualizer.create_full_report(result, data, args.output_dir)
        else:
            visualizer.plot_all(result, data)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Trading System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train ML models')
    train_parser.add_argument('--symbols', default='EURUSD,GBPUSD', help='Comma-separated symbols')
    train_parser.add_argument('--history-size', type=int, default=120, help='History size')
    train_parser.add_argument('--training-days', type=int, default=120, help='Training days')
    train_parser.add_argument('--epochs', type=int, default=300, help='Max epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--patience', type=int, default=20, help='Early stop patience')
    train_parser.add_argument('--output-dir', default='./models', help='Output directory')
    train_parser.add_argument('--verbose', action='store_true', help='Verbose output')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--symbol', default='EURUSD', help='Trading symbol')
    predict_parser.add_argument('--history-size', type=int, default=120, help='History size')
    predict_parser.add_argument('--lookback-hours', type=int, default=2880, help='Lookback hours for normalization')
    predict_parser.add_argument('--model-dir', default='./models', help='Model directory')

    # Sentiment command
    sentiment_parser = subparsers.add_parser('sentiment', help='Analyze sentiment')
    sentiment_parser.add_argument('--pair', default='EURUSD', help='Currency pair')
    sentiment_parser.add_argument('--lookback-hours', type=int, default=24, help='Lookback hours')
    sentiment_parser.add_argument('--min-score', type=float, default=10.0, help='Min sentiment score')
    sentiment_parser.add_argument('--newsapi-key', default='', help='NewsAPI key')
    sentiment_parser.add_argument('--finnhub-key', default='', help='Finnhub key')

    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--symbol', default='EURUSD', help='Trading symbol')
    backtest_parser.add_argument('--days', type=int, default=90, help='Days to backtest')
    backtest_parser.add_argument('--balance', type=float, default=10000, help='Initial balance')
    backtest_parser.add_argument('--model-dir', default='./models', help='Model directory')
    backtest_parser.add_argument('--visualize', action='store_true', help='Show visualizations')
    backtest_parser.add_argument('--save-charts', action='store_true', help='Save chart images')
    backtest_parser.add_argument('--output-dir', default='./backtest_results', help='Output directory')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to command
    if args.command == 'train':
        train_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'sentiment':
        sentiment_command(args)
    elif args.command == 'backtest':
        backtest_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
