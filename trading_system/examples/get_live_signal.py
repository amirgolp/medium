"""
Get Live Trading Signal
========================

Get real-time trading signal with entry, SL, and TP.
Works without MetaTrader - uses Yahoo Finance data.
"""

import argparse
import logging

from trading_system.live_trading import LiveSignalGenerator

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)


def main():
    """Get live trading signal."""
    parser = argparse.ArgumentParser(
        description="Get live trading signal with entry, SL, and TP"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="EURUSD",
        help="Trading symbol (e.g., EURUSD, XAUUSD, CL)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="hybrid",
        choices=["ml", "pattern", "hybrid"],
        help="Strategy to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to ONNX model (optional, auto-detected)"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Continuously monitor and print signals"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols to monitor (e.g., EURUSD,GBPUSD,XAUUSD)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Monitoring interval in minutes"
    )

    args = parser.parse_args()

    # Create signal generator
    generator = LiveSignalGenerator(
        strategy_type=args.strategy,
        model_path=args.model
    )

    if args.monitor:
        # Monitor mode
        symbols = args.symbols.split(',') if args.symbols else [args.symbol]
        generator.monitor_symbols(symbols, interval_minutes=args.interval)
    else:
        # Single signal mode
        print(f"\n🔍 Generating signal for {args.symbol}...")
        print(f"Strategy: {args.strategy.upper()}")
        print()

        try:
            signal = generator.generate_signal(args.symbol)
            signal.print_signal()

            # Print usage instructions
            if signal.signal != "NEUTRAL":
                print("📝 HOW TO USE THIS SIGNAL:\n")
                print("1. Open your broker platform (any broker)")
                print(f"2. Find {signal.symbol}")
                print(f"3. Place a {signal.signal} order:")
                print(f"   - Entry: Market order at ~{signal.entry_price:.5f}")
                print(f"   - Stop Loss: {signal.stop_loss:.5f}")
                print(f"   - Take Profit: {signal.take_profit:.5f}")

                if signal.trailing_stop:
                    print(f"   - Trailing Stop: {signal.trailing_stop:.5f} pips")

                print(f"4. Position size: {signal.lot_size} lots")
                print(f"5. Risk: ${signal.risk_amount:.2f}\n")
            else:
                print("⏸️  No trading opportunity at this time.")
                print("   Check back later or monitor continuously with --monitor\n")

        except FileNotFoundError:
            print(f"\n❌ Error: Model not found for {args.symbol}")
            print(f"\n💡 Train a model first:")
            print(f"   poe train --symbol {args.symbol} --epochs 200\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
