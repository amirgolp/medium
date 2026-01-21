"""
Example: Make Forex Predictions
================================

Use trained models to predict forex price movements.
"""

from trading_system import ForexPredictor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Make predictions for EURUSD."""

    symbol = "EURUSD"
    model_path = f"./models/model.{symbol}.H1.120.onnx"

    print("="*60)
    print("FOREX PRICE PREDICTION")
    print("="*60)
    print(f"Symbol: {symbol}")
    print(f"Model: {model_path}")
    print("="*60)

    # Initialize predictor
    print("\nLoading model...")
    predictor = ForexPredictor(
        model_path=model_path,
        history_size=120
    )

    # Update min/max for normalization (last 120 days = 2880 hours)
    print("Updating normalization parameters...")
    predictor.update_min_max(symbol, lookback_hours=2880)

    print(f"Min: {predictor.min_price:.5f}")
    print(f"Max: {predictor.max_price:.5f}")

    # Make prediction
    print("\nMaking prediction...")
    prediction_class, predicted_price, last_close = predictor.predict_symbol(symbol)

    # Get signal name
    signal = predictor.get_signal_name(prediction_class)

    # Calculate change
    price_change = predicted_price - last_close
    price_change_pct = (price_change / last_close) * 100

    # Print results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Current Price:    {last_close:.5f}")
    print(f"Predicted Price:  {predicted_price:.5f}")
    print(f"Price Change:     {price_change:+.5f} ({price_change_pct:+.2f}%)")
    print(f"Signal:           {signal}")
    print("="*60)

    # Trading recommendation
    print("\nTRADING RECOMMENDATION:")
    if signal == "BUY":
        print("  📈 Consider opening LONG position")
        print("  💡 Use ATR-based stop loss and take profit")
        print("  ⚠️  Confirm with sentiment analysis before trading")
    elif signal == "SELL":
        print("  📉 Consider opening SHORT position")
        print("  💡 Use ATR-based stop loss and take profit")
        print("  ⚠️  Confirm with sentiment analysis before trading")
    else:
        print("  ➡️  No clear signal, wait for better opportunity")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
