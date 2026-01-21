"""
Example: Train ML Models
=========================

Train CNN-LSTM models for multiple currency pairs with weekly retraining.
"""

from trading_system import ModelTrainer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Train models for multiple currency pairs."""

    # List of symbols to train
    symbols = ["EURUSD", "GBPUSD", "USDCHF"]

    print("="*60)
    print("ML MODEL TRAINING")
    print("="*60)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"History Size: 120 hours")
    print(f"Training Days: 120 days")
    print("="*60)

    # Initialize trainer with optimized parameters
    trainer = ModelTrainer(
        history_size=120,          # 120 hours of price history
        training_days=120,         # Use last 120 days of data
        test_split=0.2,           # 80/20 train/test split
        early_stop_patience=20    # Stop if no improvement for 20 epochs
    )

    # Train models for all symbols
    results = trainer.train_multiple_symbols(
        symbols=symbols,
        epochs=300,               # Max 300 epochs
        batch_size=32,
        verbose=1
    )

    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)

    # Save models in ONNX format
    for symbol, data in results.items():
        if data['model'] is not None:
            model = data['model']
            history = data['history']

            # Save as ONNX
            output_path = trainer.save_model_onnx(
                model=model,
                symbol=symbol,
                timeframe="H1",
                output_dir="./models"
            )

            # Print training summary
            final_val_loss = history['val_loss'][-1]
            final_val_rmse = history['val_root_mean_squared_error'][-1]

            print(f"\n{symbol}:")
            print(f"  Model: {output_path}")
            print(f"  Final Val Loss: {final_val_loss:.6f}")
            print(f"  Final Val RMSE: {final_val_rmse:.6f}")
            print(f"  Epochs Trained: {len(history['loss'])}")

    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Successfully trained {len(results)} models")
    print("Models saved in ./models/ directory")
    print("\nNext steps:")
    print("1. Test predictions using predict_forex.py")
    print("2. Deploy models to MetaTrader 5")
    print("="*60)


if __name__ == "__main__":
    main()
