"""
Model Trainer for Forex Prediction
===================================

Implements CNN-LSTM hybrid neural network for price prediction.
Weekly retraining with fresh market data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
import logging

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logging.warning("MetaTrader5 not available. Using mock data for training.")

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, LSTM
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import tf2onnx
import onnx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Train CNN-LSTM models for forex price prediction.

    Architecture:
        Input (120 hours) → Conv1D (256 filters) → MaxPooling →
        LSTM (100 units) → Dropout → LSTM (100 units) → Dropout →
        Dense (sigmoid) → Output (predicted price)
    """

    def __init__(
        self,
        history_size: int = 120,
        training_days: int = 120,
        test_split: float = 0.2,
        early_stop_patience: int = 20
    ):
        """
        Initialize the model trainer.

        Args:
            history_size: Number of hours to look back for prediction
            training_days: Days of historical data to use for training
            test_split: Fraction of data to use for validation
            early_stop_patience: Epochs to wait before early stopping
        """
        self.history_size = history_size
        self.training_days = training_days
        self.test_split = test_split
        self.early_stop_patience = early_stop_patience
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def split_sequence(self, sequence: np.ndarray, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split a univariate sequence into samples for supervised learning.

        Args:
            sequence: Time series data
            n_steps: Number of time steps per sample

        Returns:
            Tuple of (X, y) arrays for training
        """
        X, y = [], []
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence) - 1:
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def get_market_data(
        self,
        symbol: str,
        timeframe: str = "H1",
        days: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical market data from MetaTrader 5.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: Timeframe string (default "H1")
            days: Number of days to fetch (uses self.training_days if None)

        Returns:
            DataFrame with OHLCV data or None if error
        """
        if not MT5_AVAILABLE:
            logger.warning("MetaTrader5 not available, returning mock data")
            return self._generate_mock_data(days or self.training_days)

        if not mt5.initialize():
            logger.error(f"MT5 initialize() failed: {mt5.last_error()}")
            return None

        # Map timeframe string to MT5 constant
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

        tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
        days = days or self.training_days

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        logger.info(f"Fetching {symbol} data from {start_date} to {end_date}")

        rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
        mt5.shutdown()

        if rates is None or len(rates) == 0:
            logger.error(f"No data retrieved for {symbol}")
            return None

        df = pd.DataFrame(rates)
        logger.info(f"Retrieved {len(df)} bars for {symbol}")
        return df

    def _generate_mock_data(self, days: int) -> pd.DataFrame:
        """Generate mock price data for testing without MT5."""
        np.random.seed(42)
        n_bars = days * 24  # Hourly data

        # Generate realistic-looking price data with trend
        base_price = 1.1000
        trend = np.linspace(0, 0.02, n_bars)
        noise = np.random.normal(0, 0.001, n_bars)
        close_prices = base_price + trend + noise.cumsum()

        df = pd.DataFrame({
            'time': pd.date_range(end=datetime.now(), periods=n_bars, freq='H'),
            'open': close_prices + np.random.normal(0, 0.0001, n_bars),
            'high': close_prices + np.abs(np.random.normal(0, 0.0002, n_bars)),
            'low': close_prices - np.abs(np.random.normal(0, 0.0002, n_bars)),
            'close': close_prices,
            'tick_volume': np.random.randint(100, 1000, n_bars)
        })

        return df

    def build_model(self) -> keras.Model:
        """
        Build the CNN-LSTM hybrid model.

        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Conv1D(
                filters=256,
                kernel_size=2,
                activation='relu',
                padding='same',
                input_shape=(self.history_size, 1)
            ),
            MaxPooling1D(pool_size=2),
            LSTM(100, return_sequences=True),
            Dropout(0.3),
            LSTM(100, return_sequences=False),
            Dropout(0.3),
            Dense(units=1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=[RootMeanSquaredError()]
        )

        return model

    def train_model(
        self,
        symbol: str,
        epochs: int = 300,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Tuple[Optional[keras.Model], Optional[MinMaxScaler], dict]:
        """
        Train a model for a specific symbol.

        Args:
            symbol: Trading symbol to train on
            epochs: Maximum training epochs
            batch_size: Training batch size
            verbose: Verbosity level (0, 1, or 2)

        Returns:
            Tuple of (model, scaler, history_dict)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training model for {symbol}...")
        logger.info(f"{'='*60}\n")

        # Get market data
        df = self.get_market_data(symbol)
        if df is None:
            logger.error(f"Failed to get data for {symbol}")
            return None, None, {}

        # Extract close prices
        data = df[['close']].values

        # Scale data
        scaled_data = self.scaler.fit_transform(data)

        # Split into train/test
        training_size = int(len(scaled_data) * (1 - self.test_split))
        train_data = scaled_data[:training_size, :]
        test_data = scaled_data[training_size:, :]

        logger.info(f"Training size: {training_size}, Test size: {len(test_data)}")

        # Create sequences
        x_train, y_train = self.split_sequence(train_data, self.history_size)
        x_test, y_test = self.split_sequence(test_data, self.history_size)

        # Reshape for LSTM [samples, time steps, features]
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        logger.info(f"X_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"X_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

        # Build model
        model = self.build_model()

        # Setup early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=self.early_stop_patience,
            restore_best_weights=True,
            verbose=1
        )

        # Train model
        logger.info(f"\nTraining model for {symbol}...")
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            validation_data=(x_test, y_test),
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=verbose
        )

        # Evaluate
        train_loss, train_rmse = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
        test_loss, test_rmse = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)

        logger.info(f"\nTraining Results:")
        logger.info(f"  Train Loss: {train_loss:.6f}, Train RMSE: {train_rmse:.6f}")
        logger.info(f"  Test Loss: {test_loss:.6f}, Test RMSE: {test_rmse:.6f}")

        return model, self.scaler, history.history

    def save_model_onnx(
        self,
        model: keras.Model,
        symbol: str,
        timeframe: str = "H1",
        output_dir: str = "./models"
    ) -> str:
        """
        Save trained model in ONNX format.

        Args:
            model: Trained Keras model
            symbol: Trading symbol
            timeframe: Timeframe string
            output_dir: Directory to save model

        Returns:
            Path to saved ONNX model
        """
        import os
        import shutil
        import subprocess

        os.makedirs(output_dir, exist_ok=True)

        model_name = f"model.{symbol}.{timeframe}.{self.history_size}.onnx"
        temp_model_path = os.path.join(output_dir, "temp_model")
        output_path = os.path.join(output_dir, model_name)

        # Remove temp directory if exists
        if os.path.exists(temp_model_path):
            shutil.rmtree(temp_model_path)

        # Export as SavedModel
        model.export(temp_model_path)

        # Convert to ONNX
        cmd = f'python -m tf2onnx.convert --saved-model "{temp_model_path}" --output "{output_path}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"✓ Model saved: {output_path}")
        else:
            logger.error(f"✗ Error saving model: {result.stderr}")

        # Clean up
        if os.path.exists(temp_model_path):
            shutil.rmtree(temp_model_path)

        return output_path

    def train_multiple_symbols(
        self,
        symbols: List[str],
        **kwargs
    ) -> dict:
        """
        Train models for multiple symbols.

        Args:
            symbols: List of trading symbols
            **kwargs: Additional arguments passed to train_model

        Returns:
            Dictionary mapping symbols to (model, scaler, history)
        """
        results = {}

        for symbol in symbols:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {symbol}...")
            logger.info(f"{'='*60}\n")

            try:
                model, scaler, history = self.train_model(symbol, **kwargs)

                if model is not None:
                    results[symbol] = {
                        'model': model,
                        'scaler': scaler,
                        'history': history
                    }
                    logger.info(f"✓ {symbol} completed successfully!\n")
                else:
                    logger.error(f"✗ {symbol} training failed\n")

            except Exception as e:
                logger.error(f"✗ Error processing {symbol}: {str(e)}\n")
                continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Training completed for {len(results)}/{len(symbols)} symbols")
        logger.info(f"{'='*60}")

        return results
