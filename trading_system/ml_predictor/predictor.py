"""
Forex Predictor
===============

Real-time price prediction using trained ONNX models.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import logging
import pickle
import onnxruntime as ort

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForexPredictor:
    """
    Make predictions using trained ONNX models.

    Handles data collection, normalization, inference, and signal generation.
    """

    # Prediction classes
    PRICE_UP = 0
    PRICE_SAME = 1
    PRICE_DOWN = 2

    def __init__(
        self,
        model_path: str,
        scaler_path: Optional[str] = None,
        history_size: int = 120,
        min_max: Optional[Tuple[float, float]] = None
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to ONNX model file
            scaler_path: Path to pickled MinMaxScaler (optional)
            history_size: Number of hours for input sequence
            min_max: Tuple of (min, max) for normalization (optional)
        """
        self.model_path = model_path
        self.history_size = history_size
        self.min_price = None
        self.max_price = None

        # Load ONNX model
        try:
            self.session = ort.InferenceSession(model_path)
            logger.info(f"✓ Loaded ONNX model: {model_path}")
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            raise

        # Load scaler if provided
        if scaler_path:
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"✓ Loaded scaler: {scaler_path}")
            except Exception as e:
                logger.warning(f"Could not load scaler: {e}")
                self.scaler = None
        else:
            self.scaler = None

        # Set min/max if provided
        if min_max:
            self.min_price, self.max_price = min_max

    def update_min_max(self, symbol: str, lookback_hours: int = 2880) -> bool:
        """
        Update min/max prices for normalization.

        Args:
            symbol: Trading symbol
            lookback_hours: Hours to look back (default 120 days = 2880 hours)

        Returns:
            True if successful, False otherwise
        """
        if not MT5_AVAILABLE:
            logger.warning("MT5 not available, using default min/max")
            return False

        if not mt5.initialize():
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return False

        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, lookback_hours)
        mt5.shutdown()

        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get rates for {symbol}")
            return False

        df = pd.DataFrame(rates)
        close_prices = df['close'].values

        self.min_price = float(np.min(close_prices))
        self.max_price = float(np.max(close_prices))

        logger.info(f"Updated min/max: {self.min_price:.5f} - {self.max_price:.5f}")
        return True

    def get_recent_closes(
        self,
        symbol: str,
        count: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Get recent close prices from MT5.

        Args:
            symbol: Trading symbol
            count: Number of bars (uses history_size + 1 if None)

        Returns:
            Array of close prices or None if error
        """
        if not MT5_AVAILABLE:
            logger.warning("MT5 not available")
            return None

        if not mt5.initialize():
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return None

        count = count or (self.history_size + 1)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 1, count)
        mt5.shutdown()

        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get rates for {symbol}")
            return None

        df = pd.DataFrame(rates)
        return df['close'].values

    def normalize_prices(self, prices: np.ndarray) -> np.ndarray:
        """
        Normalize prices to [0, 1] range.

        Args:
            prices: Array of prices

        Returns:
            Normalized prices
        """
        if self.scaler is not None:
            # Use fitted scaler
            return self.scaler.transform(prices.reshape(-1, 1)).flatten()
        elif self.min_price is not None and self.max_price is not None:
            # Use min/max
            if self.max_price <= self.min_price:
                logger.error("Invalid min/max values")
                return prices
            return (prices - self.min_price) / (self.max_price - self.min_price)
        else:
            logger.warning("No normalization method available")
            return prices

    def denormalize_price(self, normalized_price: float) -> float:
        """
        Denormalize a predicted price.

        Args:
            normalized_price: Price in [0, 1] range

        Returns:
            Actual price value
        """
        if self.scaler is not None:
            return float(self.scaler.inverse_transform([[normalized_price]])[0, 0])
        elif self.min_price is not None and self.max_price is not None:
            return float(normalized_price * (self.max_price - self.min_price) + self.min_price)
        else:
            return normalized_price

    def predict(
        self,
        prices: np.ndarray,
        threshold: float = 0.00001
    ) -> Tuple[int, float, float]:
        """
        Make a prediction using the ONNX model.

        Args:
            prices: Array of recent close prices (length = history_size)
            threshold: Minimum price change to consider significant

        Returns:
            Tuple of (prediction_class, predicted_price, last_close)
        """
        if len(prices) < self.history_size:
            logger.error(f"Not enough data: {len(prices)} < {self.history_size}")
            return self.PRICE_SAME, 0.0, 0.0

        # Get last N prices
        recent_prices = prices[-self.history_size:]
        last_close = float(recent_prices[-1])

        # Normalize
        normalized = self.normalize_prices(recent_prices)

        # Prepare input for ONNX model [batch_size, time_steps, features]
        input_data = normalized.reshape(1, self.history_size, 1).astype(np.float32)

        # Run inference
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        result = self.session.run([output_name], {input_name: input_data})
        predicted_normalized = float(result[0][0, 0])

        # Denormalize
        predicted_price = self.denormalize_price(predicted_normalized)

        # Classify movement
        delta = last_close - predicted_price

        if abs(delta) <= threshold:
            prediction_class = self.PRICE_SAME
        elif delta < 0:
            prediction_class = self.PRICE_UP
        else:
            prediction_class = self.PRICE_DOWN

        logger.debug(f"Last: {last_close:.5f}, Predicted: {predicted_price:.5f}, "
                    f"Delta: {delta:.5f}, Class: {prediction_class}")

        return prediction_class, predicted_price, last_close

    def predict_symbol(
        self,
        symbol: str,
        update_minmax: bool = True
    ) -> Tuple[int, float, float]:
        """
        Predict next price movement for a symbol.

        Args:
            symbol: Trading symbol
            update_minmax: Whether to update min/max before prediction

        Returns:
            Tuple of (prediction_class, predicted_price, last_close)
        """
        # Update min/max if needed
        if update_minmax and (self.min_price is None or self.max_price is None):
            self.update_min_max(symbol)

        # Get recent prices
        prices = self.get_recent_closes(symbol)
        if prices is None:
            logger.error(f"Could not get prices for {symbol}")
            return self.PRICE_SAME, 0.0, 0.0

        # Make prediction
        return self.predict(prices)

    def get_signal_name(self, prediction_class: int) -> str:
        """
        Get human-readable signal name.

        Args:
            prediction_class: Prediction class constant

        Returns:
            Signal name string
        """
        if prediction_class == self.PRICE_UP:
            return "BUY"
        elif prediction_class == self.PRICE_DOWN:
            return "SELL"
        else:
            return "NEUTRAL"
