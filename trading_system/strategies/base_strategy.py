"""
Base Strategy Class
===================

Abstract base class for all trading strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class StrategySignal:
    """Trading signal with all necessary information."""
    signal: str  # "BUY", "SELL", or "NEUTRAL"
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    trailing_stop: Optional[float] = None

    # Metadata for analysis
    confidence: float = 0.0
    pattern_type: Optional[str] = None
    entry_zone: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary format (for backtest engine)."""
        result = {
            'signal': self.signal,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'lot_size': self.lot_size,
        }

        if self.trailing_stop:
            result['trailing_stop'] = self.trailing_stop

        # Add metadata fields
        result.update(self.metadata)

        return result

    def calculate_risk_reward(self) -> float:
        """Calculate risk/reward ratio."""
        if self.signal == "BUY":
            risk = abs(self.entry_price - self.stop_loss)
            reward = abs(self.take_profit - self.entry_price)
        elif self.signal == "SELL":
            risk = abs(self.stop_loss - self.entry_price)
            reward = abs(self.entry_price - self.take_profit)
        else:
            return 0.0

        return reward / risk if risk > 0 else 0.0


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    All strategies must implement:
    - generate_signal(): Generate trading signal from data
    - get_name(): Return strategy name
    - get_description(): Return strategy description
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize strategy.

        Args:
            name: Optional custom name for the strategy
        """
        self._custom_name = name
        self._setup_complete = False

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, index: int) -> Optional[StrategySignal]:
        """
        Generate trading signal at given data index.

        Args:
            data: OHLC DataFrame with 'open', 'high', 'low', 'close' columns
            index: Current bar index in the DataFrame

        Returns:
            StrategySignal object if signal generated, None otherwise
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get strategy name.

        Returns:
            Strategy name
        """
        pass

    def get_description(self) -> str:
        """
        Get strategy description.

        Returns:
            Strategy description
        """
        return "No description available"

    def setup(self, data: pd.DataFrame) -> None:
        """
        Optional setup method called before backtesting.

        Use this to pre-calculate indicators, load models, etc.

        Args:
            data: Full OHLC DataFrame
        """
        self._setup_complete = True

    def calculate_atr(self, data: pd.DataFrame, index: int, period: int = 14) -> float:
        """
        Helper method to calculate ATR at given index.

        Args:
            data: OHLC DataFrame
            index: Current bar index
            period: ATR period

        Returns:
            ATR value
        """
        if index < period:
            return 0.0

        highs = data.iloc[index-period:index]['high'].values
        lows = data.iloc[index-period:index]['low'].values
        closes = data.iloc[index-period:index]['close'].values

        high_low = highs - lows
        high_close = np.abs(highs - np.roll(closes, 1))
        low_close = np.abs(lows - np.roll(closes, 1))

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))

        return np.mean(true_range[1:])  # Skip first value (roll artifact)

    def calculate_ma(self, data: pd.DataFrame, index: int, period: int,
                     column: str = 'close') -> float:
        """
        Helper method to calculate moving average.

        Args:
            data: OHLC DataFrame
            index: Current bar index
            period: MA period
            column: Column to calculate MA on

        Returns:
            MA value
        """
        if index < period:
            return 0.0

        return np.mean(data.iloc[index-period:index][column].values)

    def is_valid_signal(self, signal: StrategySignal) -> bool:
        """
        Validate signal before execution.

        Args:
            signal: Signal to validate

        Returns:
            True if valid, False otherwise
        """
        if signal.signal not in ["BUY", "SELL"]:
            return False

        if signal.entry_price <= 0:
            return False

        if signal.stop_loss <= 0 or signal.take_profit <= 0:
            return False

        if signal.lot_size <= 0:
            return False

        # Check risk/reward ratio
        rr = signal.calculate_risk_reward()
        if rr < 1.0:  # Minimum 1:1 risk/reward
            return False

        return True

    def __call__(self, data: pd.DataFrame, index: int) -> Optional[Dict[str, Any]]:
        """
        Make strategy callable for backtest engine compatibility.

        Args:
            data: OHLC DataFrame
            index: Current bar index

        Returns:
            Signal dictionary for backtest engine
        """
        signal = self.generate_signal(data, index)

        if signal is None:
            return None

        if not self.is_valid_signal(signal):
            return None

        return signal.to_dict()

    def __str__(self) -> str:
        """String representation."""
        return self._custom_name if self._custom_name else self.get_name()

    def __repr__(self) -> str:
        """Representation."""
        return f"{self.__class__.__name__}(name='{str(self)}')"
