"""
ATR Calculator
==============

Calculate Average True Range for dynamic stop loss and take profit.
"""

import logging
import numpy as np
from typing import Optional

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ATRCalculator:
    """
    Calculate ATR (Average True Range) for volatility-based risk management.

    ATR is used to set dynamic stops and targets that adapt to market volatility.
    """

    def __init__(self, period: int = 14):
        """
        Initialize ATR calculator.

        Args:
            period: ATR period (default 14)
        """
        self.period = period

    def get_atr_from_mt5(
        self,
        symbol: str,
        timeframe: str = "H1",
        shift: int = 0
    ) -> Optional[float]:
        """
        Get ATR value from MT5 indicator.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            shift: Bar shift (0 = current bar)

        Returns:
            ATR value or None if error
        """
        if not MT5_AVAILABLE:
            logger.warning("MT5 not available")
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

        if not mt5.initialize():
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return None

        # Create ATR indicator
        atr_handle = mt5.copy_rates_from_pos(symbol, tf, shift, self.period + 1)

        if atr_handle is None:
            mt5.shutdown()
            logger.error(f"Failed to get rates for {symbol}")
            return None

        # Calculate ATR manually from rates
        import pandas as pd
        df = pd.DataFrame(atr_handle)

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # Calculate True Range
        tr = self._calculate_true_range(high, low, close)

        # Calculate ATR as simple moving average of TR
        atr = np.mean(tr[-self.period:])

        mt5.shutdown()

        logger.debug(f"ATR({self.period}) for {symbol} {timeframe}: {atr:.5f}")
        return float(atr)

    def _calculate_true_range(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> np.ndarray:
        """
        Calculate True Range.

        TR = max(high - low, |high - prev_close|, |low - prev_close|)

        Args:
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            Array of true range values
        """
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]  # First bar uses its own close

        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)

        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        return tr

    def calculate_atr_from_data(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> float:
        """
        Calculate ATR from price arrays.

        Args:
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            ATR value
        """
        if len(high) < self.period + 1:
            logger.error(f"Not enough data: {len(high)} < {self.period + 1}")
            return 0.0

        tr = self._calculate_true_range(high, low, close)
        atr = np.mean(tr[-self.period:])

        return float(atr)

    def calculate_sl_tp(
        self,
        symbol: str,
        signal: str,
        sl_multiplier: float = 2.5,
        tp_multiplier: float = 3.0,
        timeframe: str = "H1"
    ) -> tuple:
        """
        Calculate stop loss and take profit based on ATR.

        Args:
            symbol: Trading symbol
            signal: "BUY" or "SELL"
            sl_multiplier: ATR multiplier for stop loss
            tp_multiplier: ATR multiplier for take profit
            timeframe: Timeframe string

        Returns:
            Tuple of (sl_price, tp_price, atr_value)
        """
        atr = self.get_atr_from_mt5(symbol, timeframe)

        if atr is None:
            logger.error("Could not calculate ATR")
            return None, None, None

        if not MT5_AVAILABLE:
            # Use mock price
            current_price = 1.1000
        else:
            if not mt5.initialize():
                return None, None, None

            tick = mt5.symbol_info_tick(symbol)
            mt5.shutdown()

            if tick is None:
                return None, None, None

            current_price = tick.bid if signal == "BUY" else tick.ask

        # Calculate SL and TP
        if signal == "BUY":
            sl_price = current_price - (atr * sl_multiplier)
            tp_price = current_price + (atr * tp_multiplier)
        elif signal == "SELL":
            sl_price = current_price + (atr * sl_multiplier)
            tp_price = current_price - (atr * tp_multiplier)
        else:
            logger.error(f"Invalid signal: {signal}")
            return None, None, None

        logger.info(f"ATR-based levels: Entry={current_price:.5f}, "
                   f"SL={sl_price:.5f}, TP={tp_price:.5f}, ATR={atr:.5f}")

        return sl_price, tp_price, atr

    def get_sl_distance_pips(
        self,
        symbol: str,
        sl_multiplier: float = 2.5,
        timeframe: str = "H1"
    ) -> Optional[float]:
        """
        Get stop loss distance in pips based on ATR.

        Args:
            symbol: Trading symbol
            sl_multiplier: ATR multiplier
            timeframe: Timeframe string

        Returns:
            SL distance in pips
        """
        atr = self.get_atr_from_mt5(symbol, timeframe)

        if atr is None:
            return None

        # Convert ATR to pips
        pip_value = 0.0001 if "JPY" not in symbol else 0.01
        sl_pips = (atr * sl_multiplier) / pip_value

        logger.debug(f"ATR SL distance: {sl_pips:.1f} pips")
        return sl_pips
