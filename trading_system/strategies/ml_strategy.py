"""
ML Strategy
===========

Optimized machine learning strategy using CNN-LSTM predictions.
"""

from typing import Optional
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, StrategySignal


class MLStrategy(BaseStrategy):
    """
    Machine learning strategy with optimizations:
    - 200-period MA trend filter
    - Time-based trading hours (11:00-22:00 GMT)
    - Risk/reward ratio 1:2 (SL: 2.0 ATR, TP: 4.0 ATR)
    - Trailing stop at 1.0 ATR
    """

    def __init__(self, predictor, history_size: int = 120,
                 sl_atr_mult: float = 2.0, tp_atr_mult: float = 4.0,
                 trailing_atr_mult: float = 1.0, min_rr: float = 2.0):
        """
        Initialize ML strategy.

        Args:
            predictor: ForexPredictor instance
            history_size: History bars for prediction
            sl_atr_mult: Stop loss ATR multiplier
            tp_atr_mult: Take profit ATR multiplier
            trailing_atr_mult: Trailing stop ATR multiplier
            min_rr: Minimum risk/reward ratio
        """
        super().__init__()
        self.predictor = predictor
        self.history_size = history_size
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.trailing_atr_mult = trailing_atr_mult
        self.min_rr = min_rr

    def get_name(self) -> str:
        return "ML Strategy (Optimized)"

    def get_description(self) -> str:
        return (
            f"CNN-LSTM predictions with 200-MA trend filter, "
            f"trading hours 11:00-22:00 GMT, "
            f"SL={self.sl_atr_mult} ATR, TP={self.tp_atr_mult} ATR, "
            f"trailing={self.trailing_atr_mult} ATR"
        )

    def generate_signal(self, data: pd.DataFrame, index: int) -> Optional[StrategySignal]:
        """Generate ML-based signal."""

        # Need enough history for MA filter and prediction
        if index < 200:
            return None

        # Get recent prices for prediction
        recent_prices = data.iloc[index-self.history_size:index]['close'].values

        try:
            # 1. Get ML prediction
            prediction, pred_price, last_close = self.predictor.predict(recent_prices)
            signal = self.predictor.get_signal_name(prediction)

            if signal == "NEUTRAL":
                return None

            # 2. Trend filter - Check 200-period MA
            ma_200 = self.calculate_ma(data, index, period=200, column='close')
            current_price = data.iloc[index]['close']

            # Don't trade against major trend
            if signal == "BUY" and current_price < ma_200:
                return None
            if signal == "SELL" and current_price > ma_200:
                return None

            # 3. Time filter - Only trade during high liquidity
            bar_time = data.iloc[index]['time']
            trade_hour = bar_time.hour
            if trade_hour < 11 or trade_hour >= 22:
                return None

            # 4. Calculate ATR
            atr = self.calculate_atr(data, index, period=14)
            if atr <= 0:
                return None

            # 5. Calculate stops
            if signal == "BUY":
                sl = current_price - (atr * self.sl_atr_mult)
                tp = current_price + (atr * self.tp_atr_mult)
            else:
                sl = current_price + (atr * self.sl_atr_mult)
                tp = current_price - (atr * self.tp_atr_mult)

            trailing = atr * self.trailing_atr_mult

            # Create signal
            strategy_signal = StrategySignal(
                signal=signal,
                entry_price=current_price,
                stop_loss=sl,
                take_profit=tp,
                lot_size=0.1,
                trailing_stop=trailing,
                confidence=0.0,  # ML doesn't provide confidence score directly
                metadata={
                    'ml_signal': signal,
                    'predicted_price': pred_price,
                    'ma_200': ma_200,
                    'atr': atr
                }
            )

            # Check risk/reward ratio
            if strategy_signal.calculate_risk_reward() < self.min_rr:
                return None

            return strategy_signal

        except Exception as e:
            # Silently fail on prediction errors
            return None
