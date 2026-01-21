"""
Hybrid Strategy
===============

Combined ML predictions + HLC pattern recognition.
"""

from typing import Optional
import pandas as pd
from .base_strategy import BaseStrategy, StrategySignal
from ..pattern_analyzer import HLCPatternAnalyzer, PatternType


class HybridStrategy(BaseStrategy):
    """
    Hybrid strategy combining ML predictions with pattern analysis.

    Signal generation:
    1. Get ML prediction (BUY/SELL)
    2. Analyze HLC pattern
    3. Only trade if both agree on direction
    4. Use pattern-based stops if pattern is strong (>=80%)
    5. Otherwise use ML-based stops

    This reduces false signals while maintaining pattern recognition benefits.
    """

    def __init__(self, predictor, history_size: int = 120,
                 swing_lookback: int = 20, atr_period: int = 14,
                 min_pattern_strength: int = 60, min_rr: float = 2.0):
        """
        Initialize hybrid strategy.

        Args:
            predictor: ForexPredictor instance
            history_size: History bars for ML prediction
            swing_lookback: Bars for swing detection
            atr_period: Period for ATR calculation
            min_pattern_strength: Minimum pattern strength
            min_rr: Minimum risk/reward ratio
        """
        super().__init__()
        self.predictor = predictor
        self.history_size = history_size
        self.analyzer = HLCPatternAnalyzer(
            swing_lookback=swing_lookback,
            atr_period=atr_period
        )
        self.min_pattern_strength = min_pattern_strength
        self.min_rr = min_rr

    def get_name(self) -> str:
        return "Hybrid Strategy (ML + Pattern)"

    def get_description(self) -> str:
        return (
            f"Combined CNN-LSTM predictions with HLC pattern recognition. "
            f"Requires both signals to agree. "
            f"Uses pattern stops when strength >= 80%, ML stops otherwise."
        )

    def generate_signal(self, data: pd.DataFrame, index: int) -> Optional[StrategySignal]:
        """Generate hybrid signal."""

        # Need enough history for both ML and pattern analysis
        if index < 200:
            return None

        try:
            # 1. Get ML prediction
            recent_prices = data.iloc[index-self.history_size:index]['close'].values
            prediction, pred_price, last_close = self.predictor.predict(recent_prices)
            ml_signal = self.predictor.get_signal_name(prediction)

            if ml_signal == "NEUTRAL":
                return None

            # 2. Analyze pattern
            lookback_data = data.iloc[max(0, index-100):index+1].copy()
            if len(lookback_data) < 20:
                return None

            pattern_result = self.analyzer.analyze(lookback_data)

            # 3. Check pattern strength
            if pattern_result.strength < self.min_pattern_strength:
                return None

            # 4. Check if ML and pattern agree
            pattern_direction = None
            if pattern_result.is_bullish:
                pattern_direction = "BUY"
            elif pattern_result.is_bearish:
                pattern_direction = "SELL"

            if ml_signal != pattern_direction:
                return None  # Signals don't agree

            # 5. Trend filter - Check 200-period MA
            ma_200 = self.calculate_ma(data, index, period=200, column='close')
            current_price = data.iloc[index]['close']

            if ml_signal == "BUY" and current_price < ma_200:
                return None
            if ml_signal == "SELL" and current_price > ma_200:
                return None

            # 6. Time filter
            bar_time = data.iloc[index]['time']
            trade_hour = bar_time.hour
            if trade_hour < 11 or trade_hour >= 22:
                return None

            # 7. Calculate stops - use pattern stops if strong, ML stops otherwise
            use_pattern_stops = pattern_result.strength >= 80

            if use_pattern_stops:
                # Use pattern-based stops
                sl = pattern_result.optimal_sl
                tp = pattern_result.optimal_tp
                trailing = None
            else:
                # Use ML-based stops
                atr = self.calculate_atr(data, index, period=14)
                if atr <= 0:
                    return None

                if ml_signal == "BUY":
                    sl = current_price - (atr * 2.0)
                    tp = current_price + (atr * 4.0)
                else:
                    sl = current_price + (atr * 2.0)
                    tp = current_price - (atr * 4.0)

                trailing = atr * 1.0

            # Create signal
            strategy_signal = StrategySignal(
                signal=ml_signal,
                entry_price=current_price,
                stop_loss=sl,
                take_profit=tp,
                lot_size=0.1,
                trailing_stop=trailing,
                confidence=pattern_result.confidence,
                pattern_type=pattern_result.pattern.value,
                entry_zone=pattern_result.current_zone.name if pattern_result.current_zone else None,
                metadata={
                    'ml_signal': ml_signal,
                    'pattern': pattern_result.pattern.value,
                    'pattern_strength': pattern_result.strength,
                    'use_pattern_stops': use_pattern_stops,
                    'predicted_price': pred_price,
                    'ma_200': ma_200,
                    'entry_zone': pattern_result.current_zone.name if pattern_result.current_zone else None,
                    'is_bullish': pattern_result.is_bullish,
                    'is_bearish': pattern_result.is_bearish
                }
            )

            # Check risk/reward ratio
            if strategy_signal.calculate_risk_reward() < self.min_rr:
                return None

            return strategy_signal

        except Exception as e:
            # Silently fail on errors
            return None
