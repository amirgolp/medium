"""
Pattern Strategy
================

Pure HLC pattern-based strategy using Nobel Prize pattern recognition.
"""

from typing import Optional
import pandas as pd
from .base_strategy import BaseStrategy, StrategySignal
from ..pattern_analyzer import HLCPatternAnalyzer, PatternType, EntryZone


class PatternStrategy(BaseStrategy):
    """
    Pattern-based strategy using HLC pattern recognition.

    Uses adaptive stop loss based on pattern strength:
    - High strength patterns (>=90%): Tighter stops (1.2 ATR)
    - Medium strength (>=70%): Balanced stops (1.5 ATR)
    - Lower strength: Wider stops (2.0 ATR)

    Only trades when:
    - Pattern strength >= 70%
    - In optimal or acceptable Fibonacci zone
    - Risk/reward >= 2.0
    - Volatility <= 5%
    """

    def __init__(self, swing_lookback: int = 20, atr_period: int = 14,
                 min_strength: int = 70, min_rr: float = 2.0,
                 max_volatility_pct: float = 5.0):
        """
        Initialize pattern strategy.

        Args:
            swing_lookback: Bars to look back for swing detection
            atr_period: Period for ATR calculation
            min_strength: Minimum pattern strength (0-100)
            min_rr: Minimum risk/reward ratio
            max_volatility_pct: Maximum volatility percentage
        """
        super().__init__()
        self.analyzer = HLCPatternAnalyzer(
            swing_lookback=swing_lookback,
            atr_period=atr_period
        )
        self.min_strength = min_strength
        self.min_rr = min_rr
        self.max_volatility_pct = max_volatility_pct

    def get_name(self) -> str:
        return "Pattern Strategy (HLC)"

    def get_description(self) -> str:
        return (
            f"Nobel Prize HLC pattern recognition with "
            f"min strength={self.min_strength}%, "
            f"Fibonacci zone entry, "
            f"adaptive stops based on pattern confidence"
        )

    def generate_signal(self, data: pd.DataFrame, index: int) -> Optional[StrategySignal]:
        """Generate pattern-based signal."""

        # Need enough history for swing detection
        if index < self.analyzer.swing_lookback + 10:
            return None

        # Get recent data for analysis
        lookback_data = data.iloc[max(0, index-100):index+1].copy()

        if len(lookback_data) < 20:
            return None

        try:
            # Analyze pattern
            result = self.analyzer.analyze(lookback_data)

            # Check pattern strength
            if result.strength < self.min_strength:
                return None

            # Check if in optimal zone
            if not result.in_optimal_zone:
                return None

            # Check risk/reward ratio
            if result.rr_ratio < self.min_rr:
                return None

            # Check volatility
            current_price = data.iloc[index]['close']
            atr = self.calculate_atr(data, index, period=self.analyzer.atr_period)
            volatility_pct = (atr / current_price) * 100 if current_price > 0 else 0

            if volatility_pct > self.max_volatility_pct:
                return None

            # Determine signal direction
            if result.valid_for_long:
                signal = "BUY"
            elif result.valid_for_short:
                signal = "SELL"
            else:
                return None

            # Create signal with pattern levels
            strategy_signal = StrategySignal(
                signal=signal,
                entry_price=current_price,
                stop_loss=result.optimal_sl,
                take_profit=result.optimal_tp,
                lot_size=0.1,
                trailing_stop=None,  # Pattern-based doesn't use trailing stop
                confidence=result.confidence,
                pattern_type=result.pattern.value,
                entry_zone=result.current_zone.name,
                metadata={
                    'pattern': result.pattern.value,
                    'pattern_strength': result.strength,
                    'entry_zone': result.current_zone.name,
                    'retracement_pct': result.retracement_pct,
                    'fib_382': result.fib_382,
                    'fib_500': result.fib_500,
                    'fib_618': result.fib_618,
                    'support': result.support,
                    'resistance': result.resistance,
                    'is_bullish': result.is_bullish,
                    'is_bearish': result.is_bearish
                }
            )

            return strategy_signal

        except Exception as e:
            # Silently fail on analysis errors
            return None
