"""
HLC Pattern Analyzer
====================

Implements the 8-pattern classification system inspired by neural network
pattern recognition. Analyzes swing highs/lows to identify market structure.

Pattern Types:
- HHH: Higher High, Higher Low (strongest bullish)
- HHL: Higher High, Lower Low (bullish with pullback)
- HLH: Higher Low, Higher High (bullish reversal)
- HLL: Higher Low, Lower Low (weak bullish)
- LHH: Lower High, Higher Low (consolidation)
- LHL: Lower High, Lower Low (bearish with bounce)
- LLH: Lower Low, Higher Low (bearish reversal)
- LLL: Lower Low, Lower High (strongest bearish)
- RANGE: No clear structure
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


class PatternType(Enum):
    """8-pattern classification system"""
    HHH = "HHH"  # Strongest bullish
    HHL = "HHL"  # Strong bullish
    HLH = "HLH"  # Bullish continuation
    HLL = "HLL"  # Weak bullish
    LHH = "LHH"  # Neutral/consolidation
    LHL = "LHL"  # Bearish continuation
    LLH = "LLH"  # Weak bearish
    LLL = "LLL"  # Strongest bearish
    RANGE = "RANGE"  # No clear pattern


class EntryZone(Enum):
    """Fibonacci retracement zones"""
    OPTIMAL = 1      # 38.2%-50% (best entry)
    ACCEPTABLE = 2   # 50%-61.8% (good entry)
    EXTENDED = 3     # 61.8%-78.6% (risky)
    TOO_LATE = 4     # Beyond 78.6% (avoid)
    NONE = 0


@dataclass
class SwingPoints:
    """Stores detected swing high/low points"""
    last_high: float = 0.0
    last_low: float = 0.0
    prev_high: float = 0.0
    prev_low: float = 0.0
    prev2_high: float = 0.0
    prev2_low: float = 0.0


@dataclass
class PatternResult:
    """Complete pattern analysis result"""
    pattern: PatternType
    strength: int  # 0-100
    is_bullish: bool
    is_bearish: bool

    # Fibonacci levels
    fib_382: float
    fib_500: float
    fib_618: float
    fib_786: float

    # Dynamic S/R
    support: float
    resistance: float

    # Entry analysis
    current_zone: EntryZone
    in_optimal_zone: bool
    retracement_pct: float

    # Trade levels
    optimal_entry: float
    optimal_sl: float
    optimal_tp: float
    rr_ratio: float

    # Validity
    valid_for_long: bool
    valid_for_short: bool
    confidence: float


class HLCPatternAnalyzer:
    """
    HLC Pattern Analyzer

    Detects swing points and classifies market structure into 8 patterns
    using Higher/Lower analysis of consecutive swing highs and lows.
    """

    def __init__(self, swing_lookback: int = 20, atr_period: int = 14):
        """
        Initialize pattern analyzer.

        Args:
            swing_lookback: Bars to look back for swing detection
            atr_period: Period for ATR calculation
        """
        self.swing_lookback = swing_lookback
        self.atr_period = atr_period

    def detect_swing_high(self, highs: np.ndarray, idx: int, window: int = 2) -> bool:
        """Detect if index is a swing high"""
        if idx < window or idx >= len(highs) - window:
            return False

        center = highs[idx]
        for j in range(1, window + 1):
            if highs[idx - j] >= center or highs[idx + j] >= center:
                return False
        return True

    def detect_swing_low(self, lows: np.ndarray, idx: int, window: int = 2) -> bool:
        """Detect if index is a swing low"""
        if idx < window or idx >= len(lows) - window:
            return False

        center = lows[idx]
        for j in range(1, window + 1):
            if lows[idx - j] <= center or lows[idx + j] <= center:
                return False
        return True

    def find_swing_points(self, data: pd.DataFrame) -> SwingPoints:
        """
        Find the last 3 swing highs and lows.

        Args:
            data: DataFrame with 'high' and 'low' columns

        Returns:
            SwingPoints object with detected swings
        """
        swings = SwingPoints()
        highs = data['high'].values
        lows = data['low'].values

        high_count = 0
        low_count = 0

        # Scan backwards from recent to older
        for i in range(3, min(self.swing_lookback, len(data) - 3)):
            # Detect swing highs
            if high_count < 3 and self.detect_swing_high(highs, i):
                if high_count == 0:
                    swings.last_high = highs[i]
                elif high_count == 1:
                    swings.prev_high = highs[i]
                elif high_count == 2:
                    swings.prev2_high = highs[i]
                high_count += 1

            # Detect swing lows
            if low_count < 3 and self.detect_swing_low(lows, i):
                if low_count == 0:
                    swings.last_low = lows[i]
                elif low_count == 1:
                    swings.prev_low = lows[i]
                elif low_count == 2:
                    swings.prev2_low = lows[i]
                low_count += 1

            if high_count >= 3 and low_count >= 3:
                break

        # Fallback if not enough swings found
        if swings.last_high == 0 or swings.last_low == 0:
            recent_data = data.iloc[-10:]
            swings.last_high = recent_data['high'].max()
            swings.prev_high = recent_data['high'].nlargest(2).iloc[-1]
            swings.prev2_high = swings.prev_high * 0.999

            swings.last_low = recent_data['low'].min()
            swings.prev_low = recent_data['low'].nsmallest(2).iloc[-1]
            swings.prev2_low = swings.prev_low * 1.001

        return swings

    def classify_pattern(self, swings: SwingPoints) -> Tuple[PatternType, int, bool, bool]:
        """
        Classify pattern based on swing analysis.

        Returns:
            (pattern_type, strength, is_bullish, is_bearish)
        """
        # Check if we have valid swings
        if swings.last_high == 0 or swings.last_low == 0:
            return PatternType.RANGE, 20, False, False

        # Analyze highs and lows
        HH = swings.last_high > swings.prev_high  # Higher high
        HL = swings.last_high < swings.prev_high  # Lower high
        HiL = swings.last_low > swings.prev_low   # Higher low
        LoL = swings.last_low < swings.prev_low   # Lower low

        # Pattern classification with strength scoring
        if HH and HiL:  # Both making higher points
            # Check if trend continues from previous
            if (swings.prev_high > swings.prev2_high and
                swings.prev_low > swings.prev2_low):
                return PatternType.HHH, 95, True, False  # Strongest bullish
            else:
                return PatternType.HHL, 75, True, False  # Strong bullish

        elif HH and LoL:  # Higher high, lower low
            return PatternType.HLH, 85, True, False  # Bullish continuation

        elif HL and HiL:  # Lower high, higher low
            return PatternType.LHH, 50, False, False  # Consolidation

        elif HL and LoL:  # Both making lower points
            # Check if downtrend continues
            if (swings.prev_high < swings.prev2_high and
                swings.prev_low < swings.prev2_low):
                return PatternType.LLL, 95, False, True  # Strongest bearish
            else:
                return PatternType.LHL, 85, False, True  # Strong bearish

        elif not HH and HiL:  # Equal high, higher low
            return PatternType.HLL, 40, False, False  # Weak bullish

        elif HL and not LoL:  # Lower high, equal low
            return PatternType.LLH, 60, False, True  # Weak bearish

        else:
            return PatternType.RANGE, 20, False, False

    def calculate_fibonacci_zones(self, swings: SwingPoints,
                                  is_bullish: bool, is_bearish: bool) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement zones.

        Returns:
            Dictionary with fib levels and support/resistance
        """
        support = min(swings.last_low, swings.prev_low)
        resistance = max(swings.last_high, swings.prev_high)
        range_val = resistance - support

        if range_val <= 0:
            range_val = 0.001  # Prevent division by zero

        if is_bullish:
            # Retracement from resistance downward
            fib_382 = resistance - (range_val * 0.382)
            fib_500 = resistance - (range_val * 0.500)
            fib_618 = resistance - (range_val * 0.618)
            fib_786 = resistance - (range_val * 0.786)
        elif is_bearish:
            # Retracement from support upward
            fib_382 = support + (range_val * 0.382)
            fib_500 = support + (range_val * 0.500)
            fib_618 = support + (range_val * 0.618)
            fib_786 = support + (range_val * 0.786)
        else:
            # Neutral - use midpoint
            mid = (support + resistance) / 2
            fib_382 = fib_500 = fib_618 = fib_786 = mid

        return {
            'fib_382': fib_382,
            'fib_500': fib_500,
            'fib_618': fib_618,
            'fib_786': fib_786,
            'support': support,
            'resistance': resistance
        }

    def determine_entry_zone(self, current_price: float, fib_levels: Dict[str, float],
                            is_bullish: bool, is_bearish: bool) -> Tuple[EntryZone, bool, float]:
        """
        Determine which Fibonacci zone current price is in.

        Returns:
            (zone, in_optimal_zone, retracement_pct)
        """
        if is_bullish:
            if fib_levels['fib_382'] <= current_price <= fib_levels['fib_500']:
                return EntryZone.OPTIMAL, True, 38.2 + (50 - 38.2) * 0.5
            elif fib_levels['fib_500'] <= current_price <= fib_levels['fib_618']:
                return EntryZone.ACCEPTABLE, True, 50 + (61.8 - 50) * 0.5
            elif fib_levels['fib_618'] <= current_price <= fib_levels['fib_786']:
                return EntryZone.EXTENDED, False, 61.8 + (78.6 - 61.8) * 0.5
            elif current_price < fib_levels['fib_786']:
                return EntryZone.TOO_LATE, False, 78.6

        elif is_bearish:
            if fib_levels['fib_500'] <= current_price <= fib_levels['fib_382']:
                return EntryZone.OPTIMAL, True, 38.2 + (50 - 38.2) * 0.5
            elif fib_levels['fib_618'] <= current_price <= fib_levels['fib_500']:
                return EntryZone.ACCEPTABLE, True, 50 + (61.8 - 50) * 0.5
            elif fib_levels['fib_786'] <= current_price <= fib_levels['fib_618']:
                return EntryZone.EXTENDED, False, 61.8 + (78.6 - 61.8) * 0.5
            elif current_price > fib_levels['fib_786']:
                return EntryZone.TOO_LATE, False, 78.6

        return EntryZone.NONE, False, 0.0

    def calculate_trade_levels(self, swings: SwingPoints, current_price: float,
                               atr: float, pattern_strength: int,
                               is_bullish: bool, is_bearish: bool) -> Dict[str, float]:
        """
        Calculate optimal entry, stop loss, and take profit levels.
        Adapts based on pattern strength (stronger patterns = tighter stops).
        """
        # Adaptive ATR multipliers based on pattern strength
        if pattern_strength >= 90:
            sl_mult = 1.2  # Tighter stop for high confidence
            tp_mult = 4.5  # More aggressive target
        elif pattern_strength >= 70:
            sl_mult = 1.5
            tp_mult = 3.5
        else:
            sl_mult = 2.0  # Wider stop for weaker patterns
            tp_mult = 2.5

        if is_bullish:
            entry = current_price
            sl = swings.last_low - (atr * sl_mult)
            tp = swings.last_high + (atr * tp_mult)

            risk = entry - sl
            reward = tp - entry
            rr = reward / risk if risk > 0 else 0

        elif is_bearish:
            entry = current_price
            sl = swings.last_high + (atr * sl_mult)
            tp = swings.last_low - (atr * tp_mult)

            risk = sl - entry
            reward = entry - tp
            rr = reward / risk if risk > 0 else 0
        else:
            entry = sl = tp = current_price
            rr = 0

        return {
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'rr_ratio': rr
        }

    def analyze(self, data: pd.DataFrame) -> PatternResult:
        """
        Complete pattern analysis.

        Args:
            data: DataFrame with OHLC columns

        Returns:
            PatternResult with full analysis
        """
        # Find swing points
        swings = self.find_swing_points(data)

        # Classify pattern
        pattern, strength, is_bullish, is_bearish = self.classify_pattern(swings)

        # Calculate ATR
        atr = self._calculate_atr(data)

        # Calculate Fibonacci zones
        fib_levels = self.calculate_fibonacci_zones(swings, is_bullish, is_bearish)

        # Current price
        current_price = data['close'].iloc[-1]

        # Determine entry zone
        zone, in_optimal, retr_pct = self.determine_entry_zone(
            current_price, fib_levels, is_bullish, is_bearish
        )

        # Calculate trade levels
        levels = self.calculate_trade_levels(
            swings, current_price, atr, strength, is_bullish, is_bearish
        )

        # Determine validity (multi-factor validation)
        volatility = (atr / current_price) * 100 if current_price > 0 else 0
        valid_long = (is_bullish and in_optimal and
                     levels['rr_ratio'] >= 2.0 and
                     strength >= 70 and
                     volatility <= 5.0)

        valid_short = (is_bearish and in_optimal and
                      levels['rr_ratio'] >= 2.0 and
                      strength >= 70 and
                      volatility <= 5.0)

        return PatternResult(
            pattern=pattern,
            strength=strength,
            is_bullish=is_bullish,
            is_bearish=is_bearish,
            fib_382=fib_levels['fib_382'],
            fib_500=fib_levels['fib_500'],
            fib_618=fib_levels['fib_618'],
            fib_786=fib_levels['fib_786'],
            support=fib_levels['support'],
            resistance=fib_levels['resistance'],
            current_zone=zone,
            in_optimal_zone=in_optimal,
            retracement_pct=retr_pct,
            optimal_entry=levels['entry'],
            optimal_sl=levels['sl'],
            optimal_tp=levels['tp'],
            rr_ratio=levels['rr_ratio'],
            valid_for_long=valid_long,
            valid_for_short=valid_short,
            confidence=strength
        )

    def _calculate_atr(self, data: pd.DataFrame, period: Optional[int] = None) -> float:
        """Calculate Average True Range"""
        if period is None:
            period = self.atr_period

        df = data.tail(period + 1).copy()

        df['h-l'] = df['high'] - df['low']
        df['h-pc'] = abs(df['high'] - df['close'].shift(1))
        df['l-pc'] = abs(df['low'] - df['close'].shift(1))

        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)

        return df['tr'].mean()
