"""
Sentiment Strategy
==================

ML predictions combined with sentiment analysis.
"""

from typing import Optional
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, StrategySignal


class SentimentStrategy(BaseStrategy):
    """
    Strategy combining ML predictions with sentiment analysis.

    Only trades when both ML and sentiment agree on direction.
    Uses sentiment confidence to adjust position sizing.
    """

    def __init__(self, predictor, sentiment_analyzer, symbol: str,
                 history_size: int = 120, min_sentiment_score: float = 10.0,
                 sl_atr_mult: float = 2.0, tp_atr_mult: float = 4.0):
        """
        Initialize sentiment strategy.

        Args:
            predictor: ForexPredictor instance
            sentiment_analyzer: SentimentAnalyzer instance
            symbol: Currency pair (e.g., "EURUSD")
            history_size: History bars for prediction
            min_sentiment_score: Minimum sentiment score
            sl_atr_mult: Stop loss ATR multiplier
            tp_atr_mult: Take profit ATR multiplier
        """
        super().__init__()
        self.predictor = predictor
        self.sentiment_analyzer = sentiment_analyzer
        self.symbol = symbol
        self.history_size = history_size
        self.min_sentiment_score = min_sentiment_score
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult

        # Extract base currencies
        if len(symbol) == 6:
            self.base_currency = symbol[:3]
            self.quote_currency = symbol[3:]
        else:
            self.base_currency = "EUR"
            self.quote_currency = "USD"

    def get_name(self) -> str:
        return "Sentiment Strategy (ML + Sentiment)"

    def get_description(self) -> str:
        return (
            f"CNN-LSTM predictions combined with sentiment analysis for {self.symbol}. "
            f"Requires both signals to agree on direction."
        )

    def generate_signal(self, data: pd.DataFrame, index: int) -> Optional[StrategySignal]:
        """Generate sentiment-based signal."""

        # Need enough history
        if index < 120:
            return None

        # Get recent prices for prediction
        recent_prices = data.iloc[index-self.history_size:index]['close'].values

        try:
            # 1. Get ML prediction
            prediction, pred_price, last_close = self.predictor.predict(recent_prices)
            ml_signal = self.predictor.get_signal_name(prediction)

            if ml_signal == "NEUTRAL":
                return None

            # 2. Get sentiment signal
            sent_signal, confidence, details = self.sentiment_analyzer.generate_signal(
                self.base_currency,
                self.quote_currency
            )

            # 3. Check if signals align
            if not sent_signal or ml_signal != sent_signal:
                return None

            # 4. Check sentiment confidence
            if confidence < self.min_sentiment_score:
                return None

            # 5. Calculate ATR
            atr = self.calculate_atr(data, index, period=14)
            if atr <= 0:
                return None

            # Get current price
            current_price = data.iloc[index]['close']

            # 6. Calculate stops
            if ml_signal == "BUY":
                sl = current_price - (atr * self.sl_atr_mult)
                tp = current_price + (atr * self.tp_atr_mult)
            else:
                sl = current_price + (atr * self.sl_atr_mult)
                tp = current_price - (atr * self.tp_atr_mult)

            # Create signal
            strategy_signal = StrategySignal(
                signal=ml_signal,
                entry_price=current_price,
                stop_loss=sl,
                take_profit=tp,
                lot_size=0.1,
                trailing_stop=None,
                confidence=confidence,
                metadata={
                    'ml_signal': ml_signal,
                    'sentiment_signal': sent_signal,
                    'sentiment_score': confidence,
                    'sentiment_details': details,
                    'predicted_price': pred_price,
                    'atr': atr
                }
            )

            return strategy_signal

        except Exception as e:
            # Silently fail on errors
            return None
