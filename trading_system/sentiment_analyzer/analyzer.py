"""
Sentiment Analyzer
==================

Aggregates sentiment from multiple news sources and generates trading signals.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Represents a single sentiment data point."""
    time: datetime
    currency: str
    event_name: str
    score: float  # 0-100 scale (50=neutral, >50=positive, <50=negative)
    source: str  # "MT5", "NewsAPI", "Finnhub"
    impact: str  # "High", "Medium", "Low", "API"


class SentimentAnalyzer:
    """
    Analyze sentiment from multiple sources and generate trading signals.

    Aggregates news from:
    - MT5 Economic Calendar (structured economic data)
    - NewsAPI (unstructured news headlines)
    - Finnhub (forex-specific news)
    """

    # Sentiment keywords
    POSITIVE_KEYWORDS = [
        "gains", "rise", "surge", "boost", "strong", "growth", "improved",
        "higher", "rally", "advance", "bullish", "optimistic", "recovery",
        "strengthen", "jump", "soar", "positive", "expansion", "beat",
        "exceed", "outperform", "robust", "solid", "accelerate"
    ]

    NEGATIVE_KEYWORDS = [
        "falls", "drop", "decline", "weak", "recession", "crisis", "lower",
        "plunge", "bearish", "pessimistic", "concern", "worry", "deteriorate",
        "slump", "tumble", "negative", "contraction", "risk", "miss",
        "disappoint", "weaken", "slowdown", "struggle", "fear"
    ]

    def __init__(
        self,
        lookback_hours: int = 24,
        min_sentiment_score: float = 10.0,
        high_impact_only: bool = False
    ):
        """
        Initialize the sentiment analyzer.

        Args:
            lookback_hours: Hours to look back for sentiment aggregation
            min_sentiment_score: Minimum score difference from neutral (50) for signal
            high_impact_only: Only consider high-impact news
        """
        self.lookback_hours = lookback_hours
        self.min_sentiment_score = min_sentiment_score
        self.high_impact_only = high_impact_only
        self.sentiment_history: List[SentimentScore] = []

    def add_sentiment(self, sentiment: SentimentScore) -> None:
        """
        Add a sentiment score to history.

        Args:
            sentiment: SentimentScore object to add
        """
        self.sentiment_history.append(sentiment)

    def analyze_text_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using keyword matching.

        Args:
            text: Text to analyze (headline, description, etc.)

        Returns:
            Sentiment score (0-100, 50=neutral)
        """
        text_lower = text.lower()

        positive_count = sum(1 for word in self.POSITIVE_KEYWORDS if word in text_lower)
        negative_count = sum(1 for word in self.NEGATIVE_KEYWORDS if word in text_lower)

        total = positive_count + negative_count
        if total == 0:
            return 50.0  # Neutral

        # Calculate score with 10-point increments per keyword
        sentiment = 50.0 + ((positive_count - negative_count) * 10.0)
        sentiment = max(0.0, min(100.0, sentiment))

        return sentiment

    def detect_currency(self, text: str) -> Optional[str]:
        """
        Detect which currency is mentioned in text.

        Args:
            text: Text to analyze

        Returns:
            Currency code ("USD", "EUR", etc.) or None
        """
        text_upper = text.upper()

        currency_keywords = {
            "EUR": ["EUR", "EURO", "ECB", "EUROZONE", "EUROPEAN CENTRAL BANK"],
            "USD": ["USD", "DOLLAR", "FED", "FEDERAL RESERVE", "FOMC"],
            "GBP": ["GBP", "POUND", "STERLING", "BOE", "BANK OF ENGLAND"],
            "JPY": ["JPY", "YEN", "BOJ", "BANK OF JAPAN"],
            "CHF": ["CHF", "FRANC", "SNB", "SWISS NATIONAL BANK"],
            "CAD": ["CAD", "LOONIE", "BOC", "BANK OF CANADA"],
        }

        for currency, keywords in currency_keywords.items():
            if any(keyword in text_upper for keyword in keywords):
                return currency

        return None

    def calculate_economic_sentiment(
        self,
        actual: float,
        forecast: float,
        is_positive_indicator: bool = True
    ) -> float:
        """
        Calculate sentiment from economic calendar event.

        Args:
            actual: Actual value released
            forecast: Forecasted value
            is_positive_indicator: True if higher is better (GDP, employment)
                                   False if lower is better (unemployment, inflation)

        Returns:
            Sentiment score (0-100)
        """
        if forecast == 0:
            return 50.0  # Neutral if no forecast

        # Calculate percentage difference
        pct_diff = ((actual - forecast) / abs(forecast)) * 100.0

        # Adjust for indicator type
        if not is_positive_indicator:
            pct_diff = -pct_diff

        # Map to 0-100 scale
        # -10% diff = 0, 0% diff = 50, +10% diff = 100
        sentiment = 50.0 + (pct_diff * 5.0)
        sentiment = max(0.0, min(100.0, sentiment))

        return sentiment

    def get_currency_sentiment(
        self,
        currency: str,
        hours_back: Optional[int] = None
    ) -> Tuple[float, int]:
        """
        Get aggregated sentiment for a currency.

        Args:
            currency: Currency code (e.g., "USD", "EUR")
            hours_back: Hours to look back (uses self.lookback_hours if None)

        Returns:
            Tuple of (average_sentiment, count)
        """
        hours_back = hours_back or self.lookback_hours
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        relevant_scores = [
            s.score for s in self.sentiment_history
            if s.currency == currency
            and s.time >= cutoff_time
            and (not self.high_impact_only or s.impact == "High")
        ]

        if not relevant_scores:
            return 50.0, 0  # Neutral, no data

        avg_sentiment = np.mean(relevant_scores)
        return float(avg_sentiment), len(relevant_scores)

    def generate_signal(
        self,
        base_currency: str,
        quote_currency: str
    ) -> Tuple[Optional[str], float, Dict[str, any]]:
        """
        Generate trading signal by comparing two currencies' sentiment.

        Args:
            base_currency: Base currency (e.g., "EUR" in EURUSD)
            quote_currency: Quote currency (e.g., "USD" in EURUSD)

        Returns:
            Tuple of (signal, confidence, details)
            - signal: "BUY", "SELL", or None
            - confidence: Score difference from neutral
            - details: Dictionary with analysis details
        """
        base_sentiment, base_count = self.get_currency_sentiment(base_currency)
        quote_sentiment, quote_count = self.get_currency_sentiment(quote_currency)

        # Calculate relative sentiment
        # If base currency sentiment > quote currency sentiment -> BUY
        # If base currency sentiment < quote currency sentiment -> SELL
        sentiment_diff = base_sentiment - quote_sentiment

        details = {
            "base_currency": base_currency,
            "quote_currency": quote_currency,
            "base_sentiment": base_sentiment,
            "quote_sentiment": quote_sentiment,
            "base_count": base_count,
            "quote_count": quote_count,
            "sentiment_diff": sentiment_diff,
        }

        # Check if signal is strong enough
        if abs(sentiment_diff) < self.min_sentiment_score:
            logger.info(f"No signal: sentiment diff {sentiment_diff:.1f} < {self.min_sentiment_score}")
            return None, 0.0, details

        # Generate signal
        if sentiment_diff > 0:
            signal = "BUY"
            confidence = sentiment_diff
        else:
            signal = "SELL"
            confidence = abs(sentiment_diff)

        logger.info(f"Signal: {signal} {base_currency}{quote_currency}")
        logger.info(f"  {base_currency} sentiment: {base_sentiment:.1f} ({base_count} events)")
        logger.info(f"  {quote_currency} sentiment: {quote_sentiment:.1f} ({quote_count} events)")
        logger.info(f"  Confidence: {confidence:.1f}")

        return signal, confidence, details

    def get_recent_sentiment_summary(self, hours: int = 24) -> Dict[str, any]:
        """
        Get summary of recent sentiment data.

        Args:
            hours: Hours to look back

        Returns:
            Dictionary with summary statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent = [s for s in self.sentiment_history if s.time >= cutoff_time]

        if not recent:
            return {"total": 0, "by_currency": {}, "by_source": {}}

        # Group by currency
        by_currency = {}
        for sentiment in recent:
            if sentiment.currency not in by_currency:
                by_currency[sentiment.currency] = []
            by_currency[sentiment.currency].append(sentiment.score)

        # Group by source
        by_source = {}
        for sentiment in recent:
            if sentiment.source not in by_source:
                by_source[sentiment.source] = 0
            by_source[sentiment.source] += 1

        # Calculate averages
        currency_avg = {
            currency: np.mean(scores)
            for currency, scores in by_currency.items()
        }

        return {
            "total": len(recent),
            "by_currency": currency_avg,
            "by_source": by_source,
            "time_range": f"Last {hours} hours",
        }

    def clear_old_data(self, hours_to_keep: int = 48) -> int:
        """
        Remove sentiment data older than specified hours.

        Args:
            hours_to_keep: How many hours of data to retain

        Returns:
            Number of items removed
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_to_keep)
        original_count = len(self.sentiment_history)

        self.sentiment_history = [
            s for s in self.sentiment_history
            if s.time >= cutoff_time
        ]

        removed = original_count - len(self.sentiment_history)
        if removed > 0:
            logger.info(f"Cleared {removed} old sentiment entries")

        return removed
