"""
Sentiment Analyzer Module
=========================

News and economic event sentiment analysis for forex trading.
"""

from .analyzer import SentimentAnalyzer
from .news_sources import NewsAPIClient, FinnhubClient, MT5CalendarClient

__all__ = ["SentimentAnalyzer", "NewsAPIClient", "FinnhubClient", "MT5CalendarClient"]
