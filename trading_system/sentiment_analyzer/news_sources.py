"""
News Source Clients
===================

API clients for fetching news from various sources.
"""

import logging
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from .analyzer import SentimentScore

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsAPIClient:
    """
    Client for NewsAPI.org - general news headlines and articles.

    Get your free API key at: https://newsapi.org
    """

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str):
        """
        Initialize NewsAPI client.

        Args:
            api_key: Your NewsAPI key
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def fetch_news(
        self,
        query: str = "(EUR OR euro OR ECB OR eurozone) OR (USD OR dollar OR Fed OR Federal Reserve)",
        language: str = "en",
        page_size: int = 20,
        hours_back: int = 24
    ) -> List[Dict[str, any]]:
        """
        Fetch news from NewsAPI.

        Args:
            query: Search query
            language: Language code
            page_size: Number of results
            hours_back: Hours to look back

        Returns:
            List of article dictionaries
        """
        from_date = (datetime.now() - timedelta(hours=hours_back)).isoformat()

        params = {
            "q": query,
            "language": language,
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "from": from_date,
            "apiKey": self.api_key,
        }

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get("status") != "ok":
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []

            articles = data.get("articles", [])
            logger.info(f"✓ NewsAPI: Fetched {len(articles)} articles")

            return articles

        except requests.exceptions.RequestException as e:
            logger.error(f"NewsAPI request failed: {e}")
            return []

    def parse_to_sentiment(
        self,
        articles: List[Dict[str, any]],
        sentiment_analyzer
    ) -> List[SentimentScore]:
        """
        Parse NewsAPI articles into sentiment scores.

        Args:
            articles: List of article dictionaries from NewsAPI
            sentiment_analyzer: SentimentAnalyzer instance for text analysis

        Returns:
            List of SentimentScore objects
        """
        sentiments = []

        for article in articles:
            title = article.get("title", "")
            description = article.get("description", "")
            published_at = article.get("publishedAt", "")

            full_text = f"{title} {description}"

            # Analyze sentiment
            score = sentiment_analyzer.analyze_text_sentiment(full_text)
            currency = sentiment_analyzer.detect_currency(full_text)

            if currency and score != 50.0:
                try:
                    pub_time = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                except:
                    pub_time = datetime.now()

                sentiment = SentimentScore(
                    time=pub_time,
                    currency=currency,
                    event_name=f"NewsAPI: {title[:50]}",
                    score=score,
                    source="NewsAPI",
                    impact="API"
                )
                sentiments.append(sentiment)

        logger.info(f"✓ NewsAPI: Parsed {len(sentiments)} sentiment scores")
        return sentiments


class FinnhubClient:
    """
    Client for Finnhub.io - forex and financial news.

    Get your free API key at: https://finnhub.io
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str):
        """
        Initialize Finnhub client.

        Args:
            api_key: Your Finnhub API key
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def fetch_forex_news(
        self,
        min_id: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Fetch forex news from Finnhub.

        Args:
            min_id: Minimum news ID to fetch (for pagination)

        Returns:
            List of news dictionaries
        """
        url = f"{self.BASE_URL}/news"
        params = {
            "category": "forex",
            "token": self.api_key,
        }

        if min_id:
            params["minId"] = min_id

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            news = response.json()
            logger.info(f"✓ Finnhub: Fetched {len(news)} news items")

            return news

        except requests.exceptions.RequestException as e:
            logger.error(f"Finnhub request failed: {e}")
            return []

    def parse_to_sentiment(
        self,
        news_items: List[Dict[str, any]],
        sentiment_analyzer
    ) -> List[SentimentScore]:
        """
        Parse Finnhub news into sentiment scores.

        Args:
            news_items: List of news dictionaries from Finnhub
            sentiment_analyzer: SentimentAnalyzer instance for text analysis

        Returns:
            List of SentimentScore objects
        """
        sentiments = []

        for item in news_items:
            headline = item.get("headline", "")
            summary = item.get("summary", "")
            timestamp = item.get("datetime", 0)

            full_text = f"{headline} {summary}"

            # Analyze sentiment
            score = sentiment_analyzer.analyze_text_sentiment(full_text)
            currency = sentiment_analyzer.detect_currency(full_text)

            if currency and score != 50.0:
                pub_time = datetime.fromtimestamp(timestamp)

                sentiment = SentimentScore(
                    time=pub_time,
                    currency=currency,
                    event_name=f"Finnhub: {headline[:50]}",
                    score=score,
                    source="Finnhub",
                    impact="API"
                )
                sentiments.append(sentiment)

        logger.info(f"✓ Finnhub: Parsed {len(sentiments)} sentiment scores")
        return sentiments


class MT5CalendarClient:
    """
    Client for MetaTrader 5 Economic Calendar.

    Fetches structured economic events directly from MT5.
    """

    # Indicator types where higher is better
    POSITIVE_INDICATORS = [
        "GDP", "Employment", "Retail Sales", "PMI", "Consumer Confidence",
        "Industrial Production", "Exports", "Business Confidence"
    ]

    # Indicator types where lower is better
    NEGATIVE_INDICATORS = [
        "Unemployment", "Inflation", "CPI", "PPI", "Trade Deficit",
        "Jobless Claims", "Deficit"
    ]

    def __init__(self):
        """Initialize MT5 Calendar client."""
        if not MT5_AVAILABLE:
            logger.warning("MetaTrader5 not available")

    def fetch_calendar_events(
        self,
        hours_back: int = 24,
        hours_forward: int = 0
    ) -> List[Dict[str, any]]:
        """
        Fetch economic calendar events from MT5.

        Args:
            hours_back: Hours to look back
            hours_forward: Hours to look forward

        Returns:
            List of event dictionaries
        """
        if not MT5_AVAILABLE:
            return []

        if not mt5.initialize():
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return []

        start_time = datetime.now() - timedelta(hours=hours_back)
        end_time = datetime.now() + timedelta(hours=hours_forward)

        calendar = mt5.calendar_value_last_by_event()
        mt5.shutdown()

        if calendar is None:
            logger.warning("No calendar data available")
            return []

        # Filter by time range
        events = [
            event for event in calendar
            if start_time <= datetime.fromtimestamp(event.time) <= end_time
        ]

        logger.info(f"✓ MT5 Calendar: Fetched {len(events)} events")
        return events

    def is_positive_indicator(self, event_name: str) -> bool:
        """
        Determine if higher values are positive for the indicator.

        Args:
            event_name: Name of the economic indicator

        Returns:
            True if higher is better, False if lower is better
        """
        event_upper = event_name.upper()

        for indicator in self.POSITIVE_INDICATORS:
            if indicator.upper() in event_upper:
                return True

        for indicator in self.NEGATIVE_INDICATORS:
            if indicator.upper() in event_upper:
                return False

        # Default: higher is better
        return True

    def parse_to_sentiment(
        self,
        events: List[any],
        sentiment_analyzer
    ) -> List[SentimentScore]:
        """
        Parse MT5 calendar events into sentiment scores.

        Args:
            events: List of MT5 calendar event objects
            sentiment_analyzer: SentimentAnalyzer instance

        Returns:
            List of SentimentScore objects
        """
        sentiments = []

        for event in events:
            # Skip events without actual values
            if not hasattr(event, 'actual_value') or not hasattr(event, 'forecast_value'):
                continue

            if event.actual_value == 0 and event.forecast_value == 0:
                continue

            # Determine currency
            currency_code = getattr(event, 'currency', '')
            if not currency_code:
                continue

            # Calculate sentiment
            is_positive = self.is_positive_indicator(event.name)
            score = sentiment_analyzer.calculate_economic_sentiment(
                actual=event.actual_value,
                forecast=event.forecast_value,
                is_positive_indicator=is_positive
            )

            # Determine impact level
            impact = getattr(event, 'impact_type', 0)
            impact_map = {0: "Low", 1: "Medium", 2: "High"}
            impact_str = impact_map.get(impact, "Low")

            event_time = datetime.fromtimestamp(event.time)

            sentiment = SentimentScore(
                time=event_time,
                currency=currency_code,
                event_name=event.name,
                score=score,
                source="MT5",
                impact=impact_str
            )
            sentiments.append(sentiment)

        logger.info(f"✓ MT5 Calendar: Parsed {len(sentiments)} sentiment scores")
        return sentiments
