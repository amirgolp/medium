"""
Example: Analyze News Sentiment
================================

Fetch and analyze news sentiment from multiple sources.
"""

from trading_system import SentimentAnalyzer
from trading_system.sentiment_analyzer import NewsAPIClient, FinnhubClient, MT5CalendarClient
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Analyze sentiment from multiple news sources."""

    print("="*60)
    print("NEWS SENTIMENT ANALYSIS")
    print("="*60)

    # Get API keys from environment or use placeholders
    newsapi_key = os.getenv("NEWSAPI_KEY", "your_newsapi_key_here")
    finnhub_key = os.getenv("FINNHUB_KEY", "your_finnhub_key_here")

    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer(
        lookback_hours=24,
        min_sentiment_score=10.0,
        high_impact_only=False
    )

    print("Configuration:")
    print(f"  Lookback: 24 hours")
    print(f"  Min Score: 10.0")
    print("="*60)

    # 1. Fetch from NewsAPI
    if newsapi_key != "your_newsapi_key_here":
        print("\n📡 Fetching from NewsAPI...")
        news_client = NewsAPIClient(newsapi_key)

        articles = news_client.fetch_news(
            query="(EUR OR euro OR ECB) OR (USD OR dollar OR Fed)",
            hours_back=24
        )

        if articles:
            sentiments = news_client.parse_to_sentiment(articles, analyzer)
            for sentiment in sentiments:
                analyzer.add_sentiment(sentiment)
            print(f"  ✓ Added {len(sentiments)} sentiment scores")
    else:
        print("\n⚠️  NewsAPI key not set, skipping...")

    # 2. Fetch from Finnhub
    if finnhub_key != "your_finnhub_key_here":
        print("\n📡 Fetching from Finnhub...")
        finnhub_client = FinnhubClient(finnhub_key)

        news = finnhub_client.fetch_forex_news()

        if news:
            sentiments = finnhub_client.parse_to_sentiment(news, analyzer)
            for sentiment in sentiments:
                analyzer.add_sentiment(sentiment)
            print(f"  ✓ Added {len(sentiments)} sentiment scores")
    else:
        print("\n⚠️  Finnhub key not set, skipping...")

    # 3. Fetch from MT5 Calendar (if available)
    print("\n📡 Fetching from MT5 Economic Calendar...")
    mt5_client = MT5CalendarClient()

    events = mt5_client.fetch_calendar_events(hours_back=24)

    if events:
        sentiments = mt5_client.parse_to_sentiment(events, analyzer)
        for sentiment in sentiments:
            analyzer.add_sentiment(sentiment)
        print(f"  ✓ Added {len(sentiments)} sentiment scores")
    else:
        print("  ⚠️  No MT5 calendar data available")

    # Summary
    print("\n" + "="*60)
    print("SENTIMENT SUMMARY")
    print("="*60)

    summary = analyzer.get_recent_sentiment_summary(hours=24)

    print(f"Total Events: {summary['total']}")
    print("\nBy Currency:")
    for currency, avg_score in summary['by_currency'].items():
        sentiment_label = "POSITIVE 📈" if avg_score > 55 else "NEGATIVE 📉" if avg_score < 45 else "NEUTRAL ➡️"
        print(f"  {currency}: {avg_score:.1f} - {sentiment_label}")

    print("\nBy Source:")
    for source, count in summary['by_source'].items():
        print(f"  {source}: {count}")

    # Generate trading signal
    print("\n" + "="*60)
    print("TRADING SIGNAL")
    print("="*60)

    signal, confidence, details = analyzer.generate_signal("EUR", "USD")

    if signal:
        print(f"\n🎯 Signal: {signal} EURUSD")
        print(f"💪 Confidence: {confidence:.1f}")
        print(f"\nDetails:")
        print(f"  EUR Sentiment: {details['base_sentiment']:.1f} ({details['base_count']} events)")
        print(f"  USD Sentiment: {details['quote_sentiment']:.1f} ({details['quote_count']} events)")
        print(f"  Difference: {details['sentiment_diff']:.1f}")

        print("\n📋 Recommendation:")
        if signal == "BUY":
            print("  📈 EUR sentiment stronger than USD")
            print("  💡 Consider buying EURUSD")
        else:
            print("  📉 USD sentiment stronger than EUR")
            print("  💡 Consider selling EURUSD")

        print("  ⚠️  Confirm with ML prediction before trading")
    else:
        print("\n➡️  No clear signal")
        print(f"   Sentiment difference ({details['sentiment_diff']:.1f}) below threshold")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
