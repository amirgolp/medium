"""
Example: Complete Trading Strategy
===================================

Combine ML predictions, sentiment analysis, and risk management
for complete trading decisions.
"""

from trading_system import (
    ForexPredictor,
    SentimentAnalyzer,
    PositionSizer,
    RiskManager,
    ATRCalculator
)
from trading_system.sentiment_analyzer import NewsAPIClient, FinnhubClient
from trading_system.risk_management import LotMode
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """Execute complete trading strategy."""

    symbol = "EURUSD"
    base_currency = "EUR"
    quote_currency = "USD"

    print("="*60)
    print("COMPLETE TRADING STRATEGY")
    print("="*60)
    print(f"Symbol: {symbol}")
    print(f"Strategy: ML + Sentiment + Risk Management")
    print("="*60)

    # ========================================
    # STEP 1: RISK MANAGEMENT CHECK
    # ========================================
    print("\n[1/5] Risk Management Check...")

    risk_mgr = RiskManager(
        max_positions=1,
        max_daily_loss_pct=5.0,
        holding_period_hours=24,
        magic_number=123456
    )

    can_trade, reason = risk_mgr.can_open_position(symbol)

    if not can_trade:
        print(f"  ❌ Trading not allowed: {reason}")
        risk_mgr.log_portfolio_status()
        return

    print(f"  ✅ Trading allowed")

    # Log portfolio status
    status = risk_mgr.get_portfolio_status()
    print(f"  Positions: {status['position_count']}/{status['max_positions']}")
    print(f"  Daily PnL: ${status['daily_pnl']:.2f}")

    # ========================================
    # STEP 2: ML PREDICTION
    # ========================================
    print("\n[2/5] ML Prediction...")

    model_path = f"./models/model.{symbol}.H1.120.onnx"

    try:
        predictor = ForexPredictor(model_path, history_size=120)
        predictor.update_min_max(symbol, lookback_hours=2880)

        ml_prediction, pred_price, last_close = predictor.predict_symbol(symbol)
        ml_signal = predictor.get_signal_name(ml_prediction)

        print(f"  Current: {last_close:.5f}")
        print(f"  Predicted: {pred_price:.5f}")
        print(f"  ML Signal: {ml_signal}")

    except Exception as e:
        print(f"  ⚠️  ML prediction failed: {e}")
        print(f"  Continuing without ML signal...")
        ml_signal = None

    # ========================================
    # STEP 3: SENTIMENT ANALYSIS
    # ========================================
    print("\n[3/5] Sentiment Analysis...")

    analyzer = SentimentAnalyzer(
        lookback_hours=24,
        min_sentiment_score=10.0
    )

    # Fetch news (if API keys available)
    newsapi_key = os.getenv("NEWSAPI_KEY")
    finnhub_key = os.getenv("FINNHUB_KEY")

    total_sentiments = 0

    if newsapi_key:
        try:
            news_client = NewsAPIClient(newsapi_key)
            articles = news_client.fetch_news(hours_back=24)
            sentiments = news_client.parse_to_sentiment(articles, analyzer)

            for s in sentiments:
                analyzer.add_sentiment(s)

            total_sentiments += len(sentiments)
        except Exception as e:
            print(f"  ⚠️  NewsAPI error: {e}")

    if finnhub_key:
        try:
            finnhub_client = FinnhubClient(finnhub_key)
            news = finnhub_client.fetch_forex_news()
            sentiments = finnhub_client.parse_to_sentiment(news, analyzer)

            for s in sentiments:
                analyzer.add_sentiment(s)

            total_sentiments += len(sentiments)
        except Exception as e:
            print(f"  ⚠️  Finnhub error: {e}")

    print(f"  Analyzed {total_sentiments} news items")

    # Generate sentiment signal
    sent_signal, confidence, details = analyzer.generate_signal(base_currency, quote_currency)

    if sent_signal:
        print(f"  Sentiment Signal: {sent_signal}")
        print(f"  Confidence: {confidence:.1f}")
        print(f"  {base_currency}: {details['base_sentiment']:.1f} | {quote_currency}: {details['quote_sentiment']:.1f}")
    else:
        print(f"  No sentiment signal (diff: {details['sentiment_diff']:.1f})")

    # ========================================
    # STEP 4: COMBINE SIGNALS
    # ========================================
    print("\n[4/5] Signal Combination...")

    if ml_signal and sent_signal:
        if ml_signal == sent_signal:
            final_signal = ml_signal
            print(f"  ✅ Signals ALIGNED: {final_signal}")
        else:
            print(f"  ⚠️  Signals CONFLICT: ML={ml_signal}, Sentiment={sent_signal}")
            print(f"  No trade - waiting for alignment")
            return
    elif ml_signal:
        final_signal = ml_signal
        print(f"  ⚠️  Using ML signal only: {final_signal}")
    elif sent_signal:
        final_signal = sent_signal
        print(f"  ⚠️  Using sentiment signal only: {final_signal}")
    else:
        print(f"  ❌ No signals available")
        return

    # ========================================
    # STEP 5: POSITION SIZING
    # ========================================
    print("\n[5/5] Position Sizing...")

    # Calculate ATR-based levels
    atr_calc = ATRCalculator(period=14)
    sl_price, tp_price, atr = atr_calc.calculate_sl_tp(
        symbol=symbol,
        signal=final_signal,
        sl_multiplier=2.5,
        tp_multiplier=3.0
    )

    if sl_price is None:
        print(f"  ⚠️  Could not calculate ATR levels")
        # Use default values
        sl_pips = 50.0
    else:
        # Calculate SL distance in pips
        pip_value = 0.0001 if "JPY" not in symbol else 0.01
        sl_distance = abs(last_close - sl_price)
        sl_pips = sl_distance / pip_value

    # Calculate lot size
    sizer = PositionSizer(
        mode=LotMode.RISK_BASED,
        risk_percent=3.0,
        max_lot=10.0
    )

    lot = sizer.calculate_lot(symbol, sl_distance_pips=sl_pips)

    print(f"  ATR: {atr:.5f}" if atr else "  ATR: N/A")
    print(f"  Entry: {last_close:.5f}")
    print(f"  Stop Loss: {sl_price:.5f} ({sl_pips:.1f} pips)" if sl_price else f"  Stop Loss: {sl_pips:.1f} pips")
    print(f"  Take Profit: {tp_price:.5f}" if tp_price else "  Take Profit: N/A")
    print(f"  Position Size: {lot:.2f} lots")
    print(f"  Risk: 3% of account")

    # ========================================
    # TRADING DECISION
    # ========================================
    print("\n" + "="*60)
    print("TRADING DECISION")
    print("="*60)

    print(f"\n🎯 SIGNAL: {final_signal} {symbol}")
    print(f"📊 LOT SIZE: {lot:.2f}")
    print(f"🛑 STOP LOSS: {sl_price:.5f}" if sl_price else f"🛑 STOP LOSS: {sl_pips:.1f} pips")
    print(f"🎯 TAKE PROFIT: {tp_price:.5f}" if tp_price else "🎯 TAKE PROFIT: N/A")

    print("\n💡 NEXT STEPS:")
    print("  1. Verify MT5 connection")
    print("  2. Check market conditions")
    print("  3. Execute trade via MT5 API")
    print("  4. Monitor position with trailing stop")

    print("\n⚠️  RISK WARNING:")
    print("  - This is for educational purposes only")
    print("  - Always test with paper trading first")
    print("  - Past performance ≠ future results")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
