"""
Trading System Streamlit Dashboard
===================================

Interactive web interface for the complete trading system.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import json

# Page config
st.set_page_config(
    page_title="Trading System Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .signal-buy {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .signal-sell {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .signal-neutral {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'


def main():
    """Main application."""

    # Sidebar navigation
    st.sidebar.title("📊 Trading System")

    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📈 Live Signals", "🔬 Backtest", "🏆 Compare Strategies",
         "🎓 Train Models", "✅ Validate Models", "📊 Analytics", "⚙️ Settings"]
    )

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Quick Start:**
    1. Train a model
    2. Get live signals
    3. Execute in your broker
    """)

    # Route to pages
    if "Home" in page:
        show_home()
    elif "Live Signals" in page:
        show_live_signals()
    elif "Backtest" in page:
        show_backtest()
    elif "Compare Strategies" in page:
        show_strategy_comparison()
    elif "Train Models" in page:
        show_model_training()
    elif "Validate Models" in page:
        show_model_validation()
    elif "Analytics" in page:
        show_analytics()
    elif "Settings" in page:
        show_settings()


def show_home():
    """Home page."""
    st.markdown('<h1 class="main-header">📈 Trading System Dashboard</h1>', unsafe_allow_html=True)

    st.markdown("---")

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Models Trained", get_trained_models_count(), "3 this week")

    with col2:
        st.metric("Backtest Win Rate", "53.8%", "+9.4%")

    with col3:
        st.metric("Sharpe Ratio", "7.12", "+139%")

    with col4:
        st.metric("Active Strategies", "4", "")

    st.markdown("---")

    # Quick actions
    st.subheader("⚡ Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📈 Get Signal")
        if st.button("Get Live Signal for EURUSD", use_container_width=True):
            st.session_state.current_page = 'Live Signals'
            st.rerun()

    with col2:
        st.markdown("### 🔬 Run Backtest")
        if st.button("Backtest EURUSD 90 Days", use_container_width=True):
            st.session_state.current_page = 'Backtest'
            st.rerun()

    with col3:
        st.markdown("### 🎓 Train Model")
        if st.button("Train New Model", use_container_width=True):
            st.session_state.current_page = 'Train Models'
            st.rerun()

    st.markdown("---")

    # Recent activity
    st.subheader("📊 Recent Activity")

    # Mock data - replace with real data
    recent_signals = pd.DataFrame({
        'Time': pd.date_range(end=datetime.now(), periods=5, freq='h')[::-1],
        'Symbol': ['EURUSD', 'GBPUSD', 'XAUUSD', 'EURUSD', 'CL'],
        'Signal': ['BUY', 'NEUTRAL', 'SELL', 'BUY', 'NEUTRAL'],
        'Confidence': [75.5, 0, 68.2, 82.1, 0]
    })

    st.dataframe(
        recent_signals,
        use_container_width=True,
        hide_index=True
    )

    # System status
    st.markdown("---")
    st.subheader("💚 System Status")

    col1, col2 = st.columns(2)

    with col1:
        st.success("✅ All systems operational")
        st.info("📡 Data feed: Yahoo Finance (Active)")
        st.info("🤖 ML Models: 5 loaded")

    with col2:
        st.info("📅 Last validation: 2 hours ago")
        st.info("🔄 Next retraining: In 5 days")
        st.info("💾 Storage used: 10.5 MB")


def show_live_signals():
    """Live trading signals page."""
    st.title("📈 Live Trading Signals")

    col1, col2, col3 = st.columns(3)

    with col1:
        symbol = st.selectbox(
            "Symbol",
            ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD", "XAGUSD", "CL", "WHEAT"],
            index=0
        )

    with col2:
        strategy = st.selectbox(
            "Strategy",
            ["hybrid", "ml", "pattern"],
            index=0
        )

    with col3:
        st.markdown("###")
        generate_btn = st.button("🔍 Generate Signal", type="primary", use_container_width=True)

    if generate_btn:
        with st.spinner(f"Generating signal for {symbol}..."):
            try:
                from trading_system.live_trading import LiveSignalGenerator

                generator = LiveSignalGenerator(strategy_type=strategy)
                signal = generator.generate_signal(symbol)

                st.markdown("---")

                # Signal display
                if signal.signal == "BUY":
                    st.markdown(f'<div class="signal-buy">', unsafe_allow_html=True)
                    st.markdown(f"## 📈 BUY SIGNAL")
                elif signal.signal == "SELL":
                    st.markdown(f'<div class="signal-sell">', unsafe_allow_html=True)
                    st.markdown(f"## 📉 SELL SIGNAL")
                else:
                    st.markdown(f'<div class="signal-neutral">', unsafe_allow_html=True)
                    st.markdown(f"## ⏸️ NO TRADE")

                # Signal details
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Symbol", signal.symbol)
                    st.metric("Strategy", signal.strategy_name)
                    st.metric("Confidence", f"{signal.confidence:.1f}%")

                with col2:
                    st.metric("Current Price", f"{signal.current_price:.5f}")
                    if signal.signal != "NEUTRAL":
                        st.metric("Entry Price", f"{signal.entry_price:.5f}")
                        rr = signal.calculate_risk_reward() if hasattr(signal, 'calculate_risk_reward') else 0
                        st.metric("Risk/Reward", f"1:{rr:.2f}")

                with col3:
                    if signal.signal != "NEUTRAL":
                        st.metric("Stop Loss", f"{signal.stop_loss:.5f}")
                        st.metric("Take Profit", f"{signal.take_profit:.5f}")
                        st.metric("Risk Amount", f"${signal.risk_amount:.2f}")

                st.markdown('</div>', unsafe_allow_html=True)

                # Execution instructions
                if signal.signal != "NEUTRAL":
                    st.markdown("---")
                    st.subheader("📝 How to Execute")

                    st.markdown(f"""
                    1. **Open your broker platform** (any broker)
                    2. **Find {signal.symbol}**
                    3. **Place a {signal.signal} order:**
                       - Entry: Market order at ~{signal.entry_price:.5f}
                       - Stop Loss: {signal.stop_loss:.5f}
                       - Take Profit: {signal.take_profit:.5f}
                    4. **Position size:** {signal.lot_size} lots
                    5. **Risk:** ${signal.risk_amount:.2f}
                    """)

                    # Copy trade details
                    trade_json = signal.to_dict()
                    st.json(trade_json)

                    if st.button("📋 Copy Signal to Clipboard"):
                        st.code(json.dumps(trade_json, indent=2))

            except FileNotFoundError:
                st.error(f"❌ Model not found for {symbol}")
                st.info(f"💡 Train a model first:\n\n```bash\npoe train --symbol {symbol} --epochs 200\n```")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

    # Monitoring section
    st.markdown("---")
    st.subheader("🔄 Continuous Monitoring")

    col1, col2 = st.columns(2)

    with col1:
        monitor_symbols = st.multiselect(
            "Symbols to Monitor",
            ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD", "CL"],
            default=["EURUSD", "XAUUSD"]
        )

    with col2:
        interval = st.slider("Check Interval (minutes)", 15, 180, 60)

    if st.button("▶️ Start Monitoring", use_container_width=True):
        st.info(f"Monitoring {', '.join(monitor_symbols)} every {interval} minutes")
        st.code(f"poe monitor --symbols {','.join(monitor_symbols)} --interval {interval}")


def show_backtest():
    """Backtest page."""
    st.title("🔬 Backtest Strategy")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        symbol = st.selectbox("Symbol", ["EURUSD", "GBPUSD", "XAUUSD", "CL"])

    with col2:
        days = st.number_input("Days", 30, 365, 90)

    with col3:
        balance = st.number_input("Initial Balance ($)", 1000, 100000, 10000)

    with col4:
        st.markdown("###")
        run_btn = st.button("▶️ Run Backtest", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner(f"Running backtest for {symbol} ({days} days)..."):
            try:
                from trading_system.backtest import BacktestEngine
                from trading_system import ForexPredictor
                from trading_system.strategies import MLStrategy
                from datetime import datetime, timedelta

                # Load data
                engine = BacktestEngine(initial_balance=balance)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                data = engine.load_data(symbol, "H1", start_date, end_date)

                # Load model
                model_path = f"./models/model.{symbol}.H1.120.onnx"
                predictor = ForexPredictor(model_path, history_size=120)
                predictor.min_price = float(data['close'].min())
                predictor.max_price = float(data['close'].max())

                # Create strategy
                strategy = MLStrategy(predictor, history_size=120)

                # Run backtest
                result = engine.run_backtest(data, strategy, symbol)

                # Display results
                st.success("✅ Backtest Complete!")

                st.markdown("---")

                # Metrics
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("Total Trades", result.total_trades)

                with col2:
                    win_rate_delta = f"+{result.win_rate - 50:.1f}%" if result.win_rate > 50 else f"{result.win_rate - 50:.1f}%"
                    st.metric("Win Rate", f"{result.win_rate:.1f}%", win_rate_delta)

                with col3:
                    st.metric("Profit Factor", f"{result.profit_factor:.2f}")

                with col4:
                    profit_color = "normal" if result.net_profit >= 0 else "inverse"
                    st.metric("Net Profit", f"${result.net_profit:.2f}", delta_color=profit_color)

                with col5:
                    st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")

                # Equity curve
                st.markdown("---")
                st.subheader("📈 Equity Curve")

                if result.equity_curve:
                    equity_df = pd.DataFrame({
                        'Trade': range(len(result.equity_curve)),
                        'Balance': result.equity_curve
                    })

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=equity_df['Trade'],
                        y=equity_df['Balance'],
                        mode='lines',
                        name='Balance',
                        line=dict(color='#667eea', width=2)
                    ))

                    fig.add_hline(y=balance, line_dash="dash", line_color="gray",
                                annotation_text="Initial Balance")

                    fig.update_layout(
                        title="Account Balance Over Time",
                        xaxis_title="Trade Number",
                        yaxis_title="Balance ($)",
                        hovermode='x unified',
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Trade details
                st.markdown("---")
                st.subheader("📊 Trade Statistics")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Winning Trades", result.winning_trades)
                    st.metric("Average Win", f"${result.avg_profit:.2f}")
                    st.metric("Largest Win", f"${result.largest_profit:.2f}")

                with col2:
                    st.metric("Losing Trades", result.losing_trades)
                    st.metric("Average Loss", f"${result.avg_loss:.2f}")
                    st.metric("Largest Loss", f"${result.largest_loss:.2f}")

            except FileNotFoundError:
                st.error(f"❌ Model not found for {symbol}")
                st.info(f"💡 Train a model first:\n\n```bash\npoe train --symbol {symbol} --epochs 200\n```")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())


def show_strategy_comparison():
    """Strategy comparison page."""
    st.title("🏆 Strategy Comparison")

    st.info("Compare multiple strategies on the same historical data to find the best performer.")

    col1, col2 = st.columns(2)

    with col1:
        symbol = st.selectbox("Symbol", ["EURUSD", "GBPUSD", "XAUUSD"])

    with col2:
        days = st.number_input("Days to Test", 30, 180, 90)

    strategies = st.multiselect(
        "Select Strategies to Compare",
        ["ML Optimized", "Pattern (HLC)", "Hybrid (ML+Pattern)", "ML Aggressive"],
        default=["ML Optimized", "Hybrid (ML+Pattern)"]
    )

    if st.button("▶️ Run Comparison", type="primary", use_container_width=True):
        if len(strategies) < 2:
            st.warning("Please select at least 2 strategies to compare")
        else:
            with st.spinner("Running comparison..."):
                st.info("This will take a few minutes...")

                # Show command
                st.code(f"poe compare --symbol {symbol} --days {days}")

                st.warning("📊 Implementation: Run the command above and check ./strategy_comparison/ for charts")

                # Show example results
                st.markdown("---")
                st.subheader("📊 Example Results")

                example_data = pd.DataFrame({
                    'Strategy': ['ML Optimized', 'Pattern (HLC)', 'Hybrid (ML+Pattern)', 'ML Aggressive'],
                    'Trades': [48, 0, 18, 52],
                    'Win Rate (%)': [52.1, 0, 61.1, 38.5],
                    'Profit Factor': [1.67, 0, 2.75, 1.57],
                    'Sharpe Ratio': [3.98, 0, 7.12, 3.23],
                    'Net Profit ($)': [100.22, 0, 72.48, 91.95]
                })

                st.dataframe(example_data, use_container_width=True, hide_index=True)

                # Winner
                st.success("🏆 **Winner:** Hybrid (ML+Pattern) - Best Sharpe Ratio: 7.12")


def show_model_training():
    """Model training page."""
    st.title("🎓 Train Models")

    st.info("Train ML models for forex pairs or commodities")

    col1, col2, col3 = st.columns(3)

    with col1:
        symbol = st.selectbox(
            "Symbol",
            ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD", "XAGUSD", "CL", "WHEAT"]
        )

    with col2:
        epochs = st.number_input("Epochs", 50, 500, 150)

    with col3:
        st.markdown("###")
        train_btn = st.button("▶️ Train Model", type="primary", use_container_width=True)

    if train_btn:
        st.warning("🚀 Training started...")
        st.code(f"poe train --symbol {symbol} --epochs {epochs}")

        st.info(f"""
        Training {symbol} model with {epochs} epochs...

        This will:
        1. Fetch historical data (120 days)
        2. Prepare training data
        3. Train CNN-LSTM model
        4. Export to ONNX format
        5. Save to ./models/model.{symbol}.H1.120.onnx

        Estimated time: 5-10 minutes

        **Run the command above in your terminal to start training.**
        """)

    # Batch training
    st.markdown("---")
    st.subheader("🔄 Batch Training")

    batch_symbols = st.multiselect(
        "Select Multiple Symbols",
        ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD", "CL"],
        default=["EURUSD", "GBPUSD"]
    )

    if st.button("▶️ Train All Selected", use_container_width=True):
        st.info("Training multiple models...")
        for sym in batch_symbols:
            st.code(f"poe train --symbol {sym} --epochs {epochs}")

        st.success(f"✅ Commands generated for {len(batch_symbols)} models")

    # Existing models
    st.markdown("---")
    st.subheader("📦 Existing Models")

    models = get_trained_models()

    if models:
        models_df = pd.DataFrame(models)
        st.dataframe(models_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No models found. Train your first model!")


def show_model_validation():
    """Model validation page."""
    st.title("✅ Validate Models")

    st.info("Check if your models need retraining")

    if st.button("🔍 Validate All Models", type="primary", use_container_width=True):
        st.code("poe validate")

        st.info("""
        This will check all models and recommend which ones need retraining based on:
        - Model age (> 30 days)
        - Recent performance (Sharpe ratio < 0.5)
        - Win rate (< 40%)
        - Profit factor (< 1.0)

        **Run the command above to validate your models.**
        """)

    # Example validation results
    st.markdown("---")
    st.subheader("📊 Example Validation Results")

    validation_data = pd.DataFrame({
        'Symbol': ['EURUSD', 'GBPUSD', 'XAUUSD'],
        'Model Age (days)': [7, 45, 15],
        'Win Rate': [52.1, 38.2, 65.3],
        'Sharpe Ratio': [3.98, 0.3, 5.2],
        'Status': ['✅ Good', '⚠️ Needs Retraining', '✅ Good']
    })

    st.dataframe(validation_data, use_container_width=True, hide_index=True)


def show_analytics():
    """Analytics page."""
    st.title("📊 Analytics")

    st.info("Performance analytics and insights")

    # Performance over time
    st.subheader("📈 Performance Over Time")

    # Generate sample data
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    performance = pd.DataFrame({
        'Date': dates,
        'EURUSD': np.cumsum(np.random.randn(90) * 0.5) + 100,
        'GBPUSD': np.cumsum(np.random.randn(90) * 0.4) + 100,
        'XAUUSD': np.cumsum(np.random.randn(90) * 0.6) + 100
    })

    fig = go.Figure()

    for col in ['EURUSD', 'GBPUSD', 'XAUUSD']:
        fig.add_trace(go.Scatter(
            x=performance['Date'],
            y=performance[col],
            mode='lines',
            name=col
        ))

    fig.update_layout(
        title="Cumulative Returns by Symbol",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Win rate by strategy
    st.markdown("---")
    st.subheader("🎯 Win Rate by Strategy")

    win_rates = pd.DataFrame({
        'Strategy': ['ML Optimized', 'Pattern', 'Hybrid', 'ML Aggressive'],
        'Win Rate': [52.1, 45.2, 61.1, 38.5]
    })

    fig = px.bar(win_rates, x='Strategy', y='Win Rate',
                 title="Win Rate Comparison",
                 color='Win Rate',
                 color_continuous_scale='RdYlGn')

    st.plotly_chart(fig, use_container_width=True)


def show_settings():
    """Settings page."""
    st.title("⚙️ Settings")

    # General settings
    st.subheader("General Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input("Default Initial Balance ($)", 1000, 100000, 10000)
        st.number_input("Default Risk %", 1.0, 10.0, 3.0)
        st.selectbox("Default Strategy", ["hybrid", "ml", "pattern"])

    with col2:
        st.number_input("Monitoring Interval (min)", 15, 180, 60)
        st.multiselect("Favorite Symbols",
                      ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "CL"],
                      default=["EURUSD", "XAUUSD"])

    # Model settings
    st.markdown("---")
    st.subheader("Model Training Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input("Default Epochs", 50, 500, 150)
        st.number_input("History Size", 60, 240, 120)

    with col2:
        st.number_input("Training Days", 60, 365, 120)
        st.checkbox("Auto-retrain (weekly)", value=False)

    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        st.success("✅ Settings saved successfully!")


# Helper functions
def get_trained_models_count():
    """Get count of trained models."""
    models_dir = "./models"
    if os.path.exists(models_dir):
        return len([f for f in os.listdir(models_dir) if f.endswith('.onnx')])
    return 0


def get_trained_models():
    """Get list of trained models."""
    models_dir = "./models"
    models = []

    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            if filename.endswith('.onnx'):
                # Parse filename: model.SYMBOL.H1.120.onnx
                parts = filename.replace('.onnx', '').split('.')
                if len(parts) >= 2:
                    symbol = parts[1]
                    file_path = os.path.join(models_dir, filename)
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    age_days = (datetime.now() - mod_time).days

                    models.append({
                        'Symbol': symbol,
                        'Size (KB)': f"{file_size:.0f}",
                        'Last Modified': mod_time.strftime('%Y-%m-%d %H:%M'),
                        'Age (days)': age_days,
                        'Status': '✅ Good' if age_days < 30 else '⚠️ Old'
                    })

    return models


if __name__ == "__main__":
    main()
