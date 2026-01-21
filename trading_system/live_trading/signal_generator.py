"""
Live Trading Signal Generator
==============================

Generate real-time trading signals for manual or automated execution.
Works without MetaTrader - use any broker!
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import logging

from ..ml_predictor import ForexPredictor
from ..strategies import MLStrategy, PatternStrategy, HybridStrategy
from ..ml_predictor.correlations import get_commodity_spec, is_commodity


logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """
    Complete trading signal with all information needed to place a trade.
    """
    # Basic info
    symbol: str
    timestamp: datetime
    signal: str  # "BUY", "SELL", "NEUTRAL"

    # Entry details
    current_price: float
    entry_price: float  # Recommended entry (might be limit order)

    # Risk management
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None

    # Position sizing
    lot_size: float = 0.1
    risk_amount: float = 0.0  # $ at risk

    # Signal quality
    confidence: float = 0.0  # 0-100
    strategy_name: str = ""
    pattern_type: Optional[str] = None

    # Additional info
    atr: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

        # Calculate risk amount
        if self.signal in ["BUY", "SELL"]:
            risk_pips = abs(self.entry_price - self.stop_loss)
            self.risk_amount = risk_pips * 10.0 * self.lot_size  # Simplified

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'signal': self.signal,
            'current_price': self.current_price,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trailing_stop': self.trailing_stop,
            'lot_size': self.lot_size,
            'risk_amount': self.risk_amount,
            'confidence': self.confidence,
            'strategy_name': self.strategy_name,
            'pattern_type': self.pattern_type,
            'atr': self.atr,
            'metadata': self.metadata
        }

    def print_signal(self):
        """Print formatted trading signal."""
        print("\n" + "="*70)
        print(f"{'🔔 TRADING SIGNAL':^70}")
        print("="*70)

        print(f"\n{'Symbol:':<20} {self.symbol}")
        print(f"{'Time:':<20} {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'Strategy:':<20} {self.strategy_name}")

        # Signal direction with emoji
        signal_emoji = "📈" if self.signal == "BUY" else "📉" if self.signal == "SELL" else "⏸️"
        signal_color = self.signal if self.signal != "NEUTRAL" else "NO TRADE"
        print(f"{'Signal:':<20} {signal_emoji} {signal_color}")

        if self.signal != "NEUTRAL":
            print(f"\n{'Current Price:':<20} {self.current_price:.5f}")
            print(f"{'Entry Price:':<20} {self.entry_price:.5f}")
            print(f"{'Stop Loss:':<20} {self.stop_loss:.5f}")
            print(f"{'Take Profit:':<20} {self.take_profit:.5f}")

            if self.trailing_stop:
                print(f"{'Trailing Stop:':<20} {self.trailing_stop:.5f}")

            # Risk/Reward
            if self.signal == "BUY":
                risk = self.entry_price - self.stop_loss
                reward = self.take_profit - self.entry_price
            else:
                risk = self.stop_loss - self.entry_price
                reward = self.entry_price - self.take_profit

            rr_ratio = reward / risk if risk > 0 else 0
            print(f"\n{'Risk/Reward:':<20} 1:{rr_ratio:.2f}")
            print(f"{'Risk Amount:':<20} ${self.risk_amount:.2f}")
            print(f"{'Lot Size:':<20} {self.lot_size}")
            print(f"{'Confidence:':<20} {self.confidence:.1f}%")

            if self.pattern_type:
                print(f"{'Pattern:':<20} {self.pattern_type}")

        print("\n" + "="*70 + "\n")


class LiveSignalGenerator:
    """
    Generate live trading signals using trained models.

    Works without MetaTrader - fetches data from Yahoo Finance.
    """

    def __init__(self, strategy_type: str = "hybrid", model_path: Optional[str] = None):
        """
        Initialize signal generator.

        Args:
            strategy_type: "ml", "pattern", or "hybrid"
            model_path: Path to ONNX model (optional, auto-detected if not provided)
        """
        self.strategy_type = strategy_type
        self.model_path = model_path
        self.predictor = None
        self.strategy = None

    def _load_model(self, symbol: str):
        """Load ML model for symbol."""
        if self.model_path is None:
            # Auto-detect model path
            self.model_path = f"./models/model.{symbol}.H1.120.onnx"

        try:
            self.predictor = ForexPredictor(self.model_path, history_size=120)
            logger.info(f"Loaded model: {self.model_path}")
        except FileNotFoundError:
            logger.warning(f"Model not found: {self.model_path}")
            logger.warning(f"Train model first: poe train --symbol {symbol} --epochs 200")
            raise

    def _fetch_live_data(self, symbol: str, bars: int = 200) -> pd.DataFrame:
        """
        Fetch live market data from Yahoo Finance.

        Args:
            symbol: Trading symbol
            bars: Number of bars to fetch

        Returns:
            DataFrame with OHLC data
        """
        # Convert symbol to Yahoo Finance format
        if symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'USDCHF']:
            yf_symbol = f"{symbol}=X"  # Forex
        elif symbol.startswith('XAU'):
            yf_symbol = "GC=F"  # Gold futures
        elif symbol.startswith('XAG'):
            yf_symbol = "SI=F"  # Silver futures
        elif symbol == 'CL':
            yf_symbol = "CL=F"  # Crude oil
        elif symbol == 'BRENT':
            yf_symbol = "BZ=F"  # Brent oil
        else:
            yf_symbol = symbol

        try:
            # Fetch hourly data
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period="1mo", interval="1h")

            if data.empty:
                raise ValueError(f"No data received for {symbol}")

            # Rename columns to match our format
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Add time column
            data['time'] = data.index

            # Get last N bars
            data = data.tail(bars)

            logger.info(f"Fetched {len(data)} bars for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise

    def _initialize_strategy(self, symbol: str):
        """Initialize trading strategy."""
        if self.strategy_type in ["ml", "hybrid"]:
            self._load_model(symbol)

        if self.strategy_type == "ml":
            self.strategy = MLStrategy(
                predictor=self.predictor,
                history_size=120,
                sl_atr_mult=2.0,
                tp_atr_mult=4.0,
                trailing_atr_mult=1.0,
                min_rr=2.0
            )
        elif self.strategy_type == "pattern":
            from ..strategies import PatternStrategy
            self.strategy = PatternStrategy(
                swing_lookback=20,
                atr_period=14,
                min_strength=70,
                min_rr=2.0
            )
        elif self.strategy_type == "hybrid":
            from ..strategies import HybridStrategy
            self.strategy = HybridStrategy(
                predictor=self.predictor,
                history_size=120,
                swing_lookback=20,
                atr_period=14,
                min_pattern_strength=60,
                min_rr=2.0
            )
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")

    def generate_signal(self, symbol: str) -> TradingSignal:
        """
        Generate live trading signal for a symbol.

        Args:
            symbol: Trading symbol (e.g., "EURUSD", "XAUUSD", "CL")

        Returns:
            TradingSignal object with entry, SL, TP

        Example:
            >>> generator = LiveSignalGenerator(strategy_type="hybrid")
            >>> signal = generator.generate_signal("EURUSD")
            >>> signal.print_signal()
        """
        # Initialize strategy if needed
        if self.strategy is None:
            self._initialize_strategy(symbol)

        # Fetch live data
        data = self._fetch_live_data(symbol, bars=250)

        # Set price normalization for ML models
        if self.predictor:
            self.predictor.min_price = float(data['close'].min())
            self.predictor.max_price = float(data['close'].max())

        # Generate strategy signal
        current_index = len(data) - 1
        strategy_signal = self.strategy.generate_signal(data, current_index)

        current_price = float(data.iloc[-1]['close'])
        timestamp = datetime.now()

        if strategy_signal is None:
            # No signal
            return TradingSignal(
                symbol=symbol,
                timestamp=timestamp,
                signal="NEUTRAL",
                current_price=current_price,
                entry_price=current_price,
                stop_loss=current_price,
                take_profit=current_price,
                strategy_name=self.strategy.get_name()
            )

        # Create trading signal
        signal = TradingSignal(
            symbol=symbol,
            timestamp=timestamp,
            signal=strategy_signal.signal,
            current_price=current_price,
            entry_price=current_price,  # Market order
            stop_loss=strategy_signal.stop_loss,
            take_profit=strategy_signal.take_profit,
            trailing_stop=strategy_signal.trailing_stop,
            lot_size=strategy_signal.lot_size,
            confidence=strategy_signal.confidence,
            strategy_name=self.strategy.get_name(),
            pattern_type=strategy_signal.pattern_type,
            atr=strategy_signal.metadata.get('atr', 0.0) if strategy_signal.metadata else 0.0,
            metadata=strategy_signal.metadata
        )

        return signal

    def monitor_symbols(self, symbols: list, interval_minutes: int = 60):
        """
        Continuously monitor multiple symbols and print signals.

        Args:
            symbols: List of symbols to monitor
            interval_minutes: Check interval in minutes

        Example:
            >>> generator = LiveSignalGenerator(strategy_type="hybrid")
            >>> generator.monitor_symbols(["EURUSD", "GBPUSD", "XAUUSD"])
        """
        import time

        print(f"\n🤖 Live Signal Monitor Started")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Strategy: {self.strategy_type}")
        print(f"Interval: {interval_minutes} minutes")
        print(f"Press Ctrl+C to stop\n")

        try:
            while True:
                for symbol in symbols:
                    try:
                        signal = self.generate_signal(symbol)

                        if signal.signal != "NEUTRAL":
                            signal.print_signal()

                            # Optional: Send notification (email, Telegram, etc.)
                            # self._send_notification(signal)

                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")

                # Wait for next check
                print(f"Next check in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            print("\n\n✋ Monitor stopped by user")
