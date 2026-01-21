"""
Backtesting Engine
==================

Simulate trading strategies on historical data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Callable
from dataclasses import dataclass, field
import logging

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    symbol: str = ""
    direction: str = ""  # "BUY" or "SELL"
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    lot_size: float = 0.0

    # Results
    profit_loss: float = 0.0
    profit_loss_pct: float = 0.0
    pips: float = 0.0
    duration_hours: float = 0.0
    exit_reason: str = ""  # "TP", "SL", "TIME", "SIGNAL"

    # Metadata
    ml_signal: Optional[str] = None
    sentiment_signal: Optional[str] = None
    sentiment_score: float = 0.0

    def close_trade(self, exit_time: datetime, exit_price: float, reason: str):
        """Close the trade and calculate results."""
        from ..ml_predictor.correlations import get_commodity_spec

        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = reason

        # Calculate duration
        self.duration_hours = (exit_time - self.entry_time).total_seconds() / 3600.0

        # Get commodity specs if applicable
        commodity_spec = get_commodity_spec(self.symbol)

        if commodity_spec:
            # Commodity P&L calculation
            tick_size = commodity_spec['tick_size']
            tick_value = commodity_spec['tick_value']

            price_change = self.exit_price - self.entry_price if self.direction == "BUY" else self.entry_price - self.exit_price
            ticks = price_change / tick_size
            self.pips = ticks  # Store as ticks for commodities

            # P&L = ticks * tick_value * lot_size
            self.profit_loss = ticks * tick_value * self.lot_size
        else:
            # Forex P&L calculation
            pip_value = 0.0001 if "JPY" not in self.symbol else 0.01

            if self.direction == "BUY":
                self.pips = (self.exit_price - self.entry_price) / pip_value
            else:  # SELL
                self.pips = (self.entry_price - self.exit_price) / pip_value

            # Assuming 1 lot = $10 per pip for major pairs
            self.profit_loss = self.pips * 10.0 * self.lot_size

        self.profit_loss_pct = (self.profit_loss / 10000.0) * 100  # Assuming $10k account


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    # Basic stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # P&L stats
    total_profit: float = 0.0
    total_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0

    # Trade stats
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    largest_profit: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration_hours: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0

    # Equity curve
    equity_curve: List[float] = field(default_factory=list)
    equity_dates: List[datetime] = field(default_factory=list)

    # Trades
    trades: List[Trade] = field(default_factory=list)

    # Additional metrics
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_balance: float = 10000.0
    final_balance: float = 10000.0

    def calculate_metrics(self):
        """Calculate all performance metrics from trades."""
        if not self.trades:
            return

        # Basic counts
        self.total_trades = len(self.trades)
        self.winning_trades = sum(1 for t in self.trades if t.profit_loss > 0)
        self.losing_trades = sum(1 for t in self.trades if t.profit_loss < 0)

        # Win rate
        self.win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        # P&L
        profits = [t.profit_loss for t in self.trades if t.profit_loss > 0]
        losses = [t.profit_loss for t in self.trades if t.profit_loss < 0]

        self.total_profit = sum(profits) if profits else 0
        self.total_loss = abs(sum(losses)) if losses else 0
        self.net_profit = self.total_profit - self.total_loss

        # Profit factor
        self.profit_factor = (self.total_profit / self.total_loss) if self.total_loss > 0 else 0

        # Averages
        self.avg_profit = np.mean(profits) if profits else 0
        self.avg_loss = abs(np.mean(losses)) if losses else 0
        self.largest_profit = max(profits) if profits else 0
        self.largest_loss = abs(min(losses)) if losses else 0

        # Duration
        durations = [t.duration_hours for t in self.trades if t.duration_hours > 0]
        self.avg_trade_duration_hours = np.mean(durations) if durations else 0

        # Equity curve and drawdown
        self._calculate_equity_curve()
        self._calculate_drawdown()

        # Sharpe ratio
        self._calculate_sharpe_ratio()

        # Final balance
        self.final_balance = self.initial_balance + self.net_profit

    def _calculate_equity_curve(self):
        """Calculate equity curve over time."""
        self.equity_curve = [self.initial_balance]
        self.equity_dates = [self.trades[0].entry_time if self.trades else datetime.now()]

        running_balance = self.initial_balance
        for trade in self.trades:
            running_balance += trade.profit_loss
            self.equity_curve.append(running_balance)
            self.equity_dates.append(trade.exit_time or trade.entry_time)

    def _calculate_drawdown(self):
        """Calculate maximum drawdown."""
        if len(self.equity_curve) < 2:
            return

        peak = self.equity_curve[0]
        max_dd = 0

        for equity in self.equity_curve:
            if equity > peak:
                peak = equity

            drawdown = peak - equity
            if drawdown > max_dd:
                max_dd = drawdown

        self.max_drawdown = max_dd
        self.max_drawdown_pct = (max_dd / peak * 100) if peak > 0 else 0

    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio (annualized)."""
        if len(self.trades) < 2:
            return

        returns = [t.profit_loss / self.initial_balance for t in self.trades]

        if not returns:
            return

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return

        # Annualized Sharpe (assuming ~252 trading days)
        self.sharpe_ratio = (mean_return / std_return) * np.sqrt(252)

    def print_summary(self):
        """Print backtest summary."""
        print("\n" + "="*70)
        print("BACKTEST RESULTS SUMMARY")
        print("="*70)

        print(f"\n📅 Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")

        print(f"\n📊 TRADE STATISTICS")
        print(f"  Total Trades:     {self.total_trades}")
        print(f"  Winning Trades:   {self.winning_trades} ({self.win_rate:.1f}%)")
        print(f"  Losing Trades:    {self.losing_trades} ({100-self.win_rate:.1f}%)")
        print(f"  Avg Duration:     {self.avg_trade_duration_hours:.1f} hours")

        print(f"\n💰 PROFIT & LOSS")
        print(f"  Initial Balance:  ${self.initial_balance:,.2f}")
        print(f"  Final Balance:    ${self.final_balance:,.2f}")
        print(f"  Net Profit:       ${self.net_profit:,.2f} ({(self.net_profit/self.initial_balance*100):+.2f}%)")
        print(f"  Total Profit:     ${self.total_profit:,.2f}")
        print(f"  Total Loss:       ${self.total_loss:,.2f}")
        print(f"  Profit Factor:    {self.profit_factor:.2f}")

        print(f"\n📈 AVERAGE TRADES")
        print(f"  Avg Profit:       ${self.avg_profit:,.2f}")
        print(f"  Avg Loss:         ${self.avg_loss:,.2f}")
        print(f"  Largest Profit:   ${self.largest_profit:,.2f}")
        print(f"  Largest Loss:     ${self.largest_loss:,.2f}")

        print(f"\n⚠️  RISK METRICS")
        print(f"  Max Drawdown:     ${self.max_drawdown:,.2f} ({self.max_drawdown_pct:.2f}%)")
        print(f"  Sharpe Ratio:     {self.sharpe_ratio:.2f}")

        print("\n" + "="*70)


class BacktestEngine:
    """
    Backtest trading strategies on historical data.

    Supports:
    - ML prediction strategies
    - Sentiment-based strategies
    - Combined strategies
    - Custom strategy functions
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        commission: float = 0.0,  # Per lot commission
        slippage_pips: float = 1.0
    ):
        """
        Initialize backtest engine.

        Args:
            initial_balance: Starting account balance
            commission: Commission per lot traded
            slippage_pips: Average slippage in pips
        """
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage_pips = slippage_pips

        self.current_trade: Optional[Trade] = None
        self.closed_trades: List[Trade] = []

    def load_data(
        self,
        symbol: str,
        timeframe: str = "H1",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days: int = 90
    ) -> pd.DataFrame:
        """
        Load historical data for backtesting.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            start_date: Start date
            end_date: End date
            days: Days of data if dates not specified

        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now()

        if start_date is None:
            start_date = end_date - timedelta(days=days)

        logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")

        if not MT5_AVAILABLE:
            logger.warning("MT5 not available, generating mock data")
            return self._generate_mock_data(symbol, start_date, end_date)

        if not mt5.initialize():
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return self._generate_mock_data(symbol, start_date, end_date)

        # Map timeframe
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

        tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)

        rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
        mt5.shutdown()

        if rates is None or len(rates) == 0:
            logger.error(f"No data retrieved for {symbol}")
            return self._generate_mock_data(symbol, start_date, end_date)

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        logger.info(f"Loaded {len(df)} bars")
        return df

    def _generate_mock_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate mock price data for testing."""
        hours = int((end_date - start_date).total_seconds() / 3600)

        np.random.seed(42)
        base_price = 1.1000 if "EUR" in symbol else 1.3000

        dates = pd.date_range(start=start_date, periods=hours, freq='h')

        # Generate realistic price movement
        returns = np.random.normal(0, 0.0002, hours)
        prices = base_price * (1 + returns).cumprod()

        df = pd.DataFrame({
            'time': dates,
            'open': prices + np.random.normal(0, 0.00005, hours),
            'high': prices + np.abs(np.random.normal(0, 0.0001, hours)),
            'low': prices - np.abs(np.random.normal(0, 0.0001, hours)),
            'close': prices,
            'tick_volume': np.random.randint(100, 1000, hours)
        })

        return df

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        symbol: str,
        **strategy_params
    ) -> BacktestResult:
        """
        Run backtest with a strategy function.

        Args:
            data: Historical price data
            strategy_func: Function that returns signals
            symbol: Trading symbol
            **strategy_params: Additional parameters for strategy

        Returns:
            BacktestResult object
        """
        logger.info(f"Starting backtest for {symbol}")
        logger.info(f"Data points: {len(data)}")

        self.closed_trades = []
        self.current_trade = None

        # Run through each bar
        for i in range(len(data)):
            bar = data.iloc[i]

            # Get signal from strategy
            signal_data = strategy_func(data, i, **strategy_params)

            if signal_data is None:
                continue

            # Check if we have an open trade
            if self.current_trade is not None:
                self._check_exit(bar, signal_data)
            else:
                self._check_entry(bar, signal_data, symbol)

        # Close any remaining trade
        if self.current_trade is not None:
            last_bar = data.iloc[-1]
            self.current_trade.close_trade(
                last_bar['time'],
                last_bar['close'],
                "END"
            )
            self.closed_trades.append(self.current_trade)

        # Calculate results
        result = BacktestResult(
            trades=self.closed_trades,
            start_date=data.iloc[0]['time'],
            end_date=data.iloc[-1]['time'],
            initial_balance=self.initial_balance
        )

        result.calculate_metrics()

        logger.info(f"Backtest complete: {result.total_trades} trades")

        return result

    def _check_entry(self, bar: pd.Series, signal_data: dict, symbol: str):
        """Check for entry signals."""
        signal = signal_data.get('signal')

        if signal not in ['BUY', 'SELL']:
            return

        # Create new trade
        entry_price = bar['close']

        # Apply slippage
        pip_value = 0.0001 if "JPY" not in symbol else 0.01
        slippage = self.slippage_pips * pip_value

        if signal == 'BUY':
            entry_price += slippage
        else:
            entry_price -= slippage

        self.current_trade = Trade(
            entry_time=bar['time'],
            symbol=symbol,
            direction=signal,
            entry_price=entry_price,
            stop_loss=signal_data.get('stop_loss', 0),
            take_profit=signal_data.get('take_profit', 0),
            lot_size=signal_data.get('lot_size', 0.01),
            ml_signal=signal_data.get('ml_signal'),
            sentiment_signal=signal_data.get('sentiment_signal'),
            sentiment_score=signal_data.get('sentiment_score', 0)
        )

    def _check_exit(self, bar: pd.Series, signal_data: dict):
        """Check for exit conditions."""
        if self.current_trade is None:
            return

        exit_price = None
        exit_reason = None

        # Check stop loss
        if self.current_trade.stop_loss > 0:
            if self.current_trade.direction == 'BUY':
                if bar['low'] <= self.current_trade.stop_loss:
                    exit_price = self.current_trade.stop_loss
                    exit_reason = "SL"
            else:  # SELL
                if bar['high'] >= self.current_trade.stop_loss:
                    exit_price = self.current_trade.stop_loss
                    exit_reason = "SL"

        # Check take profit
        if exit_price is None and self.current_trade.take_profit > 0:
            if self.current_trade.direction == 'BUY':
                if bar['high'] >= self.current_trade.take_profit:
                    exit_price = self.current_trade.take_profit
                    exit_reason = "TP"
            else:  # SELL
                if bar['low'] <= self.current_trade.take_profit:
                    exit_price = self.current_trade.take_profit
                    exit_reason = "TP"

        # Check reverse signal
        if exit_price is None:
            signal = signal_data.get('signal')
            if signal and signal != self.current_trade.direction:
                exit_price = bar['close']
                exit_reason = "SIGNAL"

        # Close trade if exit triggered
        if exit_price is not None:
            self.current_trade.close_trade(bar['time'], exit_price, exit_reason)
            self.closed_trades.append(self.current_trade)
            self.current_trade = None
