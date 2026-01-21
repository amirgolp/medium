"""
Risk Manager
============

Overall risk management and portfolio control.
"""

import logging
from typing import Optional, Dict
from datetime import datetime, timedelta

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskManager:
    """
    Manage overall portfolio risk and enforce trading rules.

    Controls:
    - Maximum positions
    - Daily loss limits
    - Trading hours
    - Position holding times
    - Correlation limits
    """

    def __init__(
        self,
        max_positions: int = 1,
        max_daily_loss_pct: float = 5.0,
        max_daily_loss_amount: Optional[float] = None,
        holding_period_hours: Optional[int] = None,
        magic_number: Optional[int] = None
    ):
        """
        Initialize risk manager.

        Args:
            max_positions: Maximum simultaneous positions
            max_daily_loss_pct: Maximum daily loss as % of balance
            max_daily_loss_amount: Maximum daily loss in currency
            holding_period_hours: Maximum hours to hold a position
            magic_number: EA magic number for position filtering
        """
        self.max_positions = max_positions
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_daily_loss_amount = max_daily_loss_amount
        self.holding_period_hours = holding_period_hours
        self.magic_number = magic_number

        self.daily_pnl = 0.0
        self.daily_reset_time = datetime.now().date()

    def can_open_position(self, symbol: Optional[str] = None) -> tuple:
        """
        Check if we can open a new position.

        Args:
            symbol: Trading symbol (optional)

        Returns:
            Tuple of (can_trade, reason)
        """
        # Check position count
        position_count = self.get_position_count()
        if position_count >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"

        # Check daily loss limit
        if not self.check_daily_loss_limit():
            return False, "Daily loss limit reached"

        # Check symbol-specific position
        if symbol and self.has_position_for_symbol(symbol):
            return False, f"Already have position for {symbol}"

        return True, "OK"

    def get_position_count(self) -> int:
        """
        Get current number of open positions.

        Returns:
            Number of positions
        """
        if not MT5_AVAILABLE:
            return 0

        if not mt5.initialize():
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return 0

        positions = mt5.positions_get()
        mt5.shutdown()

        if positions is None:
            return 0

        # Filter by magic number if set
        if self.magic_number is not None:
            positions = [p for p in positions if p.magic == self.magic_number]

        return len(positions)

    def has_position_for_symbol(self, symbol: str) -> bool:
        """
        Check if there's an open position for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            True if position exists
        """
        if not MT5_AVAILABLE:
            return False

        if not mt5.initialize():
            return False

        positions = mt5.positions_get(symbol=symbol)
        mt5.shutdown()

        if positions is None:
            return False

        # Filter by magic number if set
        if self.magic_number is not None:
            positions = [p for p in positions if p.magic == self.magic_number]

        return len(positions) > 0

    def check_daily_loss_limit(self) -> bool:
        """
        Check if daily loss limit has been reached.

        Returns:
            True if trading is allowed, False if limit reached
        """
        # Reset daily PnL if new day
        today = datetime.now().date()
        if today > self.daily_reset_time:
            self.daily_pnl = 0.0
            self.daily_reset_time = today
            logger.info("Daily PnL reset")

        # Update current daily PnL
        self._update_daily_pnl()

        # Check percentage limit
        if self.max_daily_loss_pct is not None:
            balance = self._get_balance()
            max_loss = balance * (self.max_daily_loss_pct / 100.0)

            if abs(self.daily_pnl) >= max_loss and self.daily_pnl < 0:
                logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2f} / {max_loss:.2f}")
                return False

        # Check absolute limit
        if self.max_daily_loss_amount is not None:
            if abs(self.daily_pnl) >= self.max_daily_loss_amount and self.daily_pnl < 0:
                logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2f} / {self.max_daily_loss_amount:.2f}")
                return False

        return True

    def _update_daily_pnl(self) -> None:
        """Update daily profit/loss from closed deals."""
        if not MT5_AVAILABLE:
            return

        if not mt5.initialize():
            return

        # Get today's deals
        today_start = datetime.combine(datetime.now().date(), datetime.min.time())

        deals = mt5.history_deals_get(today_start, datetime.now())
        mt5.shutdown()

        if deals is None:
            return

        # Filter by magic number and calculate PnL
        daily_profit = 0.0
        for deal in deals:
            if self.magic_number is None or deal.magic == self.magic_number:
                daily_profit += deal.profit

        self.daily_pnl = daily_profit

    def _get_balance(self) -> float:
        """Get current account balance."""
        if not MT5_AVAILABLE:
            return 10000.0

        if not mt5.initialize():
            return 10000.0

        account_info = mt5.account_info()
        mt5.shutdown()

        if account_info is None:
            return 10000.0

        return account_info.balance

    def check_position_holding_time(self, symbol: str) -> tuple:
        """
        Check if position has exceeded holding time.

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (should_close, hours_held)
        """
        if self.holding_period_hours is None:
            return False, 0

        if not MT5_AVAILABLE:
            return False, 0

        if not mt5.initialize():
            return False, 0

        positions = mt5.positions_get(symbol=symbol)
        mt5.shutdown()

        if positions is None or len(positions) == 0:
            return False, 0

        # Get first position (assumes one position per symbol)
        position = positions[0]

        # Filter by magic number if set
        if self.magic_number is not None and position.magic != self.magic_number:
            return False, 0

        # Calculate holding time
        open_time = datetime.fromtimestamp(position.time)
        hours_held = (datetime.now() - open_time).total_seconds() / 3600.0

        should_close = hours_held >= self.holding_period_hours

        if should_close:
            logger.info(f"Position holding time exceeded: {hours_held:.1f}h / {self.holding_period_hours}h")

        return should_close, hours_held

    def is_within_trade_hours(
        self,
        start_hour: int,
        end_hour: int
    ) -> bool:
        """
        Check if current time is within trading hours.

        Args:
            start_hour: Start hour (0-23)
            end_hour: End hour (0-23)

        Returns:
            True if within trading hours
        """
        current_hour = datetime.now().hour

        if start_hour <= end_hour:
            # Normal range (e.g., 8-18)
            return start_hour <= current_hour <= end_hour
        else:
            # Overnight range (e.g., 22-6)
            return current_hour >= start_hour or current_hour <= end_hour

    def get_portfolio_status(self) -> Dict[str, any]:
        """
        Get current portfolio status.

        Returns:
            Dictionary with portfolio information
        """
        position_count = self.get_position_count()
        balance = self._get_balance()
        self._update_daily_pnl()

        return {
            "position_count": position_count,
            "max_positions": self.max_positions,
            "balance": balance,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": (self.daily_pnl / balance * 100.0) if balance > 0 else 0,
            "can_trade": position_count < self.max_positions and self.check_daily_loss_limit(),
        }

    def log_portfolio_status(self) -> None:
        """Log current portfolio status."""
        status = self.get_portfolio_status()

        logger.info("=" * 60)
        logger.info("PORTFOLIO STATUS")
        logger.info("=" * 60)
        logger.info(f"Positions: {status['position_count']}/{status['max_positions']}")
        logger.info(f"Balance: ${status['balance']:.2f}")
        logger.info(f"Daily PnL: ${status['daily_pnl']:.2f} ({status['daily_pnl_pct']:.2f}%)")
        logger.info(f"Can Trade: {status['can_trade']}")
        logger.info("=" * 60)
