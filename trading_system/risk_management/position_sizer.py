"""
Position Sizer
==============

Calculate appropriate position sizes based on risk management rules.
"""

import logging
from enum import Enum
from typing import Optional
import math

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LotMode(Enum):
    """Position sizing modes."""
    FIXED = "fixed"              # Fixed lot size
    RISK_BASED = "risk_based"    # Based on risk percentage and SL distance
    BALANCE_PROP = "balance_prop"  # Proportion of account balance


class PositionSizer:
    """
    Calculate position sizes with proper risk management.

    Supports multiple sizing modes:
    - Fixed: Use predetermined lot size
    - Risk-based: Size based on % account risk per trade
    - Balance proportion: Size as % of account balance
    """

    def __init__(
        self,
        mode: LotMode = LotMode.RISK_BASED,
        fixed_lot: float = 0.01,
        risk_percent: float = 1.0,
        balance_percent: float = 2.0,
        max_lot: float = 10.0,
        min_lot: float = 0.01
    ):
        """
        Initialize position sizer.

        Args:
            mode: Position sizing mode
            fixed_lot: Lot size for FIXED mode
            risk_percent: Risk percentage for RISK_BASED mode (1-5)
            balance_percent: Balance percentage for BALANCE_PROP mode
            max_lot: Maximum allowed lot size
            min_lot: Minimum allowed lot size
        """
        self.mode = mode
        self.fixed_lot = fixed_lot
        self.risk_percent = risk_percent
        self.balance_percent = balance_percent
        self.max_lot = max_lot
        self.min_lot = min_lot

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """
        Get symbol trading information from MT5.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with symbol info or None
        """
        if not MT5_AVAILABLE:
            # Return default values for common forex pairs
            return {
                "volume_min": 0.01,
                "volume_max": 100.0,
                "volume_step": 0.01,
                "trade_tick_value": 1.0,
                "point": 0.00001 if "JPY" not in symbol else 0.001,
            }

        if not mt5.initialize():
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return None

        symbol_info = mt5.symbol_info(symbol)
        mt5.shutdown()

        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            return None

        return {
            "volume_min": symbol_info.volume_min,
            "volume_max": symbol_info.volume_max,
            "volume_step": symbol_info.volume_step,
            "trade_tick_value": symbol_info.trade_tick_value,
            "point": symbol_info.point,
        }

    def get_account_balance(self) -> float:
        """
        Get current account balance from MT5.

        Returns:
            Account balance or default value
        """
        if not MT5_AVAILABLE:
            logger.warning("MT5 not available, using default balance")
            return 10000.0

        if not mt5.initialize():
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return 10000.0

        account_info = mt5.account_info()
        mt5.shutdown()

        if account_info is None:
            logger.error("Failed to get account info")
            return 10000.0

        return account_info.balance

    def calculate_lot_fixed(self) -> float:
        """
        Calculate lot size using fixed mode.

        Returns:
            Fixed lot size
        """
        return self.fixed_lot

    def calculate_lot_risk_based(
        self,
        symbol: str,
        sl_distance_pips: float,
        balance: Optional[float] = None
    ) -> float:
        """
        Calculate lot size based on risk percentage.

        Formula:
            Risk Amount = Balance × Risk% / 100
            Lot Size = Risk Amount / (SL Distance in Points × Tick Value)

        Args:
            symbol: Trading symbol
            sl_distance_pips: Stop loss distance in pips
            balance: Account balance (fetched if None)

        Returns:
            Calculated lot size
        """
        if balance is None:
            balance = self.get_account_balance()

        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            logger.error("Could not get symbol info, using fixed lot")
            return self.fixed_lot

        # Calculate risk amount in account currency
        risk_amount = balance * self.risk_percent / 100.0

        # Convert pips to points
        point = symbol_info["point"]
        pip_value = 0.0001 if "JPY" not in symbol else 0.01
        points_per_pip = pip_value / point

        sl_distance_points = sl_distance_pips * points_per_pip

        # Calculate lot size
        tick_value = symbol_info["trade_tick_value"]

        if tick_value <= 0 or sl_distance_points <= 0:
            logger.error("Invalid tick_value or SL distance")
            return self.fixed_lot

        lot = risk_amount / (sl_distance_points * tick_value)

        # Normalize to symbol's lot step
        lot = self._normalize_lot(lot, symbol_info)

        logger.debug(f"Risk-based lot: Balance={balance:.2f}, Risk={self.risk_percent}%, "
                    f"SL={sl_distance_pips} pips, Lot={lot:.2f}")

        return lot

    def calculate_lot_balance_prop(
        self,
        symbol: str,
        balance: Optional[float] = None
    ) -> float:
        """
        Calculate lot size as proportion of account balance.

        Args:
            symbol: Trading symbol
            balance: Account balance (fetched if None)

        Returns:
            Calculated lot size
        """
        if balance is None:
            balance = self.get_account_balance()

        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            logger.error("Could not get symbol info, using fixed lot")
            return self.fixed_lot

        # Calculate lot based on balance proportion
        # Assuming $10,000 per 0.1 lot as baseline
        baseline_balance = 10000.0
        baseline_lot = 0.1

        lot = (balance / baseline_balance) * baseline_lot * (self.balance_percent / 2.0)

        # Normalize to symbol's lot step
        lot = self._normalize_lot(lot, symbol_info)

        logger.debug(f"Balance-prop lot: Balance={balance:.2f}, "
                    f"Percent={self.balance_percent}%, Lot={lot:.2f}")

        return lot

    def _normalize_lot(self, lot: float, symbol_info: dict) -> float:
        """
        Normalize lot size to symbol's constraints.

        Args:
            lot: Raw lot size
            symbol_info: Symbol information dictionary

        Returns:
            Normalized lot size
        """
        min_lot = symbol_info["volume_min"]
        max_lot = symbol_info["volume_max"]
        lot_step = symbol_info["volume_step"]

        # Round down to nearest step
        lot = math.floor(lot / lot_step) * lot_step

        # Apply min/max constraints
        lot = max(min_lot, min(max_lot, lot))
        lot = max(self.min_lot, min(self.max_lot, lot))

        # Calculate decimal places based on lot_step
        if lot_step >= 1:
            decimals = 0
        elif lot_step >= 0.1:
            decimals = 1
        else:
            decimals = 2

        return round(lot, decimals)

    def calculate_lot(
        self,
        symbol: str,
        sl_distance_pips: Optional[float] = None,
        balance: Optional[float] = None
    ) -> float:
        """
        Calculate lot size based on current mode.

        Args:
            symbol: Trading symbol
            sl_distance_pips: Stop loss distance in pips (required for RISK_BASED)
            balance: Account balance (fetched if None)

        Returns:
            Calculated lot size
        """
        if self.mode == LotMode.FIXED:
            return self.calculate_lot_fixed()

        elif self.mode == LotMode.RISK_BASED:
            if sl_distance_pips is None:
                logger.error("SL distance required for risk-based sizing")
                return self.fixed_lot
            return self.calculate_lot_risk_based(symbol, sl_distance_pips, balance)

        elif self.mode == LotMode.BALANCE_PROP:
            return self.calculate_lot_balance_prop(symbol, balance)

        else:
            logger.error(f"Unknown lot mode: {self.mode}")
            return self.fixed_lot

    def get_mode_string(self) -> str:
        """Get human-readable mode name."""
        mode_names = {
            LotMode.FIXED: "Fixed Lot",
            LotMode.RISK_BASED: "Risk-Based",
            LotMode.BALANCE_PROP: "Balance Proportion",
        }
        return mode_names.get(self.mode, "Unknown")
