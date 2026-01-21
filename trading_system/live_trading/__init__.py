"""
Live Trading Module
===================

Generate live trading signals without needing MetaTrader.
"""

from .signal_generator import LiveSignalGenerator, TradingSignal

__all__ = ["LiveSignalGenerator", "TradingSignal"]
