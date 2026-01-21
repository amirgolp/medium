"""
Backtesting Module
==================

Backtest trading strategies with historical data and visualize results.
"""

from .engine import BacktestEngine, Trade, BacktestResult
from .visualizer import BacktestVisualizer

__all__ = ["BacktestEngine", "Trade", "BacktestResult", "BacktestVisualizer"]
