"""
Trading System Package
======================

A three-pillar automated trading system combining:
1. ML-powered forex predictions (CNN-LSTM)
2. Sentiment-based news analysis
3. Risk management and position sizing
4. Backtesting and visualization

Author: Trading System
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Trading System"

from .ml_predictor.trainer import ModelTrainer
from .ml_predictor.predictor import ForexPredictor
from .sentiment_analyzer.analyzer import SentimentAnalyzer
from .risk_management.position_sizer import PositionSizer
from .risk_management.risk_manager import RiskManager
from .risk_management.atr_calculator import ATRCalculator
from .backtest.engine import BacktestEngine, Trade, BacktestResult
from .backtest.visualizer import BacktestVisualizer

__all__ = [
    "ModelTrainer",
    "ForexPredictor",
    "SentimentAnalyzer",
    "PositionSizer",
    "RiskManager",
    "ATRCalculator",
    "BacktestEngine",
    "Trade",
    "BacktestResult",
    "BacktestVisualizer",
]
