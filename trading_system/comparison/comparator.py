"""
Strategy Comparator
===================

Compare multiple strategies on the same historical data.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from ..backtest import BacktestEngine, BacktestResult
from ..strategies import BaseStrategy


@dataclass
class ComparisonResult:
    """Results from comparing multiple strategies."""
    strategy_names: List[str]
    results: List[BacktestResult]
    comparison_metrics: pd.DataFrame

    def print_summary(self):
        """Print comparison summary."""
        print("\n" + "="*80)
        print("STRATEGY COMPARISON SUMMARY")
        print("="*80)

        # Print comparison table
        print("\nPerformance Metrics:")
        print(self.comparison_metrics.to_string())

        # Find best strategies
        print("\n" + "-"*80)
        print("BEST STRATEGIES:")
        print("-"*80)

        if 'sharpe_ratio' in self.comparison_metrics.columns:
            best_sharpe = self.comparison_metrics['sharpe_ratio'].idxmax()
            print(f"  Best Sharpe Ratio:   {best_sharpe} ({self.comparison_metrics.loc[best_sharpe, 'sharpe_ratio']:.2f})")

        if 'profit_factor' in self.comparison_metrics.columns:
            best_pf = self.comparison_metrics['profit_factor'].idxmax()
            print(f"  Best Profit Factor:  {best_pf} ({self.comparison_metrics.loc[best_pf, 'profit_factor']:.2f})")

        if 'win_rate' in self.comparison_metrics.columns:
            best_wr = self.comparison_metrics['win_rate'].idxmax()
            print(f"  Best Win Rate:       {best_wr} ({self.comparison_metrics.loc[best_wr, 'win_rate']:.1f}%)")

        if 'net_profit' in self.comparison_metrics.columns:
            best_profit = self.comparison_metrics['net_profit'].idxmax()
            print(f"  Best Net Profit:     {best_profit} (${self.comparison_metrics.loc[best_profit, 'net_profit']:,.2f})")

        print("\n" + "="*80)

    def get_winner(self, metric: str = 'sharpe_ratio') -> str:
        """
        Get best strategy by metric.

        Args:
            metric: Metric to compare ('sharpe_ratio', 'profit_factor', 'net_profit', etc.)

        Returns:
            Name of best strategy
        """
        if metric not in self.comparison_metrics.columns:
            raise ValueError(f"Metric '{metric}' not found in comparison results")

        return self.comparison_metrics[metric].idxmax()


class StrategyComparator:
    """
    Compare multiple strategies on the same historical data.

    Usage:
        comparator = StrategyComparator(initial_balance=10000)
        comparator.add_strategy(ml_strategy, "ML Strategy")
        comparator.add_strategy(pattern_strategy, "Pattern Strategy")
        result = comparator.run(data, symbol="EURUSD")
        result.print_summary()
    """

    def __init__(self, initial_balance: float = 10000.0, slippage_pips: float = 1.0):
        """
        Initialize comparator.

        Args:
            initial_balance: Starting balance for each strategy
            slippage_pips: Slippage in pips
        """
        self.initial_balance = initial_balance
        self.slippage_pips = slippage_pips
        self.strategies: List[tuple[BaseStrategy, str]] = []

    def add_strategy(self, strategy: BaseStrategy, name: Optional[str] = None):
        """
        Add strategy to comparison.

        Args:
            strategy: Strategy instance
            name: Optional custom name (uses strategy.get_name() if not provided)
        """
        strategy_name = name if name else strategy.get_name()
        self.strategies.append((strategy, strategy_name))

    def run(self, data: pd.DataFrame, symbol: str = "EURUSD",
            verbose: bool = True) -> ComparisonResult:
        """
        Run all strategies on the same data.

        Args:
            data: OHLC DataFrame
            symbol: Currency pair
            verbose: Print progress

        Returns:
            ComparisonResult with all results
        """
        if not self.strategies:
            raise ValueError("No strategies added. Use add_strategy() first.")

        results: List[BacktestResult] = []
        strategy_names: List[str] = []

        if verbose:
            print("\n" + "="*80)
            print("RUNNING STRATEGY COMPARISON")
            print("="*80)
            print(f"Symbol: {symbol}")
            print(f"Data points: {len(data)}")
            print(f"Period: {data['time'].iloc[0]} to {data['time'].iloc[-1]}")
            print(f"Initial balance: ${self.initial_balance:,.2f}")
            print(f"Strategies: {len(self.strategies)}")
            print("="*80 + "\n")

        for i, (strategy, name) in enumerate(self.strategies, 1):
            if verbose:
                print(f"[{i}/{len(self.strategies)}] Running: {name}")
                print(f"  Description: {strategy.get_description()}")

            # Create new engine for each strategy
            engine = BacktestEngine(
                initial_balance=self.initial_balance,
                slippage_pips=self.slippage_pips
            )

            # Run backtest
            result = engine.run_backtest(
                data=data,
                strategy_func=strategy,
                symbol=symbol
            )

            results.append(result)
            strategy_names.append(name)

            if verbose:
                print(f"  ✓ Completed: {result.total_trades} trades, "
                      f"Net: ${result.net_profit:,.2f}, "
                      f"Win Rate: {result.win_rate:.1f}%\n")

        # Create comparison metrics DataFrame
        comparison_df = self._create_comparison_table(strategy_names, results)

        if verbose:
            print("="*80)
            print("COMPARISON COMPLETE")
            print("="*80 + "\n")

        return ComparisonResult(
            strategy_names=strategy_names,
            results=results,
            comparison_metrics=comparison_df
        )

    def _create_comparison_table(self, names: List[str],
                                 results: List[BacktestResult]) -> pd.DataFrame:
        """Create comparison metrics DataFrame."""
        metrics = []

        for name, result in zip(names, results):
            # Calculate net profit percentage
            net_profit_pct = (result.net_profit / result.initial_balance) * 100 if result.initial_balance > 0 else 0.0

            metrics.append({
                'strategy': name,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'sharpe_ratio': result.sharpe_ratio,
                'net_profit': result.net_profit,
                'net_profit_pct': net_profit_pct,
                'max_drawdown': result.max_drawdown,
                'max_drawdown_pct': result.max_drawdown_pct,
                'avg_win': result.avg_profit,
                'avg_loss': result.avg_loss,
                'largest_win': result.largest_profit,
                'largest_loss': result.largest_loss,
                'final_balance': result.final_balance
            })

        df = pd.DataFrame(metrics)
        df = df.set_index('strategy')

        return df

    def run_and_visualize(self, data: pd.DataFrame, symbol: str = "EURUSD",
                         output_dir: str = "./strategy_comparison",
                         verbose: bool = True) -> ComparisonResult:
        """
        Run comparison and generate visualizations.

        Args:
            data: OHLC DataFrame
            symbol: Currency pair
            output_dir: Directory for output files
            verbose: Print progress

        Returns:
            ComparisonResult
        """
        # Run comparison
        result = self.run(data, symbol, verbose)

        # Generate visualizations
        if verbose:
            print("\nGenerating visualizations...")

        try:
            from .visualizer import ComparisonVisualizer

            viz = ComparisonVisualizer()
            viz.create_full_report(result, data, symbol, output_dir)

            if verbose:
                print(f"✓ Visualizations saved to {output_dir}/\n")

        except ImportError:
            if verbose:
                print("⚠️  Matplotlib not installed, skipping visualizations\n")

        return result
