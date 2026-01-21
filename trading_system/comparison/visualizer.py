"""
Comparison Visualizer
=====================

Create visual comparisons of multiple strategies.
"""

import os
from typing import List
import pandas as pd
import numpy as np

from .comparator import ComparisonResult
from ..backtest import BacktestResult


class ComparisonVisualizer:
    """Create visual comparisons of strategies."""

    def __init__(self):
        """Initialize visualizer."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            self.plt = plt
            self.gridspec = gridspec
            self.available = True
        except ImportError:
            self.available = False

    def create_full_report(self, comparison: ComparisonResult,
                          data: pd.DataFrame, symbol: str,
                          output_dir: str = "./strategy_comparison"):
        """
        Create complete comparison report with charts.

        Args:
            comparison: ComparisonResult object
            data: OHLC DataFrame
            symbol: Currency pair
            output_dir: Output directory
        """
        if not self.available:
            raise ImportError("Matplotlib not installed")

        os.makedirs(output_dir, exist_ok=True)

        # 1. Equity curves comparison
        self.plot_equity_curves(comparison, output_dir)

        # 2. Metrics comparison bar charts
        self.plot_metrics_comparison(comparison, output_dir)

        # 3. Trade distribution
        self.plot_trade_distributions(comparison, output_dir)

        # 4. Drawdown comparison
        self.plot_drawdown_comparison(comparison, output_dir)

        # 5. Win/Loss analysis
        self.plot_win_loss_analysis(comparison, output_dir)

        # 6. Summary dashboard
        self.plot_summary_dashboard(comparison, symbol, output_dir)

    def plot_equity_curves(self, comparison: ComparisonResult, output_dir: str):
        """Plot equity curves for all strategies."""
        fig, ax = self.plt.subplots(figsize=(14, 7))

        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

        for i, (name, result) in enumerate(zip(comparison.strategy_names, comparison.results)):
            if result.equity_curve:
                color = colors[i % len(colors)]
                ax.plot(result.equity_curve, label=name, linewidth=2, color=color, alpha=0.8)

        ax.set_xlabel('Trade Number', fontsize=12)
        ax.set_ylabel('Balance ($)', fontsize=12)
        ax.set_title('Strategy Equity Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add horizontal line at initial balance
        if comparison.results:
            initial_balance = comparison.results[0].initial_balance
            ax.axhline(y=initial_balance, color='gray', linestyle='--',
                      linewidth=1, alpha=0.5, label='Initial Balance')

        self.plt.tight_layout()
        self.plt.savefig(f"{output_dir}/equity_curves.png", dpi=300, bbox_inches='tight')
        self.plt.close()

    def plot_metrics_comparison(self, comparison: ComparisonResult, output_dir: str):
        """Plot comparison of key metrics."""
        fig, axes = self.plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Strategy Metrics Comparison', fontsize=16, fontweight='bold')

        df = comparison.comparison_metrics.reset_index()

        # 1. Net Profit
        ax = axes[0, 0]
        colors = ['green' if x > 0 else 'red' for x in df['net_profit']]
        ax.bar(df['strategy'], df['net_profit'], color=colors, alpha=0.7)
        ax.set_title('Net Profit ($)', fontweight='bold')
        ax.set_ylabel('Profit ($)')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')

        # 2. Win Rate
        ax = axes[0, 1]
        colors = ['green' if x >= 50 else 'orange' if x >= 40 else 'red' for x in df['win_rate']]
        ax.bar(df['strategy'], df['win_rate'], color=colors, alpha=0.7)
        ax.set_title('Win Rate (%)', fontweight='bold')
        ax.set_ylabel('Win Rate (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=50, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Profit Factor
        ax = axes[0, 2]
        colors = ['green' if x > 1.5 else 'orange' if x > 1.0 else 'red' for x in df['profit_factor']]
        ax.bar(df['strategy'], df['profit_factor'], color=colors, alpha=0.7)
        ax.set_title('Profit Factor', fontweight='bold')
        ax.set_ylabel('Profit Factor')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Sharpe Ratio
        ax = axes[1, 0]
        colors = ['green' if x > 1.0 else 'orange' if x > 0 else 'red' for x in df['sharpe_ratio']]
        ax.bar(df['strategy'], df['sharpe_ratio'], color=colors, alpha=0.7)
        ax.set_title('Sharpe Ratio', fontweight='bold')
        ax.set_ylabel('Sharpe Ratio')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')

        # 5. Max Drawdown
        ax = axes[1, 1]
        ax.bar(df['strategy'], df['max_drawdown_pct'], color='red', alpha=0.7)
        ax.set_title('Max Drawdown (%)', fontweight='bold')
        ax.set_ylabel('Drawdown (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # 6. Total Trades
        ax = axes[1, 2]
        ax.bar(df['strategy'], df['total_trades'], color='#3498db', alpha=0.7)
        ax.set_title('Total Trades', fontweight='bold')
        ax.set_ylabel('Number of Trades')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        self.plt.tight_layout()
        self.plt.savefig(f"{output_dir}/metrics_comparison.png", dpi=300, bbox_inches='tight')
        self.plt.close()

    def plot_trade_distributions(self, comparison: ComparisonResult, output_dir: str):
        """Plot trade distribution for each strategy."""
        n_strategies = len(comparison.results)
        fig, axes = self.plt.subplots(1, n_strategies, figsize=(6*n_strategies, 5))

        if n_strategies == 1:
            axes = [axes]

        fig.suptitle('Trade P&L Distribution by Strategy', fontsize=14, fontweight='bold')

        for i, (name, result) in enumerate(zip(comparison.strategy_names, comparison.results)):
            ax = axes[i]

            if result.trades:
                pnls = [t.profit_loss for t in result.trades]

                # Histogram
                ax.hist(pnls, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
                ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
                ax.axvline(x=np.mean(pnls), color='green', linestyle='--',
                          linewidth=2, label=f'Mean: ${np.mean(pnls):.2f}')

                ax.set_title(name, fontweight='bold')
                ax.set_xlabel('P&L ($)')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, 'No trades', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(name, fontweight='bold')

        self.plt.tight_layout()
        self.plt.savefig(f"{output_dir}/trade_distributions.png", dpi=300, bbox_inches='tight')
        self.plt.close()

    def plot_drawdown_comparison(self, comparison: ComparisonResult, output_dir: str):
        """Plot drawdown curves for all strategies."""
        fig, ax = self.plt.subplots(figsize=(14, 7))

        colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6']

        for i, (name, result) in enumerate(zip(comparison.strategy_names, comparison.results)):
            if result.equity_curve and len(result.equity_curve) > 0:
                color = colors[i % len(colors)]

                # Calculate drawdown from equity curve
                equity = np.array(result.equity_curve)
                peak = np.maximum.accumulate(equity)
                drawdown = (equity - peak) / peak * 100  # Percentage drawdown

                ax.plot(drawdown, label=name, linewidth=2, color=color, alpha=0.8)

        ax.set_xlabel('Trade Number', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title('Strategy Drawdown Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Invert y-axis (drawdown should go down)
        ax.invert_yaxis()

        self.plt.tight_layout()
        self.plt.savefig(f"{output_dir}/drawdown_comparison.png", dpi=300, bbox_inches='tight')
        self.plt.close()

    def plot_win_loss_analysis(self, comparison: ComparisonResult, output_dir: str):
        """Plot win/loss analysis."""
        fig, axes = self.plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Win/Loss Analysis', fontsize=14, fontweight='bold')

        df = comparison.comparison_metrics.reset_index()

        # 1. Win vs Loss comparison
        ax = axes[0]
        x = np.arange(len(df['strategy']))
        width = 0.35

        ax.bar(x - width/2, df['avg_win'], width, label='Avg Win',
              color='green', alpha=0.7)
        ax.bar(x + width/2, df['avg_loss'], width, label='Avg Loss',
              color='red', alpha=0.7)

        ax.set_xlabel('Strategy')
        ax.set_ylabel('Average P&L ($)')
        ax.set_title('Average Win vs Loss', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['strategy'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 2. Largest Win vs Loss
        ax = axes[1]
        ax.bar(x - width/2, df['largest_win'], width, label='Largest Win',
              color='darkgreen', alpha=0.7)
        ax.bar(x + width/2, df['largest_loss'], width, label='Largest Loss',
              color='darkred', alpha=0.7)

        ax.set_xlabel('Strategy')
        ax.set_ylabel('P&L ($)')
        ax.set_title('Largest Win vs Loss', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['strategy'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        self.plt.tight_layout()
        self.plt.savefig(f"{output_dir}/win_loss_analysis.png", dpi=300, bbox_inches='tight')
        self.plt.close()

    def plot_summary_dashboard(self, comparison: ComparisonResult,
                              symbol: str, output_dir: str):
        """Create summary dashboard with key information."""
        fig = self.plt.figure(figsize=(16, 10))
        gs = self.gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle(f'Strategy Comparison Dashboard - {symbol}',
                    fontsize=16, fontweight='bold')

        # Get best strategy
        best_strategy = comparison.get_winner('sharpe_ratio')
        df = comparison.comparison_metrics

        # 1. Summary table (top row, spans all columns)
        ax_table = fig.add_subplot(gs[0, :])
        ax_table.axis('tight')
        ax_table.axis('off')

        # Format table data
        table_data = []
        for idx, row in df.iterrows():
            table_data.append([
                idx,
                f"{row['total_trades']:.0f}",
                f"{row['win_rate']:.1f}%",
                f"{row['profit_factor']:.2f}",
                f"{row['sharpe_ratio']:.2f}",
                f"${row['net_profit']:,.2f}",
                f"{row['max_drawdown_pct']:.2f}%"
            ])

        headers = ['Strategy', 'Trades', 'Win Rate', 'PF', 'Sharpe', 'Net Profit', 'Max DD']

        table = ax_table.table(cellText=table_data, colLabels=headers,
                              loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Color best values
        for i, header in enumerate(headers[1:], 1):
            if header in ['Trades', 'Win Rate', 'PF', 'Sharpe', 'Net Profit']:
                best_idx = df.iloc[:, i-1].idxmax()
                row_idx = list(df.index).index(best_idx) + 1
                table[(row_idx, i)].set_facecolor('#90EE90')

        # 2. Equity curves (middle row, left)
        ax_equity = fig.add_subplot(gs[1, :2])
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

        for i, (name, result) in enumerate(zip(comparison.strategy_names, comparison.results)):
            if result.equity_curve:
                color = colors[i % len(colors)]
                ax_equity.plot(result.equity_curve, label=name,
                             linewidth=2, color=color, alpha=0.8)

        ax_equity.set_xlabel('Trade Number')
        ax_equity.set_ylabel('Balance ($)')
        ax_equity.set_title('Equity Curves', fontweight='bold')
        ax_equity.legend(loc='best', fontsize=8)
        ax_equity.grid(True, alpha=0.3)

        # 3. Metrics radar (middle row, right)
        ax_metrics = fig.add_subplot(gs[1, 2])
        df_reset = df.reset_index()
        ax_metrics.barh(df_reset['strategy'], df_reset['sharpe_ratio'], color='#3498db', alpha=0.7)
        ax_metrics.set_xlabel('Sharpe Ratio')
        ax_metrics.set_title('Sharpe Ratio Comparison', fontweight='bold')
        ax_metrics.grid(True, alpha=0.3, axis='x')

        # 4. Win rate pie charts (bottom row)
        for i, (name, result) in enumerate(zip(comparison.strategy_names[:3], comparison.results[:3])):
            ax = fig.add_subplot(gs[2, i])

            if result.total_trades > 0:
                wins = result.winning_trades
                losses = result.total_trades - wins

                colors_pie = ['#2ecc71', '#e74c3c']
                ax.pie([wins, losses], labels=['Wins', 'Losses'], autopct='%1.1f%%',
                      colors=colors_pie, startangle=90)
                ax.set_title(f'{name}\n({result.total_trades} trades)', fontweight='bold', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No trades', ha='center', va='center', fontsize=10)
                ax.set_title(name, fontweight='bold', fontsize=9)

        self.plt.savefig(f"{output_dir}/summary_dashboard.png", dpi=300, bbox_inches='tight')
        self.plt.close()
