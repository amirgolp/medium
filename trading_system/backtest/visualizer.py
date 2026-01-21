"""
Backtest Visualizer
===================

Visualize backtest results with charts and graphs.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available. Install with: pip install matplotlib")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
    if MATPLOTLIB_AVAILABLE:
        sns.set_style("darkgrid")
except ImportError:
    SEABORN_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestVisualizer:
    """
    Create visualizations for backtest results.

    Charts:
    - Equity curve
    - Drawdown chart
    - Trade distribution
    - Win/loss analysis
    - Monthly returns
    - Trade duration analysis
    """

    def __init__(self, figsize=(15, 10)):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size for plots
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for visualization. Install with: pip install matplotlib")

        self.figsize = figsize

    def plot_equity_curve(self, result, save_path: Optional[str] = None):
        """
        Plot equity curve over time.

        Args:
            result: BacktestResult object
            save_path: Path to save figure (optional)
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot equity curve
        ax.plot(result.equity_dates, result.equity_curve, linewidth=2, color='#2E86AB', label='Equity')

        # Add initial balance line
        ax.axhline(y=result.initial_balance, color='gray', linestyle='--', alpha=0.5, label='Initial Balance')

        # Format
        ax.set_title('Equity Curve', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Account Balance ($)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        # Add profit/loss text
        final_pnl = result.final_balance - result.initial_balance
        pnl_pct = (final_pnl / result.initial_balance) * 100
        color = 'green' if final_pnl >= 0 else 'red'

        textstr = f'P&L: ${final_pnl:,.2f} ({pnl_pct:+.2f}%)'
        props = dict(boxstyle='round', facecolor=color, alpha=0.2)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props, color=color, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Equity curve saved to {save_path}")

        plt.show()

    def plot_drawdown(self, result, save_path: Optional[str] = None):
        """
        Plot drawdown over time.

        Args:
            result: BacktestResult object
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Calculate drawdown series
        equity = np.array(result.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = ((equity - peak) / peak) * 100

        dates = result.equity_dates

        # Plot drawdown
        ax.fill_between(dates, drawdown, 0, alpha=0.3, color='red', label='Drawdown')
        ax.plot(dates, drawdown, linewidth=1.5, color='darkred')

        # Format
        ax.set_title('Drawdown', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        # Add max drawdown text
        textstr = f'Max Drawdown: {result.max_drawdown_pct:.2f}%'
        props = dict(boxstyle='round', facecolor='red', alpha=0.2)
        ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', bbox=props, color='darkred', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Drawdown chart saved to {save_path}")

        plt.show()

    def plot_trade_distribution(self, result, save_path: Optional[str] = None):
        """
        Plot distribution of trade profits/losses.

        Args:
            result: BacktestResult object
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        profits = [t.profit_loss for t in result.trades]

        # Histogram
        ax1.hist(profits, bins=30, alpha=0.7, color='#2E86AB', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax1.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Profit/Loss ($)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(profits, vert=True)
        ax2.set_title('Trade P&L Box Plot', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Profit/Loss ($)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trade distribution saved to {save_path}")

        plt.show()

    def plot_win_loss_analysis(self, result, save_path: Optional[str] = None):
        """
        Plot win/loss analysis.

        Args:
            result: BacktestResult object
            save_path: Path to save figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)

        # 1. Win/Loss pie chart
        sizes = [result.winning_trades, result.losing_trades]
        labels = [f'Winners\n({result.winning_trades})', f'Losers\n({result.losing_trades})']
        colors = ['#06D6A0', '#EF476F']
        explode = (0.05, 0.05)

        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax1.set_title('Win/Loss Ratio', fontsize=14, fontweight='bold')

        # 2. Cumulative P&L by trade
        cumulative_pnl = np.cumsum([t.profit_loss for t in result.trades])
        trade_numbers = range(1, len(result.trades) + 1)

        ax2.plot(trade_numbers, cumulative_pnl, linewidth=2, color='#2E86AB')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('Cumulative P&L by Trade', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Trade Number', fontsize=12)
        ax2.set_ylabel('Cumulative P&L ($)', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # 3. Exit reasons
        exit_reasons = {}
        for trade in result.trades:
            reason = trade.exit_reason
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        if exit_reasons:
            reasons = list(exit_reasons.keys())
            counts = list(exit_reasons.values())

            colors_map = {'TP': '#06D6A0', 'SL': '#EF476F', 'SIGNAL': '#FFD166', 'TIME': '#118AB2', 'END': '#073B4C'}
            bar_colors = [colors_map.get(r, '#999999') for r in reasons]

            ax3.bar(reasons, counts, color=bar_colors, alpha=0.7, edgecolor='black')
            ax3.set_title('Exit Reasons', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Reason', fontsize=12)
            ax3.set_ylabel('Count', fontsize=12)
            ax3.grid(True, alpha=0.3, axis='y')

        # 4. Monthly returns
        if result.trades:
            df_trades = pd.DataFrame([{
                'date': t.exit_time,
                'profit': t.profit_loss
            } for t in result.trades if t.exit_time])

            df_trades['date'] = pd.to_datetime(df_trades['date'])
            df_trades['month'] = df_trades['date'].dt.to_period('M')

            monthly = df_trades.groupby('month')['profit'].sum()

            if len(monthly) > 0:
                colors = ['green' if x >= 0 else 'red' for x in monthly.values]
                monthly.plot(kind='bar', ax=ax4, color=colors, alpha=0.7, edgecolor='black')
                ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax4.set_title('Monthly Returns', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Month', fontsize=12)
                ax4.set_ylabel('Profit/Loss ($)', fontsize=12)
                ax4.grid(True, alpha=0.3, axis='y')
                plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Win/loss analysis saved to {save_path}")

        plt.show()

    def plot_trade_timeline(self, result, data, save_path: Optional[str] = None):
        """
        Plot price chart with trade entry/exit markers.

        Args:
            result: BacktestResult object
            data: Price data DataFrame
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot price
        ax.plot(data['time'], data['close'], linewidth=1, color='#073B4C', alpha=0.7, label='Price')

        # Plot trades
        for trade in result.trades:
            # Entry marker
            if trade.direction == 'BUY':
                marker = '^'
                color = 'green'
            else:
                marker = 'v'
                color = 'red'

            ax.scatter(trade.entry_time, trade.entry_price, marker=marker, s=100,
                      color=color, alpha=0.8, edgecolors='black', linewidths=1.5, zorder=5)

            # Exit marker
            if trade.exit_time and trade.exit_price:
                exit_color = 'green' if trade.profit_loss > 0 else 'red'
                ax.scatter(trade.exit_time, trade.exit_price, marker='x', s=100,
                          color=exit_color, alpha=0.8, linewidths=2, zorder=5)

                # Draw line connecting entry to exit
                ax.plot([trade.entry_time, trade.exit_time],
                       [trade.entry_price, trade.exit_price],
                       color=exit_color, alpha=0.3, linewidth=1, linestyle='--')

        # Format
        ax.set_title('Trade Timeline', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.legend(['Price', 'Buy Entry', 'Sell Entry', 'Exit'], loc='best')
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trade timeline saved to {save_path}")

        plt.show()

    def create_full_report(self, result, data, output_dir: str = "./backtest_results"):
        """
        Create complete visual report with all charts.

        Args:
            result: BacktestResult object
            data: Price data DataFrame
            output_dir: Directory to save charts
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Generating backtest report...")

        # Generate all charts
        self.plot_equity_curve(result, f"{output_dir}/equity_curve.png")
        self.plot_drawdown(result, f"{output_dir}/drawdown.png")
        self.plot_trade_distribution(result, f"{output_dir}/trade_distribution.png")
        self.plot_win_loss_analysis(result, f"{output_dir}/win_loss_analysis.png")
        self.plot_trade_timeline(result, data, f"{output_dir}/trade_timeline.png")

        logger.info(f"Report saved to {output_dir}/")

    def plot_all(self, result, data):
        """
        Display all charts in one view.

        Args:
            result: BacktestResult object
            data: Price data DataFrame
        """
        logger.info("Displaying all charts...")

        self.plot_equity_curve(result)
        self.plot_drawdown(result)
        self.plot_trade_distribution(result)
        self.plot_win_loss_analysis(result)
        self.plot_trade_timeline(result, data)
