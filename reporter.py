"""Print a nice performance report and save the equity chart."""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from colorama import Fore, Style, init

from backtester import BacktestResult

# Ensure the terminal can handle the characters we print on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-16"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

init(autoreset=True)


def print_report(result: BacktestResult) -> None:
    m = result.metrics
    sep = "-" * 52

    print(f"\n{Fore.CYAN}{sep}")
    print(f"  BACKTEST PERFORMANCE REPORT")
    print(f"{sep}{Style.RESET_ALL}")

    rows = [
        ("Initial Capital",      f"$100,000.00"),
        ("Final Equity",         f"${m['final_equity_usd']:,.2f}"),
        ("Total Return",         f"{m['total_return_pct']:+.2f} %"),
        ("Annual Return",        f"{m['annual_return_pct']:+.2f} %"),
        ("Sharpe Ratio",         f"{m['sharpe_ratio']:.3f}"),
        ("Max Drawdown",         f"{m['max_drawdown_pct']:.2f} %"),
        ("Calmar Ratio",         f"{m['calmar_ratio']:.3f}"),
        ("Total Trades",         f"{m['total_trades']}"),
        ("Win Rate",             f"{m['win_rate_pct']:.2f} %"),
        ("Avg Win",              f"${m['avg_win_usd']:,.2f}"),
        ("Avg Loss",             f"${m['avg_loss_usd']:,.2f}"),
        ("Profit Factor",        f"{m['profit_factor']:.3f}"),
    ]

    for label, value in rows:
        color = Fore.GREEN if "Return" in label or "Win" in label else Fore.WHITE
        print(f"  {Fore.YELLOW}{label:<22}{color}{value}")

    print(f"{Fore.CYAN}{sep}{Style.RESET_ALL}\n")


def save_equity_chart(result: BacktestResult, path: str = "equity_curve.png") -> None:
    eq  = result.equity_curve
    dd  = (eq - eq.cummax()) / eq.cummax() * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0d0d0d")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("#333")

    ax1.plot(eq.index, eq.values, color="#00e676", linewidth=1.5, label="Portfolio")
    ax1.fill_between(eq.index, eq.values, eq.values[0], alpha=0.15, color="#00e676")
    ax1.set_ylabel("Equity (USD)", color="white")
    ax1.legend(facecolor="#222", labelcolor="white")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    ax2.fill_between(dd.index, dd.values, 0, color="#ff1744", alpha=0.7)
    ax2.set_ylabel("Drawdown %", color="white")
    ax2.set_xlabel("Date", color="white")

    plt.suptitle("Multi-Strategy Algorithmic Trading System", color="white", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"  Chart saved -> {os.path.abspath(path)}")


def save_trade_log(result: BacktestResult, path: str = "trade_log.csv") -> None:
    rows = []
    for t in result.trades:
        rows.append({
            "ticker":      t.ticker,
            "direction":   "LONG" if t.direction == 1 else "SHORT",
            "entry_date":  t.entry_date,
            "entry_price": t.entry_price,
            "exit_date":   t.exit_date,
            "exit_price":  t.exit_price,
            "shares":      t.shares,
            "pnl_usd":     round(t.pnl, 2),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Trade log saved -> {os.path.abspath(path)}")
