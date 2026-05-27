"""
Entry point — run the full backtest pipeline.

Usage:
    python main.py
    python main.py --tickers AAPL MSFT NVDA
"""

import argparse
import sys
from config import WATCHLIST
from backtester import run_backtest
from reporter import print_report, save_equity_chart, save_trade_log


def parse_args():
    p = argparse.ArgumentParser(description="Multi-Strategy Algo Trading Backtester")
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Override ticker list (default: config.WATCHLIST)")
    p.add_argument("--no-chart", action="store_true", help="Skip chart generation")
    p.add_argument("--no-log",   action="store_true", help="Skip trade log CSV")
    return p.parse_args()


def main():
    args    = parse_args()
    tickers = args.tickers or WATCHLIST

    print(f"\n>>> Running backtest on: {', '.join(tickers)}\n")
    result = run_backtest(tickers)

    print_report(result)

    if not args.no_chart:
        save_equity_chart(result)

    if not args.no_log:
        save_trade_log(result)

    print("\nDone.\n")
    return result


if __name__ == "__main__":
    main()
