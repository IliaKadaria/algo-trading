"""
Event-driven backtester.

For each ticker:
  1. Apply indicators
  2. Generate combined signal
  3. Simulate trades with ATR-based stops and trailing stops
  4. Track equity curve

Aggregates results across the portfolio.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from indicators import add_all
from strategies import combined_signal
from risk_manager import position_size, stop_price, trailing_stop
from config import (
    INITIAL_CAPITAL, COMMISSION, STRATEGY_WEIGHTS,
    BACKTEST_START, BACKTEST_END, RISK_FREE_RATE,
)
from data_fetcher import fetch


@dataclass
class Trade:
    ticker:    str
    entry_date: pd.Timestamp
    entry_price: float
    direction:  int          # +1 long, -1 short
    shares:     int
    stop:       float
    exit_date:  Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl:        float = 0.0


@dataclass
class BacktestResult:
    equity_curve:  pd.Series
    trades:        list[Trade]
    metrics:       dict = field(default_factory=dict)


# ── core simulation ────────────────────────────────────────────────────────────

def _simulate_ticker(ticker: str, capital_share: float) -> tuple[pd.Series, list[Trade]]:
    """Simulate a single ticker. Returns (equity series, trades)."""
    df  = add_all(fetch(ticker, BACKTEST_START, BACKTEST_END))
    sig = combined_signal(df, STRATEGY_WEIGHTS)

    cash     = capital_share
    position = 0        # shares held (+/-)
    direction = 0
    entry_px  = 0.0
    stop_px   = 0.0
    best_px   = 0.0
    trades: list[Trade] = []
    current_trade: Optional[Trade] = None

    equity = pd.Series(index=df.index, dtype=float)

    for i, (date, row) in enumerate(df.iterrows()):
        price = row["close"]
        atr   = row["atr"]

        # ── manage open position ───────────────────────────────────────────────
        if position != 0:
            best_px = max(best_px, price) if direction == 1 else min(best_px, price)
            trail   = trailing_stop(entry_px, direction, best_px, atr)
            stop_px = max(stop_px, trail) if direction == 1 else min(stop_px, trail)

            hit_stop = (direction == 1 and price <= stop_px) or \
                       (direction == -1 and price >= stop_px)
            reverse  = (sig.iloc[i] != 0) and (sig.iloc[i] != direction)
            should_exit = hit_stop or reverse

            if should_exit:
                exit_px = stop_px if hit_stop else price
                gross   = position * exit_px * direction
                fee     = abs(position) * exit_px * COMMISSION
                cash   += gross - fee
                if current_trade:
                    current_trade.exit_date  = date
                    current_trade.exit_price = exit_px
                    current_trade.pnl        = gross - fee - abs(position) * entry_px * COMMISSION
                    trades.append(current_trade)
                    current_trade = None
                position  = 0
                direction = 0

        # ── open new position ──────────────────────────────────────────────────
        if position == 0 and sig.iloc[i] != 0:
            direction = sig.iloc[i]
            entry_px  = price
            shares    = position_size(cash, price, atr)
            cost      = shares * price * COMMISSION
            if shares > 0 and cash >= shares * price + cost:
                position  = shares * direction
                cash     -= shares * price + cost
                stop_px   = stop_price(entry_px, direction, atr)
                best_px   = price
                current_trade = Trade(
                    ticker=ticker,
                    entry_date=date,
                    entry_price=entry_px,
                    direction=direction,
                    shares=shares,
                    stop=stop_px,
                )

        # ── mark-to-market equity ──────────────────────────────────────────────
        equity.iloc[i] = cash + position * price

    # close any open position at end
    if position != 0 and current_trade:
        last_price = df["close"].iloc[-1]
        gross = position * last_price * direction
        fee   = abs(position) * last_price * COMMISSION
        cash += gross - fee
        current_trade.exit_date  = df.index[-1]
        current_trade.exit_price = last_price
        current_trade.pnl        = gross - fee - abs(position) * entry_px * COMMISSION
        trades.append(current_trade)

    return equity, trades


# ── portfolio aggregation ──────────────────────────────────────────────────────

def run_backtest(tickers: list[str]) -> BacktestResult:
    n     = len(tickers)
    share = INITIAL_CAPITAL / n
    all_equity: list[pd.Series] = []
    all_trades: list[Trade]     = []

    for ticker in tickers:
        print(f"  Backtesting {ticker}...")
        eq, tr = _simulate_ticker(ticker, share)
        all_equity.append(eq)
        all_trades.extend(tr)

    # align on common dates, sum equity
    combined = pd.concat(all_equity, axis=1).ffill().sum(axis=1)
    combined.name = "Portfolio"

    result = BacktestResult(equity_curve=combined, trades=all_trades)
    result.metrics = compute_metrics(combined, all_trades)
    return result


# ── performance metrics ────────────────────────────────────────────────────────

def compute_metrics(equity: pd.Series, trades: list[Trade]) -> dict:
    returns     = equity.pct_change().dropna()
    total_ret   = (equity.iloc[-1] / equity.iloc[0]) - 1
    ann_factor  = 252
    ann_ret     = (1 + total_ret) ** (ann_factor / len(returns)) - 1
    vol         = returns.std() * np.sqrt(ann_factor)
    sharpe      = (ann_ret - RISK_FREE_RATE) / vol if vol else 0.0

    roll_max    = equity.cummax()
    drawdown    = (equity - roll_max) / roll_max
    max_dd      = drawdown.min()
    calmar      = ann_ret / abs(max_dd) if max_dd else 0.0

    pnls        = [t.pnl for t in trades if t.pnl is not None]
    wins        = [p for p in pnls if p > 0]
    losses      = [p for p in pnls if p <= 0]
    win_rate    = len(wins) / len(pnls) if pnls else 0.0
    avg_win     = np.mean(wins)  if wins   else 0.0
    avg_loss    = np.mean(losses) if losses else 0.0
    profit_factor = abs(sum(wins) / sum(losses)) if losses else float("inf")

    return {
        "total_return_pct":  round(total_ret * 100, 2),
        "annual_return_pct": round(ann_ret * 100, 2),
        "sharpe_ratio":      round(sharpe, 3),
        "max_drawdown_pct":  round(max_dd * 100, 2),
        "calmar_ratio":      round(calmar, 3),
        "total_trades":      len(pnls),
        "win_rate_pct":      round(win_rate * 100, 2),
        "avg_win_usd":       round(avg_win, 2),
        "avg_loss_usd":      round(avg_loss, 2),
        "profit_factor":     round(profit_factor, 3),
        "final_equity_usd":  round(equity.iloc[-1], 2),
    }
