"""
Grid-search optimizer.

Tries every combination of EMA periods, RSI thresholds and strategy weights,
scores each by Sharpe ratio (with a drawdown penalty), prints the top 5,
then overwrites config.py with the best parameters found.
"""

import itertools
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

# ── parameter grid ─────────────────────────────────────────────────────────────
EMA_FAST_VALUES    = [8, 12, 20]
EMA_SLOW_VALUES    = [26, 50, 100]
RSI_OS_VALUES      = [25, 30, 35]
RSI_OB_VALUES      = [65, 70, 75]
WEIGHT_SETS = [
    {"MomentumEMA": 0.60, "MeanReversionRSI": 0.20, "BreakoutBollinger": 0.20},
    {"MomentumEMA": 0.40, "MeanReversionRSI": 0.35, "BreakoutBollinger": 0.25},
    {"MomentumEMA": 0.20, "MeanReversionRSI": 0.60, "BreakoutBollinger": 0.20},
    {"MomentumEMA": 0.33, "MeanReversionRSI": 0.33, "BreakoutBollinger": 0.34},
    {"MomentumEMA": 0.50, "MeanReversionRSI": 0.50, "BreakoutBollinger": 0.00},
]

# tickers used during optimisation (fast subset)
OPT_TICKERS = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]

# ── imports that depend on config (must come after we patch it) ─────────────────
import config as cfg
from data_fetcher import fetch
from indicators import add_all
from strategies import combined_signal
from risk_manager import position_size, stop_price, trailing_stop
from config import (
    INITIAL_CAPITAL, COMMISSION, BACKTEST_START, BACKTEST_END, RISK_FREE_RATE
)


@dataclass
class Result:
    ema_fast: int
    ema_slow: int
    rsi_os: int
    rsi_ob: int
    weights: dict
    sharpe: float
    annual_ret: float
    max_dd: float
    score: float   # sharpe penalised by drawdown


def _simulate(df: pd.DataFrame, weights: dict, capital: float) -> pd.Series:
    sig  = combined_signal(df, weights)
    cash = capital
    position = direction = 0
    entry_px = stop_px = best_px = 0.0
    equity = pd.Series(index=df.index, dtype=float)

    for i, (_, row) in enumerate(df.iterrows()):
        price, atr = row["close"], row["atr"]

        if position != 0:
            best_px = max(best_px, price) if direction == 1 else min(best_px, price)
            trail   = trailing_stop(entry_px, direction, best_px, atr)
            stop_px = max(stop_px, trail) if direction == 1 else min(stop_px, trail)
            hit     = (direction == 1 and price <= stop_px) or \
                      (direction == -1 and price >= stop_px)
            rev     = sig.iloc[i] != 0 and sig.iloc[i] != direction
            if hit or rev:
                exit_px = stop_px if hit else price
                cash   += position * exit_px * direction - abs(position) * exit_px * COMMISSION
                position = direction = 0

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

        equity.iloc[i] = cash + position * price

    return equity


def _metrics(equity: pd.Series) -> tuple[float, float, float]:
    ret  = equity.pct_change().dropna()
    tot  = equity.iloc[-1] / equity.iloc[0] - 1
    ann  = (1 + tot) ** (252 / max(len(ret), 1)) - 1
    vol  = ret.std() * np.sqrt(252)
    shr  = (ann - RISK_FREE_RATE) / vol if vol else -99.0
    dd   = ((equity - equity.cummax()) / equity.cummax()).min()
    return round(shr, 4), round(ann * 100, 2), round(dd * 100, 2)


def run():
    print("\nLoading data...")
    data = {}
    for t in OPT_TICKERS:
        try:
            data[t] = fetch(t, BACKTEST_START, BACKTEST_END)
        except Exception as e:
            print(f"  Warning: could not load {t}: {e}")

    combos = list(itertools.product(
        EMA_FAST_VALUES, EMA_SLOW_VALUES, RSI_OS_VALUES, RSI_OB_VALUES, WEIGHT_SETS
    ))
    # filter: fast must be < slow
    combos = [(f, s, ro, rb, w) for f, s, ro, rb, w in combos if f < s]

    total = len(combos)
    print(f"Running {total} combinations on {len(data)} tickers...\n")

    results: list[Result] = []

    for idx, (ema_f, ema_s, rsi_os, rsi_ob, weights) in enumerate(combos):
        if idx % 50 == 0:
            pct = idx / total * 100
            bar = "#" * int(pct / 2) + "." * (50 - int(pct / 2))
            print(f"\r  [{bar}] {pct:.0f}%  ({idx}/{total})", end="", flush=True)

        # patch config in-memory
        cfg.EMA_FAST      = ema_f
        cfg.EMA_SLOW      = ema_s
        cfg.RSI_OVERSOLD  = rsi_os
        cfg.RSI_OVERBOUGHT = rsi_ob

        # rebuild indicators + signals with patched config
        import indicators as ind
        ind.EMA_FAST = ema_f
        ind.EMA_SLOW = ema_s
        ind.RSI_PERIOD = cfg.RSI_PERIOD   # unchanged
        import strategies as strat
        strat.RSI_OVERSOLD  = rsi_os
        strat.RSI_OVERBOUGHT = rsi_ob

        share = INITIAL_CAPITAL / len(data)
        equities = []
        for df_raw in data.values():
            try:
                df = add_all(df_raw)
                eq = _simulate(df, weights, share)
                equities.append(eq)
            except Exception:
                pass

        if not equities:
            continue

        combined = pd.concat(equities, axis=1).ffill().sum(axis=1)
        sharpe, ann_ret, max_dd = _metrics(combined)
        # score = sharpe minus penalty for deep drawdowns
        score = sharpe + (max_dd / 100) * 0.5   # dd is negative, so this subtracts

        results.append(Result(ema_f, ema_s, rsi_os, rsi_ob, weights,
                               sharpe, ann_ret, max_dd, score))

    print(f"\r  [{'#'*50}] 100%  ({total}/{total})\n")

    results.sort(key=lambda r: r.score, reverse=True)
    best = results[0]

    # ── print top 5 ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  TOP 5 PARAMETER SETS")
    print("=" * 70)
    for i, r in enumerate(results[:5], 1):
        wname = max(r.weights, key=r.weights.get)
        print(f"  #{i}  EMA {r.ema_fast}/{r.ema_slow}  RSI {r.rsi_os}/{r.rsi_ob}"
              f"  dominant={wname:<20}"
              f"  Sharpe={r.sharpe:+.3f}  Ann={r.annual_ret:+.1f}%  DD={r.max_dd:.1f}%")
    print("=" * 70)

    # ── write best params back to config.py ────────────────────────────────────
    _write_config(best)
    print(f"\n  config.py updated with best parameters (Sharpe={best.sharpe:+.3f})\n")
    return best


def _write_config(r: Result) -> None:
    w = r.weights
    lines = f"""\"\"\"
Central configuration for the trading system.
Edit these values to customize your strategy and risk parameters.
\"\"\"

# -- Universe ------------------------------------------------------------------
WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "SPY", "QQQ", "AMD",
]

# -- Backtest ------------------------------------------------------------------
BACKTEST_START   = "2020-01-01"
BACKTEST_END     = "2024-12-31"
INITIAL_CAPITAL  = 100_000       # USD
COMMISSION       = 0.001         # 0.1 % per trade (round-trip 0.2 %)

# -- Risk management -----------------------------------------------------------
MAX_POSITION_PCT   = 0.15        # max 15 % of portfolio per ticker
MAX_PORTFOLIO_RISK = 0.02        # max 2 % of portfolio at risk per trade
ATR_STOP_MULT      = 2.0         # stop = entry +/- 2 x ATR(14)
RISK_FREE_RATE     = 0.05        # annual, used for Sharpe

# -- Strategy weights (must sum to 1.0) ----------------------------------------
# Optimised automatically by optimizer.py
STRATEGY_WEIGHTS = {{
    "MomentumEMA":       {w["MomentumEMA"]:.2f},
    "MeanReversionRSI":  {w["MeanReversionRSI"]:.2f},
    "BreakoutBollinger": {w["BreakoutBollinger"]:.2f},
}}

# -- EMA Crossover (Momentum) --------------------------------------------------
EMA_FAST    = {r.ema_fast}
EMA_SLOW    = {r.ema_slow}
MACD_SIGNAL = 9

# -- RSI Mean-Reversion --------------------------------------------------------
RSI_PERIOD     = 14
RSI_OVERSOLD   = {r.rsi_os}
RSI_OVERBOUGHT = {r.rsi_ob}

# -- Bollinger-Band Breakout ---------------------------------------------------
BB_PERIOD = 20
BB_STD    = 2.0
"""
    with open("config.py", "w") as f:
        f.write(lines)


if __name__ == "__main__":
    run()
