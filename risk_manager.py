"""
Position sizing and stop-loss calculation.

Uses a fixed-fractional / ATR-based approach:
  risk_per_trade = portfolio_value × MAX_PORTFOLIO_RISK
  stop_distance  = ATR × ATR_STOP_MULT
  shares         = risk_per_trade / stop_distance
  capped at MAX_POSITION_PCT × portfolio_value / price
"""

import numpy as np
from config import MAX_POSITION_PCT, MAX_PORTFOLIO_RISK, ATR_STOP_MULT


def position_size(portfolio_value: float, price: float, atr: float) -> int:
    """Return integer share count for the next trade."""
    if atr <= 0 or price <= 0:
        return 0
    stop_dist = atr * ATR_STOP_MULT
    risk_amt  = portfolio_value * MAX_PORTFOLIO_RISK
    raw_shares = risk_amt / stop_dist
    max_shares = (portfolio_value * MAX_POSITION_PCT) / price
    shares = int(min(raw_shares, max_shares))
    return max(shares, 0)


def stop_price(entry: float, direction: int, atr: float) -> float:
    """Compute ATR-based hard stop from entry price."""
    return entry - direction * atr * ATR_STOP_MULT


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Full Kelly fraction.  Use half-Kelly for live trading:
      f* = W/L - (1-W)/W   where W=win_rate, L=loss_ratio
    """
    if avg_loss == 0 or win_rate <= 0:
        return 0.0
    b = avg_win / abs(avg_loss)
    f = win_rate - (1 - win_rate) / b
    return max(0.0, min(f, 1.0))


def trailing_stop(entry: float, direction: int, highest_favorable: float, atr: float) -> float:
    """
    Trail the stop to lock in gains.
    For longs : stop = max_price_seen - ATR_STOP_MULT × ATR
    For shorts: stop = min_price_seen + ATR_STOP_MULT × ATR
    """
    offset = atr * ATR_STOP_MULT
    if direction == 1:
        return highest_favorable - offset
    else:
        return highest_favorable + offset
