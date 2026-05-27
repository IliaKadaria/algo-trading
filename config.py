"""
Central configuration for the trading system.
Edit these values to customize your strategy and risk parameters.
"""

# ── Universe ──────────────────────────────────────────────────────────────────
WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "SPY", "QQQ", "AMD",
]

# ── Backtest ──────────────────────────────────────────────────────────────────
BACKTEST_START   = "2020-01-01"
BACKTEST_END     = "2024-12-31"
INITIAL_CAPITAL  = 100_000       # USD
COMMISSION       = 0.001         # 0.1 % per trade (round-trip 0.2 %)

# ── Risk management ───────────────────────────────────────────────────────────
MAX_POSITION_PCT   = 0.15        # max 15 % of portfolio per ticker
MAX_PORTFOLIO_RISK = 0.02        # max 2 % of portfolio at risk per trade
ATR_STOP_MULT      = 2.0         # stop = entry ± 2 × ATR(14)
RISK_FREE_RATE     = 0.05        # annual, used for Sharpe

# ── Strategy weights (must sum to 1.0) ────────────────────────────────────────
STRATEGY_WEIGHTS = {
    "MomentumEMA":       0.40,
    "MeanReversionRSI":  0.35,
    "BreakoutBollinger": 0.25,
}

# ── EMA Crossover (Momentum) ──────────────────────────────────────────────────
EMA_FAST   = 12
EMA_SLOW   = 26
MACD_SIGNAL = 9

# ── RSI Mean-Reversion ────────────────────────────────────────────────────────
RSI_PERIOD    = 14
RSI_OVERSOLD  = 30
RSI_OVERBOUGHT = 70

# ── Bollinger-Band Breakout ────────────────────────────────────────────────────
BB_PERIOD = 20
BB_STD    = 2.0
