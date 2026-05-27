"""
Central configuration for the trading system.
Edit these values to customize your strategy and risk parameters.
"""

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
MAX_POSITION_PCT   = 0.90        # deploy up to 90 % of per-ticker capital when signal fires
MAX_PORTFOLIO_RISK = 0.06        # risk 6 % of per-ticker capital per trade
ATR_STOP_MULT      = 1.5         # tighter stop = larger position size
RISK_FREE_RATE     = 0.05        # annual, used for Sharpe

# -- Strategy weights (must sum to 1.0) ----------------------------------------
# Optimised automatically by optimizer.py
STRATEGY_WEIGHTS = {
    "MomentumEMA":       0.20,
    "MeanReversionRSI":  0.60,
    "BreakoutBollinger": 0.20,
}

# -- EMA Crossover (Momentum) --------------------------------------------------
EMA_FAST    = 8
EMA_SLOW    = 26
MACD_SIGNAL = 9

# -- RSI Mean-Reversion --------------------------------------------------------
RSI_PERIOD     = 14
RSI_OVERSOLD   = 35
RSI_OVERBOUGHT = 75

# -- Bollinger-Band Breakout ---------------------------------------------------
BB_PERIOD = 20
BB_STD    = 2.0
