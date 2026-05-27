"""Download and cache OHLCV data via yfinance."""

import yfinance as yf
import pandas as pd
from config import BACKTEST_START, BACKTEST_END


def fetch(ticker: str, start: str = BACKTEST_START, end: str = BACKTEST_END) -> pd.DataFrame:
    """Return daily OHLCV DataFrame for *ticker*. Columns lowercase."""
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")
    df = raw.copy()
    # yfinance >= 1.0 returns a MultiIndex (PriceType, Ticker) — flatten to single level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [price.lower() for price, _tick in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.dropna(inplace=True)
    return df


def fetch_multi(tickers: list[str], start: str = BACKTEST_START, end: str = BACKTEST_END) -> dict[str, pd.DataFrame]:
    """Return {ticker: ohlcv_df} for a list of tickers."""
    return {t: fetch(t, start, end) for t in tickers}
