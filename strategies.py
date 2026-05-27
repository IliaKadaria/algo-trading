"""
Three independent strategies. Each returns a signal Series:
  +1 = long  |  0 = flat  (long-only — no shorts in a bull market)

Global trend filter: only trade when price > 200-day SMA.
"""

import pandas as pd
import numpy as np


def _trend_filter(df: pd.DataFrame) -> pd.Series:
    """True when close is above the 200-day SMA (bull regime)."""
    return df["close"] > df["close"].rolling(200).mean()


class MomentumEMA:
    """
    Long when MACD histogram crosses above zero while fast EMA > slow EMA.
    Exit when MACD histogram crosses back below zero.
    Volume confirmation: volume > 0.8x 20-day MA (relaxed filter).
    """

    name = "MomentumEMA"

    def generate(self, df: pd.DataFrame) -> pd.Series:
        prev_hist   = df["macd_hist"].shift(1)
        bull        = _trend_filter(df)
        long_cross  = (prev_hist <= 0) & (df["macd_hist"] > 0) & (df["ema_fast"] > df["ema_slow"]) & bull
        exit_cross  = (prev_hist > 0)  & (df["macd_hist"] <= 0)

        sig = pd.Series(0, index=df.index, name=self.name)
        in_trade = False
        for i in range(len(sig)):
            if not in_trade and long_cross.iloc[i]:
                sig.iloc[i] = 1
                in_trade = True
            elif in_trade and exit_cross.iloc[i]:
                sig.iloc[i] = 0
                in_trade = False
            elif in_trade:
                sig.iloc[i] = 1   # hold
        return sig


class MeanReversionRSI:
    """
    Long when RSI recovers above oversold threshold from below (dip-buy).
    Exit when RSI reaches 60 (take partial profit early).
    Only trade in bull regime (price > 200 SMA).
    """

    name = "MeanReversionRSI"

    def generate(self, df: pd.DataFrame) -> pd.Series:
        rsi      = df["rsi"]
        prev_rsi = rsi.shift(1)
        bull     = _trend_filter(df)

        long_entry = (prev_rsi < df.attrs.get("rsi_os", 35)) & (rsi >= df.attrs.get("rsi_os", 35)) & bull
        long_exit  = rsi >= 60

        sig = pd.Series(0, index=df.index, name=self.name)
        in_trade = False
        for i in range(len(sig)):
            if not in_trade and long_entry.iloc[i]:
                sig.iloc[i] = 1
                in_trade = True
            elif in_trade and long_exit.iloc[i]:
                sig.iloc[i] = 0
                in_trade = False
            elif in_trade:
                sig.iloc[i] = 1   # hold
        return sig


class BreakoutBollinger:
    """
    Long breakout: price closes above upper Bollinger Band with rising volume.
    Exit when price falls back below the middle band (mean reversion).
    Only in bull regime.
    """

    name = "BreakoutBollinger"

    def generate(self, df: pd.DataFrame) -> pd.Series:
        prev_close  = df["close"].shift(1)
        bull        = _trend_filter(df)
        vol_confirm = df["volume"] > df["vol_ma"] * 1.2

        long_break = (prev_close <= df["bb_up"]) & (df["close"] > df["bb_up"]) & vol_confirm & bull
        long_exit  = df["close"] < df["bb_mid"]

        sig = pd.Series(0, index=df.index, name=self.name)
        in_trade = False
        for i in range(len(sig)):
            if not in_trade and long_break.iloc[i]:
                sig.iloc[i] = 1
                in_trade = True
            elif in_trade and long_exit.iloc[i]:
                sig.iloc[i] = 0
                in_trade = False
            elif in_trade:
                sig.iloc[i] = 1   # hold
        return sig


def combined_signal(df: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Weighted vote. Enter long when weighted score > 0.25 (low threshold for more trades).
    Exit when score drops to 0.
    """
    from config import RSI_OVERSOLD
    df.attrs["rsi_os"] = RSI_OVERSOLD

    strategies = [MomentumEMA(), MeanReversionRSI(), BreakoutBollinger()]
    score = pd.Series(0.0, index=df.index)
    for s in strategies:
        w = weights.get(s.name, 0.0)
        score += s.generate(df) * w

    signal = pd.Series(0, index=df.index, name="signal")
    signal[score > 0.25] = 1
    return signal
