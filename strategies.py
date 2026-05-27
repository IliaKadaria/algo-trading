"""
Three independent strategies. Each returns a signal Series:
  +1 = long  |  -1 = short  |  0 = flat
"""

import pandas as pd
from config import RSI_OVERSOLD, RSI_OVERBOUGHT


class MomentumEMA:
    """
    Long when MACD histogram crosses above zero (bullish momentum confirmed by
    EMA fast > EMA slow).  Short on the opposite cross.
    Volume filter: signal only when volume > 1.1× 20-day MA.
    """

    name = "MomentumEMA"

    def generate(self, df: pd.DataFrame) -> pd.Series:
        prev_hist = df["macd_hist"].shift(1)
        long_cross  = (prev_hist < 0) & (df["macd_hist"] >= 0) & (df["ema_fast"] > df["ema_slow"])
        short_cross = (prev_hist > 0) & (df["macd_hist"] <= 0) & (df["ema_fast"] < df["ema_slow"])
        vol_ok = df["volume"] > df["vol_ma"] * 1.1

        sig = pd.Series(0, index=df.index, name=self.name)
        sig[long_cross  & vol_ok] =  1
        sig[short_cross & vol_ok] = -1
        return sig


class MeanReversionRSI:
    """
    Long when RSI recovers back above oversold threshold from below.
    Short when RSI drops back below overbought threshold from above.
    """

    name = "MeanReversionRSI"

    def generate(self, df: pd.DataFrame) -> pd.Series:
        rsi      = df["rsi"]
        prev_rsi = rsi.shift(1)

        long_entry  = (prev_rsi < RSI_OVERSOLD)   & (rsi >= RSI_OVERSOLD)
        short_entry = (prev_rsi > RSI_OVERBOUGHT)  & (rsi <= RSI_OVERBOUGHT)

        sig = pd.Series(0, index=df.index, name=self.name)
        sig[long_entry]  =  1
        sig[short_entry] = -1
        return sig


class BreakoutBollinger:
    """
    Long breakout: close crosses above upper band with expanding volatility.
    Short breakout: close crosses below lower band with expanding volatility.
    Volatility expansion = ATR today > ATR 10-day MA.
    """

    name = "BreakoutBollinger"

    def generate(self, df: pd.DataFrame) -> pd.Series:
        prev_close = df["close"].shift(1)
        atr_ma     = df["atr"].rolling(10).mean()
        vol_expand = df["atr"] > atr_ma

        long_break  = (prev_close <= df["bb_up"]) & (df["close"] > df["bb_up"]) & vol_expand
        short_break = (prev_close >= df["bb_low"]) & (df["close"] < df["bb_low"]) & vol_expand

        sig = pd.Series(0, index=df.index, name=self.name)
        sig[long_break]  =  1
        sig[short_break] = -1
        return sig


def combined_signal(df: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Weighted vote across all strategies.
    Score ∈ [-1, 1].  Trade when |score| > 0.5 (majority agreement).
    """
    strategies = [MomentumEMA(), MeanReversionRSI(), BreakoutBollinger()]
    score = pd.Series(0.0, index=df.index)
    for s in strategies:
        w = weights.get(s.name, 0.0)
        score += s.generate(df) * w
    signal = pd.Series(0, index=df.index, name="signal")
    signal[score >  0.5] =  1
    signal[score < -0.5] = -1
    return signal
