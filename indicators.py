"""Technical indicators computed on a price DataFrame."""

import pandas as pd
import numpy as np
from config import (
    EMA_FAST, EMA_SLOW, MACD_SIGNAL,
    RSI_PERIOD, BB_PERIOD, BB_STD,
)


def add_ema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"]  = df["close"].ewm(span=EMA_FAST,  adjust=False).mean()
    df["ema_slow"]  = df["close"].ewm(span=EMA_SLOW,  adjust=False).mean()
    macd_line       = df["ema_fast"] - df["ema_slow"]
    df["macd"]      = macd_line
    df["macd_signal"] = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    delta  = df["close"].diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    avg_l  = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mid          = df["close"].rolling(BB_PERIOD).mean()
    std          = df["close"].rolling(BB_PERIOD).std()
    df["bb_mid"] = mid
    df["bb_up"]  = mid + BB_STD * std
    df["bb_low"] = mid - BB_STD * std
    df["bb_pct"] = (df["close"] - df["bb_low"]) / (df["bb_up"] - df["bb_low"])
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift()).abs()
    lc  = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=period, adjust=False).mean()
    return df


def add_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["vol_ma"] = df["volume"].rolling(period).mean()
    return df


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    for fn in (add_ema, add_rsi, add_bollinger, add_atr, add_volume_ma):
        df = fn(df)
    return df.dropna()
