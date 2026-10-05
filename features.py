"""
Feature Engineering Module
==========================
Computes technical indicators and derived features for BTCUSD.
"""
import pandas as pd
import numpy as np


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def compute_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


def compute_features(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Compute all features based on config parameters.
    Returns a DataFrame with all indicator columns added.
    """
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Trend
    df["ema_fast"] = compute_ema(close, config.EMA_FAST)
    df["ema_medium"] = compute_ema(close, config.EMA_MEDIUM)
    df["ema_slow"] = compute_ema(close, config.EMA_SLOW)

    # Momentum
    df["rsi"] = compute_rsi(close, config.RSI_PERIOD)
    macd_line, signal_line, histogram = compute_macd(
        close, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL
    )
    df["macd_line"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = histogram

    # Volatility
    df["atr"] = compute_atr(high, low, close, config.ATR_PERIOD)
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = compute_bollinger_bands(
        close, config.BB_PERIOD, config.BB_NUM_STD
    )

    # Returns and volatility features
    df["ret_1"] = close.pct_change()
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)
    df["volatility_20"] = df["ret_1"].rolling(20).std()

    # Volume features
    df["volume_ma20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / (df["volume_ma20"] + 1e-9)

    # Price relative to EMAs
    df["price_over_ema_fast"] = close / (df["ema_fast"] + 1e-9)
    df["price_over_ema_medium"] = close / (df["ema_medium"] + 1e-9)
    df["price_over_ema_slow"] = close / (df["ema_slow"] + 1e-9)

    # Trend regime flags
    df["trend_up"] = (
        (df["ema_fast"] > df["ema_medium"]) & (df["ema_medium"] > df["ema_slow"])
    ).astype(int)
    df["trend_down"] = (
        (df["ema_fast"] < df["ema_medium"]) & (df["ema_medium"] < df["ema_slow"])
    ).astype(int)

    # Drop NaNs created by rolling windows
    df = df.dropna().reset_index(drop=False)

    return df
