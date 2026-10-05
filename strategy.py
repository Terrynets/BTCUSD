"""
Strategy Module
===============
Implements rule-based BTCUSD signals: LONG, SHORT, FLAT.
"""
import pandas as pd
import numpy as np


def generate_signals(df: pd.DataFrame, config, ml_probs: pd.Series = None) -> pd.DataFrame:
    """
    Generate trading signals based on trend, momentum, volatility, and optional ML probabilities.
    """
    df = df.copy()
    n = len(df)

    signals = np.zeros(n, dtype=int)
    confidences = np.zeros(n)
    reasons = [""] * n

    rsi_lower = config.RSI_LOWER
    rsi_upper = config.RSI_UPPER
    atr_col = df["atr"].values
    macd_hist = df["macd_hist"].values

    if ml_probs is not None:
        ml_up = ml_probs.values
    else:
        ml_up = None

    for i in range(n):
        reason_parts = []
        trend_up = df["trend_up"].iloc[i]
        trend_down = df["trend_down"].iloc[i]
        rsi_val = df["rsi"].iloc[i]
        macd_val = macd_hist[i]
        atr_val = atr_col[i]

        if np.isnan(atr_val) or atr_val <= 0:
            signals[i] = 0
            confidences[i] = 0.0
            reasons[i] = "Invalid ATR"
            continue

        signal = 0
        confidence = 0.0

        if trend_up and (rsi_lower <= rsi_val <= rsi_upper):
            if macd_val > 0:
                signal = 1
                reason_parts.extend(["uptrend", "rsi_ok", "macd_pos"])
                if ml_up is not None:
                    prob = ml_up[i]
                    if prob >= config.ML_PROB_THRESHOLD:
                        confidence = prob
                        reason_parts.append(f"ml_prob={prob:.2f}")
                    else:
                        signal = 0
                        reason_parts.append("ml_prob_low")
                else:
                    confidence = 0.6

        elif trend_down and (rsi_lower <= (100 - rsi_val) <= (100 - rsi_lower)):
            if macd_val < 0:
                signal = -1
                reason_parts.extend(["downtrend", "rsi_ok_short", "macd_neg"])
                if ml_up is not None:
                    prob_down = 1 - ml_up[i]
                    if prob_down >= config.ML_PROB_THRESHOLD:
                        confidence = prob_down
                        reason_parts.append(f"ml_prob_down={prob_down:.2f}")
                    else:
                        signal = 0
                        reason_parts.append("ml_prob_low")
                else:
                    confidence = 0.6

        signals[i] = signal
        confidences[i] = confidence
        reasons[i] = "; ".join(reason_parts) if reason_parts else "no_clear_edge"

    df["signal"] = signals
    df["signal_confidence"] = confidences
    df["signal_reason"] = reasons

    return df
