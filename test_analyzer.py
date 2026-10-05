"""
Test Suite
==========
"""
import pytest
import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import DataLoader
from features import compute_features
from strategy import generate_signals
from risk_manager import RiskManager
from backtester import Backtester


@pytest.fixture
def sample_df():
    loader = DataLoader("dummy")
    df = loader.generate_sample_data(start_date="2020-01-01", end_date="2021-12-31", timeframe="1h")
    return compute_features(df, config)


def test_data_loader_generate():
    loader = DataLoader("dummy")
    df = loader.generate_sample_data(start_date="2020-01-01", end_date="2020-01-10", timeframe="1h")
    assert len(df) > 0
    assert set(["open", "high", "low", "close", "volume"]).issubset(df.columns)


def test_features_computation(sample_df):
    required = ["ema_fast", "ema_medium", "ema_slow", "rsi", "macd_line",
                "macd_hist", "atr", "bb_upper", "bb_lower", "trend_up", "trend_down"]
    for col in required:
        assert col in sample_df.columns, f"Missing feature: {col}"
    assert not sample_df["rsi"].isna().all()
    assert not sample_df["atr"].isna().all()


def test_strategy_signals(sample_df):
    df_sig = generate_signals(sample_df, config, ml_probs=None)
    assert "signal" in df_sig.columns
    assert df_sig["signal"].isin([-1, 0, 1]).all()
    assert "signal_confidence" in df_sig.columns
    assert (df_sig["signal_confidence"] >= 0).all()
    assert (df_sig["signal_confidence"] <= 1).all()


def test_risk_manager():
    rm = RiskManager(config)
    assert rm.initial_capital == config.INITIAL_CAPITAL
    assert rm.check_circuit_breakers(config.INITIAL_CAPITAL) is False
    assert rm.check_circuit_breakers(config.INITIAL_CAPITAL * 0.7) is True
    stop, target = rm.compute_stops_and_targets(50000.0, 1000.0, 1)
    assert stop < 50000.0
    assert target > 50000.0


def test_backtester_basic(sample_df):
    rm = RiskManager(config)
    bt = Backtester(sample_df, config, generate_signals, rm)
    metrics = bt.run(ml_probs=None)
    assert "total_return" in metrics
    assert "max_drawdown" in metrics
    assert "sharpe" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
