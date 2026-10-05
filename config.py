"""
Configuration for BTCUSD Trading Analyzer
==========================================
Adjust parameters here to tune behavior.

WARNING: This is a decision-support tool, not a guarantee of profit.
Past performance ≠ future results. Use strict risk controls.
"""

# -------------------------
# Data settings
# -------------------------
DATA_FILE = "data/btcusd_1h.csv"  # Path to OHLCV data
TIMEFRAME = "1h"  # Options: "1h", "4h", "1D"

# -------------------------
# Trading parameters
# -------------------------
INITIAL_CAPITAL = 10000.0  # Starting capital in USD
POSITION_SIZE_PCT = 0.02  # 2% of equity per trade
COMMISSION_PCT = 0.001  # 0.1% per trade (10 bps)
SLIPPAGE_PCT = 0.0005  # 0.05% slippage assumption

# -------------------------
# Technical indicator parameters
# -------------------------
EMA_FAST = 20
EMA_MEDIUM = 50
EMA_SLOW = 200

RSI_PERIOD = 14
RSI_LOWER = 45
RSI_UPPER = 70

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

ATR_PERIOD = 14
ATR_MULT_STOP = 2.0  # Stop loss = 2x ATR
ATR_MULT_TARGET = 4.0  # Take profit = 4x ATR (2:1 reward:risk)

BB_PERIOD = 20
BB_NUM_STD = 2.0

# -------------------------
# ML model parameters
# -------------------------
ML_ENABLED = True
ML_PROB_THRESHOLD = 0.55  # Minimum probability to take signal
ML_FEATURE_LOOKBACK = 50  # Bars to use for features
ML_TRAIN_PCT = 0.7  # 70% train, 30% test
ML_MODEL_PATH = "models/btcusd_model.pkl"

# -------------------------
# Risk management
# -------------------------
MAX_DAILY_LOSS_PCT = 0.05  # 5% max daily loss
MAX_DRAWDOWN_PCT = 0.20  # 20% max drawdown circuit breaker
MIN_TRADE_INTERVAL = 2  # Minimum bars between trades

# -------------------------
# Backtest settings
# -------------------------
TRAIN_START = "2018-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2025-12-31"

# -------------------------
# Output settings
# -------------------------
RESULTS_DIR = "results"
LOG_FILE = "results/analyzer.log"
