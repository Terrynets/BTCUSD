"""
Main Entry Point
================
"""
import argparse
import pandas as pd
import numpy as np
import os
import sys

import config
from data_loader import DataLoader
from features import compute_features
from model import BTCUSDModel
from strategy import generate_signals
from risk_manager import RiskManager
from backtester import Backtester


def main():
    parser = argparse.ArgumentParser(description="BTCUSD Trading Analyzer")
    parser.add_argument("--mode", choices=["generate_data", "backtest", "live_signal"],
                        default="backtest", help="Mode to run")
    parser.add_argument("--data", type=str, default=config.DATA_FILE, help="Path to OHLCV CSV data")
    parser.add_argument("--use_ml", action="store_true", default=config.ML_ENABLED, help="Use ML model")
    args = parser.parse_args()

    print("=== BTCUSD Trading Analyzer ===")
    print(f"Mode: {args.mode}")

    loader = DataLoader(args.data)
    if args.mode == "generate_data":
        df = loader.generate_sample_data(start_date=config.TRAIN_START, end_date=config.TEST_END,
                                         timeframe=config.TIMEFRAME)
        loader.save_data(df, args.data)
        print("Sample data generated. Run backtest next.")
        return

    if not os.path.exists(args.data):
        print(f"Data file not found: {args.data}")
        print("Run with --mode generate_data first, or provide real OHLCV data.")
        sys.exit(1)

    df = loader.load_from_csv()
    print(f"Loaded {len(df)} bars from {args.data}")

    df = compute_features(df, config)
    print(f"Features computed. Columns: {len(df.columns)}")

    ml_probs = None
    if args.use_ml and config.ML_ENABLED:
        print("Training ML model...")
        model = BTCUSDModel(config)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        train_mask = (df["timestamp"] >= config.TRAIN_START) & (df["timestamp"] <= config.TRAIN_END)
        train_df = df.loc[train_mask].copy()
        model.train(train_df)
        model.save(config.ML_MODEL_PATH)
        ml_probs = model.predict_proba(df)
        print("ML probabilities generated.")

    if args.mode == "backtest":
        risk_mgr = RiskManager(config)
        bt = Backtester(df, config, generate_signals, risk_mgr)
        metrics = bt.run(ml_probs=ml_probs)
        print("\n=== Backtest Metrics ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        bt.save_results(config.RESULTS_DIR)
        print(f"\nResults saved to {config.RESULTS_DIR}")

    elif args.mode == "live_signal":
        last_row = df.iloc[-1]
        last_prob = ml_probs.iloc[-1] if ml_probs is not None else None
        df_sig = generate_signals(df, config, ml_probs=ml_probs)
        last_sig = df_sig.iloc[-1]
        print("\n=== Latest Signal ===")
        print(f"Timestamp: {last_sig['timestamp']}")
        print(f"Close: {last_sig['close']:.2f}")
        print(f"Signal: {['FLAT', 'LONG', 'SHORT'][int(last_sig['signal']) + 1]}")
        print(f"Confidence: {last_sig['signal_confidence']:.2f}")
        print(f"Reason: {last_sig['signal_reason']}")
        if last_prob is not None:
            print(f"ML prob(up): {last_prob:.2f}")


if __name__ == "__main__":
    main()
