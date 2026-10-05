# BTCUSD Trading Analyzer

A modular Python framework for building, testing, and running a **BTCUSD trading analyzer** with technical indicators, optional ML signals, and strict risk controls.

> **WARNING:** This is a **decision-support tool**, not a guarantee of profit.  
> Past performance ≠ future results. Crypto trading involves substantial risk of loss.  
> Do not risk money you cannot afford to lose.

## Features

- **Data handling**: Load OHLCV from CSV or generate synthetic data for code tests.
- **Feature engineering**: EMA, RSI, MACD, ATR, Bollinger Bands, volume, regime features.
- **Strategy**: Rule-based LONG/SHORT/FLAT signals with confidence scores.
- **ML layer**: Optional gradient-boosted classifier for up/down probabilities.
- **Risk management**: ATR stops/targets, position sizing, daily loss and drawdown limits.
- **Backtester**: Fees, slippage, trade log, equity curve, and performance metrics.

## Project Structure

```
btcusd_analyzer/
├── config.py            # All tunable parameters
├── data_loader.py       # Load/generate OHLCV data
├── features.py          # Technical indicators
├── strategy.py          # Signal generation
├── model.py             # Optional ML model
├── risk_manager.py      # Position sizing, stops, limits
├── backtester.py        # Backtesting engine
├── main.py              # CLI entry point
├── test_analyzer.py     # Basic tests
├── requirements.txt     # Dependencies
├── README.md            # This file
├── .gitignore
├── data/                # Place your BTCUSD CSV here
├── models/              # Saved ML models
├── results/             # Backtest outputs
└── notebooks/           # Jupyter notebooks
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**:
   - Export BTCUSD/BTCUSDT OHLCV as CSV: `timestamp,open,high,low,close,volume`
   - Save as `data/btcusd_1h.csv`

   Or generate sample data (code test only):
   ```bash
   python main.py --mode generate_data --data data/btcusd_1h.csv
   ```

## Usage

### Run a backtest
```bash
python main.py --mode backtest --data data/btcusd_1h.csv --use_ml
```

Outputs in `results/`:
- `trades.csv` – all trades with PnL
- `equity_curve.csv` – equity over time

### Get latest signal
```bash
python main.py --mode live_signal --data data/btcusd_1h.csv --use_ml
```

## Testing
```bash
pytest test_analyzer.py -v
```

## Important Risk Notes

- No strategy is perfect or always correct.
- Backtests can overfit; use out-of-sample validation.
- Include realistic fees, slippage, and liquidity constraints.
- Start with **paper trading** before using real capital.
- Never risk funds needed for living expenses, debt, education, or emergencies.

## License
MIT License – use at your own risk.
