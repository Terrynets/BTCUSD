"""
Data Loader Module
==================
Fetches and preprocesses BTCUSD OHLCV data.
Supports:
  - Loading from CSV
  - Generating synthetic data for code testing only (NOT for strategy evaluation)
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os


class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_from_csv(self) -> pd.DataFrame:
        """Load OHLCV data from CSV file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        df = pd.read_csv(self.data_path)

        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()

        # Map common column name variations
        column_mapping = {
            "time": "timestamp",
            "date": "timestamp",
            "datetime": "timestamp",
            "open_time": "timestamp",
            "close_price": "close",
            "high_price": "high",
            "low_price": "low",
            "open_price": "open",
            "vol": "volume",
        }
        df = df.rename(columns=column_mapping)

        # Ensure required columns exist
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Parse timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        # Ensure numeric types
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort by timestamp and remove duplicates
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        # Drop rows with NaN values
        df = df.dropna()

        return df

    def generate_sample_data(
        self,
        start_date: str = "2018-01-01",
        end_date: str = "2025-12-31",
        timeframe: str = "1h",
    ) -> pd.DataFrame:
        """
        Generate realistic sample BTCUSD data for testing code paths only.
        DO NOT use this to evaluate strategy profitability.
        """
        print("Generating sample BTCUSD data (for code testing only)...")

        if timeframe == "1h":
            freq = "1H"
        elif timeframe == "4h":
            freq = "4H"
        elif timeframe == "1D":
            freq = "1D"
        else:
            freq = "1H"

        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        n = len(dates)

        np.random.seed(42)

        initial_price = 10000.0
        annual_drift = 0.15
        annual_vol = 0.75

        dt = 1 / (365 * 24)
        drift = annual_drift * dt
        vol = annual_vol * np.sqrt(dt)

        returns = np.random.normal(drift, vol, n)

        regime_length = 2000
        for i in range(0, n, regime_length):
            regime_end = min(i + regime_length, n)
            if np.random.random() < 0.4:
                returns[i:regime_end] *= -1.5

        prices = initial_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame(index=dates)
        df["close"] = prices

        hourly_vol = np.abs(np.random.normal(0.005, 0.003, n))
        df["high"] = df["close"] * (1 + hourly_vol)
        df["low"] = df["close"] * (1 - hourly_vol)
        df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])

        base_volume = 1000.0
        df["volume"] = base_volume * (1 + 2 * hourly_vol / hourly_vol.mean())
        df["volume"] = df["volume"] * np.random.uniform(0.5, 1.5, n)

        df["high"] = df[["high", "close", "open"]].max(axis=1)
        df["low"] = df[["low", "close", "open"]].min(axis=1)

        df = df.dropna()
        print(f"Generated {len(df)} bars from {start_date} to {end_date}")

        return df

    def save_data(self, df: pd.DataFrame, output_path: str):
        """Save dataframe to CSV."""
        df.to_csv(output_path)
        print(f"Data saved to {output_path}")
