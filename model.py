"""
ML Model Module
===============
Optional probability model for up/down moves.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score


class BTCUSDModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.feature_cols = None

    def prepare_target(self, df: pd.DataFrame, horizon: int = 5) -> pd.Series:
        close = df["close"]
        future_close = close.shift(-horizon)
        target = (future_close > close).astype(int)
        return target

    def get_feature_columns(self, df: pd.DataFrame):
        base_features = [
            "ema_fast", "ema_medium", "ema_slow", "rsi",
            "macd_line", "macd_signal", "macd_hist", "atr",
            "bb_upper", "bb_mid", "bb_lower",
            "ret_1", "ret_5", "ret_10", "volatility_20",
            "volume_ratio", "price_over_ema_fast",
            "price_over_ema_medium", "price_over_ema_slow",
            "trend_up", "trend_down",
        ]
        self.feature_cols = [c for c in base_features if c in df.columns]
        return self.feature_cols

    def train(self, df: pd.DataFrame):
        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols].copy()
        y = self.prepare_target(df, horizon=5).loc[X.index]

        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        train_size = int(len(X) * self.config.ML_TRAIN_PCT)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        self.model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        print("ML Model Training Report (test set):")
        print(classification_report(y_test, y_pred, digits=4))
        try:
            print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
        except Exception:
            pass
        return self.model

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError("Model not trained yet.")
        X = df[self.feature_cols].copy()
        mask = ~X.isna().any(axis=1)
        probs = np.zeros(len(df))
        probs[mask] = self.model.predict_proba(X.loc[mask])[:, 1]
        return pd.Series(probs, index=df.index)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "feature_cols": self.feature_cols}, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_cols = data["feature_cols"]
        print(f"Model loaded from {path}")
