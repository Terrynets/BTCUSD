"""
Backtester Module
=================
"""
import pandas as pd
import numpy as np
import os


class Backtester:
    def __init__(self, df: pd.DataFrame, config, strategy_fn, risk_manager):
        self.df = df.copy()
        self.config = config
        self.strategy_fn = strategy_fn
        self.risk_manager = risk_manager
        self.capital = config.INITIAL_CAPITAL
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_side = 0
        self.trades = []
        self.equity_curve = []
        self.commission_pct = config.COMMISSION_PCT
        self.slippage_pct = config.SLIPPAGE_PCT

    def run(self, ml_probs: pd.Series = None):
        df = self.strategy_fn(self.df.copy(), self.config, ml_probs=ml_probs)
        n = len(df)
        position = 0.0
        entry_price = 0.0
        entry_side = 0
        capital = self.capital
        self.risk_manager.reset_daily()
        last_trade_day = None

        for i in range(n):
            row = df.iloc[i]
            price = row["close"]
            atr = row["atr"]
            signal = row["signal"]
            confidence = row.get("signal_confidence", 0.0)
            timestamp = row["timestamp"]
            current_day = pd.Timestamp(timestamp).date()

            if last_trade_day != current_day:
                self.risk_manager.reset_daily()
                last_trade_day = current_day

            current_equity = capital + position * price

            if self.risk_manager.check_circuit_breakers(current_equity):
                if position != 0:
                    exit_price = price * (1 - np.sign(position) * self.slippage_pct)
                    pnl = position * (exit_price - entry_price)
                    pnl -= abs(position * exit_price) * self.commission_pct
                    capital += pnl
                    self.trades.append({"exit_index": i, "exit_time": timestamp,
                                        "exit_price": exit_price, "pnl": pnl, "capital": capital})
                    position = 0.0
                    entry_price = 0.0
                    entry_side = 0
                self.equity_curve.append(capital)
                continue

            if position != 0:
                high, low = row["high"], row["low"]
                stop_price, target_price = self.risk_manager.compute_stops_and_targets(entry_price, atr, entry_side)
                exited, exit_price, exit_reason = False, None, ""

                if entry_side == 1:
                    if low <= stop_price:
                        exit_price, exit_reason, exited = stop_price, "stop", True
                    elif high >= target_price:
                        exit_price, exit_reason, exited = target_price, "target", True
                    elif signal != 1:
                        exit_price, exit_reason, exited = price * (1 - self.slippage_pct), "signal_flip", True
                elif entry_side == -1:
                    if high >= stop_price:
                        exit_price, exit_reason, exited = stop_price, "stop", True
                    elif low <= target_price:
                        exit_price, exit_reason, exited = target_price, "target", True
                    elif signal != -1:
                        exit_price, exit_reason, exited = price * (1 + self.slippage_pct), "signal_flip", True

                if exited:
                    pnl = position * (exit_price - entry_price)
                    pnl -= abs(position * exit_price) * self.commission_pct
                    capital += pnl
                    self.risk_manager.update_pnl(pnl)
                    self.trades.append({"exit_index": i, "exit_time": timestamp,
                                        "exit_price": exit_price, "pnl": pnl,
                                        "capital": capital, "exit_reason": exit_reason})
                    position = 0.0
                    entry_price = 0.0
                    entry_side = 0

            if signal != 0 and position == 0 and self.risk_manager.can_trade(i):
                if not (confidence < self.config.ML_PROB_THRESHOLD and ml_probs is not None):
                    btc_size = self.risk_manager.compute_position_size(capital, price, atr)
                    if btc_size > 0:
                        entry_side = signal
                        entry_price = price * (1 + np.sign(entry_side) * self.slippage_pct)
                        position = btc_size * entry_side
                        self.trades.append({"entry_index": i, "entry_time": timestamp,
                                            "entry_price": entry_price, "size": position,
                                            "side": "LONG" if entry_side == 1 else "SHORT"})
                        self.risk_manager.update_last_trade(i)

            self.equity_curve.append(capital + (position * price if position != 0 else 0))

        self.df = df
        self.final_capital = capital
        return self.compute_metrics()

    def compute_metrics(self):
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / (equity[:-1] + 1e-9)
        total_return = (self.final_capital - self.capital) / self.capital
        n_days = max(1, (self.df["timestamp"].iloc[-1] - self.df["timestamp"].iloc[0]).days)
        cagr = (self.final_capital / self.capital) ** (365 / n_days) - 1 if n_days > 0 else 0
        max_dd = self.compute_max_drawdown(equity)
        sharpe = self.compute_sharpe(returns, annual_factor=365 * 24)
        sortino = self.compute_sortino(returns, annual_factor=365 * 24)

        trade_pnls = [t.get("pnl", 0) for t in self.trades if "pnl" in t]
        win_trades = [p for p in trade_pnls if p > 0]
        loss_trades = [p for p in trade_pnls if p < 0]
        win_rate = len(win_trades) / max(1, len(trade_pnls))
        avg_win = np.mean(win_trades) if win_trades else 0
        avg_loss = np.mean(loss_trades) if loss_trades else 0
        pf = abs(sum(win_trades) / sum(loss_trades)) if loss_trades and sum(loss_trades) != 0 else np.inf

        return {"total_return": total_return, "cagr": cagr, "max_drawdown": max_dd,
                "sharpe": sharpe, "sortino": sortino, "win_rate": win_rate,
                "avg_win": avg_win, "avg_loss": avg_loss, "profit_factor": pf,
                "num_trades": len(trade_pnls), "final_capital": self.final_capital}

    def compute_max_drawdown(self, equity):
        peak = np.maximum.accumulate(equity)
        return float(((peak - equity) / (peak + 1e-9)).max())

    def compute_sharpe(self, returns, annual_factor=252):
        if len(returns) < 2:
            return 0.0
        mean_ret, std_ret = np.mean(returns), np.std(returns)
        return float((mean_ret / std_ret) * np.sqrt(annual_factor)) if std_ret > 0 else 0.0

    def compute_sortino(self, returns, annual_factor=252):
        if len(returns) < 2:
            return 0.0
        mean_ret = np.mean(returns)
        downside = returns[returns < 0]
        if len(downside) == 0:
            return 0.0
        downside_std = np.std(downside)
        return float((mean_ret / downside_std) * np.sqrt(annual_factor)) if downside_std > 0 else 0.0

    def save_results(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(self.trades).to_csv(os.path.join(output_dir, "trades.csv"), index=False)
        pd.DataFrame({"equity": self.equity_curve}).to_csv(os.path.join(output_dir, "equity_curve.csv"), index=False)
        print(f"Results saved to {output_dir}")
