"""
Risk Manager Module
===================
"""
import pandas as pd
import numpy as np


class RiskManager:
    def __init__(self, config):
        self.config = config
        self.initial_capital = config.INITIAL_CAPITAL
        self.position_size_pct = config.POSITION_SIZE_PCT
        self.max_daily_loss_pct = config.MAX_DAILY_LOSS_PCT
        self.max_drawdown_pct = config.MAX_DRAWDOWN_PCT
        self.min_trade_interval = config.MIN_TRADE_INTERVAL
        self.atr_mult_stop = config.ATR_MULT_STOP
        self.atr_mult_target = config.ATR_MULT_TARGET

        self.capital = self.initial_capital
        self.daily_pnl = 0.0
        self.peak_equity = self.initial_capital
        self.last_trade_bar = -999

    def reset_daily(self):
        self.daily_pnl = 0.0

    def check_circuit_breakers(self, current_equity: float) -> bool:
        if self.daily_pnl < -self.max_daily_loss_pct * self.initial_capital:
            return True
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        return drawdown >= self.max_drawdown_pct

    def compute_position_size(self, equity: float, price: float, atr: float) -> float:
        risk_per_trade = equity * self.position_size_pct
        stop_distance = self.atr_mult_stop * atr
        if stop_distance <= 0 or price <= 0:
            return 0.0
        return (risk_per_trade / stop_distance) / price

    def compute_stops_and_targets(self, entry_price: float, atr: float, side: int):
        stop_distance = self.atr_mult_stop * atr
        target_distance = self.atr_mult_target * atr
        if side == 1:
            return entry_price - stop_distance, entry_price + target_distance
        elif side == -1:
            return entry_price + stop_distance, entry_price - target_distance
        return np.nan, np.nan

    def update_pnl(self, pnl: float):
        self.daily_pnl += pnl

    def update_last_trade(self, bar_index: int):
        self.last_trade_bar = bar_index

    def can_trade(self, bar_index: int) -> bool:
        return (bar_index - self.last_trade_bar) >= self.min_trade_interval
