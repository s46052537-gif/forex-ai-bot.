"""Trend-following strategy: EMA cross / pullback entries with ADX + RSI filters.

Signals are evaluated on CLOSED bars only. The trade is entered at the next
bar's open (backtest) or at the current price (live).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from indicators import ema, rsi, atr, adx


@dataclass
class StrategyParams:
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200
    rsi_period: int = 14
    adx_period: int = 14
    atr_period: int = 14
    adx_min: float = 20.0
    rsi_long_max: float = 70.0
    rsi_short_min: float = 30.0
    sl_atr: float = 1.5
    tp_atr: float = 2.5
    mode: str = "both"          # cross | pullback | both
    max_bars_in_trade: int = 48

    @property
    def warmup(self):
        return self.ema_trend + 10


def add_indicators(df, p):
    d = df.copy()
    d["ema_f"] = ema(d["close"], p.ema_fast)
    d["ema_s"] = ema(d["close"], p.ema_slow)
    d["ema_t"] = ema(d["close"], p.ema_trend)
    d["rsi"] = rsi(d["close"], p.rsi_period)
    d["adx"] = adx(d["high"], d["low"], d["close"], p.adx_period)
    d["atr"] = atr(d["high"], d["low"], d["close"], p.atr_period)
    return d


def generate_signals(df, p):
    """Return a copy of df with indicator columns and a `signal` column (1, -1, 0)."""
    d = add_indicators(df, p)
    c, o, h, l = d["close"], d["open"], d["high"], d["low"]
    f, s, t = d["ema_f"], d["ema_s"], d["ema_t"]

    cross_up = (f.shift(1) <= s.shift(1)) & (f > s) & (c > t)
    cross_dn = (f.shift(1) >= s.shift(1)) & (f < s) & (c < t)

    up_trend = (f > s) & (s > t)
    dn_trend = (f < s) & (s < t)
    pull_long = up_trend & (l <= f) & (c > f) & (c > o)
    pull_short = dn_trend & (h >= f) & (c < f) & (c < o)

    if p.mode == "cross":
        long_raw, short_raw = cross_up, cross_dn
    elif p.mode == "pullback":
        long_raw, short_raw = pull_long, pull_short
    else:
        long_raw, short_raw = cross_up | pull_long, cross_dn | pull_short

    strong = d["adx"] > p.adx_min
    long_sig = long_raw & strong & (d["rsi"] < p.rsi_long_max)
    short_sig = short_raw & strong & (d["rsi"] > p.rsi_short_min)

    sig = np.where(long_sig, 1, np.where(short_sig, -1, 0)).astype(int)
    sig[: p.warmup] = 0
    sig[~np.isfinite(d["atr"].to_numpy(dtype=float))] = 0
    d["signal"] = sig
    return d
