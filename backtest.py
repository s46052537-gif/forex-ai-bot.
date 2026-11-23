"""Backtester.

Examples (run inside the project folder):
    py backtest.py --csv data/EURUSD_H1.csv --symbol EURUSD
    py backtest.py --csv data/EURUSD_H1.csv --symbol EURUSD --ai
    py backtest.py --synthetic          (fake data, only to check the program works)
"""
import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import load_config
from strategy import generate_signals
from trade_sim import walk_trade
from risk import position_size, value_per_unit
from ai_filter import train_filter


@dataclass
class BTConfig:
    initial_balance: float = 1000.0
    risk_pct: float = 0.5
    spread_pips: float = 1.2
    contract_size: float = 100000.0
    min_lot: float = 0.01
    lot_step: float = 0.01
    max_lot: float = 100.0
    max_risk_overshoot: float = 2.0
    max_daily_loss_pct: float = 3.0
    trade_hour_start: int = 7
    trade_hour_end: int = 20


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"time", "open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {sorted(need)}; found {list(df.columns)}")
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def make_synthetic(n=20000, seed=7, start_price=1.10, sigma=0.0006):
    """Random regime-switching prices. For testing the code ONLY."""
    rng = np.random.default_rng(seed)
    drifts = np.array([0.00004, -0.00004, 0.0])
    state = 2
    rets = np.empty(n)
    for i in range(n):
        if rng.random() < 0.01:
            state = int(rng.integers(0, 3))
        rets[i] = drifts[state] + sigma * rng.standard_normal()
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    up = np.abs(rng.standard_normal(n)) * sigma * start_price * 0.5
    dn = np.abs(rng.standard_normal(n)) * sigma * start_price * 0.5
    return pd.DataFrame({
        "time": pd.date_range("2022-01-03", periods=n, freq="h"),
        "open": open_, "high": np.maximum(open_, close) + up,
        "low": np.minimum(open_, close) - dn, "close": close,
    })


def run_backtest(d, symbol, p, cfg, allow=None, start_idx=None):
    """d: output of generate_signals(). allow: optional bool array (True = take signal)."""
    n = len(d)
    o = d["open"].to_numpy(float)
    h = d["high"].to_numpy(float)
    l = d["low"].to_numpy(float)
    c = d["close"].to_numpy(float)
    atr = d["atr"].to_numpy(float)
    sig = d["signal"].to_numpy(int)
    times = d["time"] if "time" in d.columns else None
    hours = times.dt.hour.to_numpy() if times is not None else None
    days = times.dt.normalize().to_numpy().astype("datetime64[D]").astype(np.int64) if times is not None else None

    pip = 0.01 if symbol.upper().endswith("JPY") else 0.0001
    spread_price = cfg.spread_pips * pip
    balance = cfg.initial_balance
    day_key, day_start = None, balance
    rows = []
    i = max(start_idx or 0, p.warmup)

    while i < n - 1:
        s = int(sig[i])
        if s == 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
            i += 1
            continue
        if allow is not None and not allow[i]:
            i += 1
            continue
        if hours is not None and not (cfg.trade_hour_start <= hours[i] < cfg.trade_hour_end):
            i += 1
            continue
        if days is not None and days[i] != day_key:
            day_key, day_start = days[i], balance
        if balance <= day_start * (1.0 - cfg.max_daily_loss_pct / 100.0):
            i += 1
            continue

        e = i + 1
        entry = o[e]
        sl_dist, tp_dist = p.sl_atr * atr[i], p.tp_atr * atr[i]
        uv = value_per_unit(symbol, entry, cfg.contract_size)
        lots = position_size(balance, cfg.risk_pct, sl_dist, uv, cfg.min_lot,
                             cfg.lot_step, cfg.max_lot, cfg.max_risk_overshoot)
        if lots <= 0:
            i += 1
            continue
        sl, tp = entry - s * sl_dist, entry + s * tp_dist
        xi, xp, why = walk_trade(h, l, c, e, s, sl, tp, p.max_bars_in_trade)
        pnl = s * (xp - entry) * lots * uv - spread_price * lots * uv
        risk_money = balance * cfg.risk_pct / 100.0
        balance += pnl
        rows.append({
            "entry_time": times.iloc[e] if times is not None else e,
            "exit_time": times.iloc[xi] if times is not None else xi,
            "dir": "BUY" if s == 1 else "SELL", "lots": lots, "entry": entry,
            "exit": xp, "reason": why, "pnl": pnl,
            "r": pnl / risk_money if risk_money > 0 else 0.0, "balance": balance,
        })
        if why == "END":
            break
        i = max(xi, i + 1)
    return pd.DataFrame(rows)


def summarize(trades, initial):
    if trades.empty:
        return {"trades": 0}
    pnl = trades["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_loss = -losses.sum()
    bal = np.concatenate([[initial], trades["balance"].to_numpy()])
    peak = np.maximum.accumulate(bal)
    max_dd = float(((peak - bal) / peak).max() * 100.0)
    streak = longest = 0
    for v in pnl:
        streak = streak + 1 if v <= 0 else 0
        longest = max(longest, streak)
    return {
        "trades": len(trades),
        "win_rate_%": 100.0 * len(wins) / len(trades),
        "net_profit": float(pnl.sum()),
        "return_%": 100.0 * float(pnl.sum()) / initial,
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf"),
        "avg_R": float(trades["r"].mean()),
        "max_drawdown_%": max_dd,
        "longest_losing_streak": longest,
        "final_balance": float(trades["balance"].iloc[-1]),
    }


def print_summary(title, s):
    print(f"\n=== {title} ===")
    if s["trades"] == 0:
        print("No trades.")
        return
    for k, v in s.items():
        print(f"  {k:<24}{v:>12.2f}" if isinstance(v, float) else f"  {k:<24}{v:>12}")


def main():
    ap = argparse.ArgumentParser(description="Backtest the strategy")
    ap.add_argument("--csv", help="CSV with time,open,high,low,close")
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--synthetic", action="store_true", help="use fake random data")
    ap.add_argument("--ai", action="store_true", help="compare with the AI filter (out-of-sample)")
    ap.add_argument("--balance", type=float, default=1000.0)
    ap.add_argument("--risk", type=float, default=None, help="risk %% per trade")
    ap.add_argument("--spread", type=float, default=1.2, help="spread in pips")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--out", help="save trades to this CSV file")
    a = ap.parse_args()

    cfg = load_config()
    p = cfg.strategy_params()
    if a.synthetic:
        df = make_synthetic()
        print("*** SYNTHETIC DATA: results are meaningless, this only tests the program. ***")
    elif a.csv:
        df = load_csv(a.csv)
    else:
        ap.error("give --csv FILE or --synthetic")

    bt = BTConfig(initial_balance=a.balance, risk_pct=a.risk or cfg.risk_per_trade_pct,
                  spread_pips=a.spread, max_daily_loss_pct=cfg.max_daily_loss_pct,
                  trade_hour_start=cfg.trade_hour_start, trade_hour_end=cfg.trade_hour_end,
                  max_risk_overshoot=cfg.max_risk_overshoot)
    d = generate_signals(df, p)
    print(f"{a.symbol}: {len(d)} bars, {int((d['signal'] != 0).sum())} raw signals")

    if a.ai:
        split = int(len(d) * a.train_frac)
        filt = train_filter(d.iloc[:split].reset_index(drop=True), p,
                            margin=cfg.ai_margin, min_auc=cfg.ai_min_auc)
        base = run_backtest(d, a.symbol, p, bt, start_idx=split)
        print_summary("OUT-OF-SAMPLE: rules only", summarize(base, bt.initial_balance))
        if filt is None:
            print("\nAI filter: not enough signals in the training part to train.")
            trades = base
        else:
            print(f"\nAI filter: {filt.note}")
            if filt.usable:
                allow = filt.predict(d) >= filt.threshold
                ai_tr = run_backtest(d, a.symbol, p, bt, allow=allow, start_idx=split)
                print_summary("OUT-OF-SAMPLE: rules + AI filter", summarize(ai_tr, bt.initial_balance))
                trades = ai_tr
            else:
                print("The AI filter found no reliable edge on this data, so it would NOT be used.")
                trades = base
    else:
        trades = run_backtest(d, a.symbol, p, bt)
        print_summary("FULL PERIOD", summarize(trades, bt.initial_balance))

    if a.out and not trades.empty:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        trades.to_csv(a.out, index=False)
        print(f"\nTrades saved to {a.out}")


if __name__ == "__main__":
    main()
