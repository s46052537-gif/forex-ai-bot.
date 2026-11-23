"""Self-checks. Run:  py tests\\test_all.py   (no MetaTrader needed)"""
import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config                                   # noqa: E402
from strategy import generate_signals                        # noqa: E402
from risk import position_size                               # noqa: E402
from trade_sim import walk_trade                             # noqa: E402
from backtest import make_synthetic, run_backtest, BTConfig, summarize  # noqa: E402
from ai_filter import train_filter                           # noqa: E402
from bot import TradingBot                                   # noqa: E402


class FakeConnector:
    """Pretends to be MetaTrader 5, replaying synthetic prices."""

    def __init__(self, df, demo=True, equity=1000.0):
        self.df, self.ptr, self.demo, self.equity = df, 3000, demo, equity
        self.orders, self.pos = [], []

    def connect(self, *a, **k): pass
    def shutdown(self): pass

    def account(self):
        return {"login": 1, "server": "Fake-Demo", "balance": self.equity, "equity": self.equity,
                "currency": "USD", "leverage": 100, "demo": self.demo}

    def symbol_info(self, sym):
        return {"point": 0.00001, "digits": 5, "tick_size": 0.00001, "tick_value": 1.0,
                "contract_size": 100000, "vol_min": 0.01, "vol_step": 0.01, "vol_max": 100,
                "spread": 5, "stops_level": 0, "filling": 1}

    def tick(self, sym):
        row = self.df.iloc[self.ptr]
        return {"bid": row.close, "ask": row.close + 0.00005, "time": row.time.to_pydatetime()}

    def rates(self, sym, tf, count):
        return self.df.iloc[max(0, self.ptr + 1 - count): self.ptr + 1].reset_index(drop=True)

    def positions(self, magic=None):
        return list(self.pos)

    def send_market(self, sym, direction, lots, sl, tp, magic, deviation, comment=""):
        t = self.tick(sym)
        self.orders.append(dict(sym=sym, dir=direction, lots=lots, sl=sl, tp=tp,
                                price=t["ask"] if direction == 1 else t["bid"]))
        self.pos.append({"ticket": len(self.orders), "symbol": sym, "direction": direction,
                         "volume": lots, "time": t["time"], "magic": magic})
        return {"ok": True, "retcode": 10009, "comment": "ok", "ticket": len(self.orders)}

    def close_position(self, pos, deviation=20):
        self.pos = [p for p in self.pos if p["ticket"] != pos["ticket"]]
        return {"ok": True, "retcode": 10009, "comment": "ok", "ticket": 0}


def make_cfg(**kw):
    cfg = Config(symbols=["EURUSD"], use_ai=False, trade_hour_start=0, trade_hour_end=24)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_position_size():
    # risk 5 on 1000 equity, stop 0.0010, 100000 per unit -> 0.05 lots
    assert position_size(1000, 0.5, 0.0010, 100000) == 0.05
    assert position_size(1000, 0.5, 0.0500, 100000) == 0.0    # min lot would risk 50 vs 5 target
    assert position_size(100, 0.5, 0.0100, 100000, max_overshoot=2.0) == 0.0   # min lot too risky
    assert position_size(0, 0.5, 0.001, 100000) == 0.0
    print("PASS position_size")


def test_walk_trade():
    h = np.array([1.0, 1.2, 1.1]); l = np.array([0.9, 1.0, 0.8]); c = np.array([1.0, 1.1, 0.9])
    assert walk_trade(h, l, c, 0, 1, 0.85, 1.15, 10)[2] == "TP"      # bar 1 high 1.2 >= 1.15
    assert walk_trade(h, l, c, 0, -1, 1.15, 0.85, 10)[2] == "SL"     # bar 1 high 1.2 >= 1.15 first
    assert walk_trade(h, l, c, 0, 1, 0.5, 2.0, 2)[2] == "TIME"
    print("PASS walk_trade")


def test_no_lookahead():
    p = Config().strategy_params()
    df = make_synthetic(6000)
    full = generate_signals(df, p)["signal"].to_numpy()
    part = generate_signals(df.iloc[:4000].reset_index(drop=True), p)["signal"].to_numpy()
    assert (full[:4000] == part).all(), "signals changed when future data was removed (look-ahead!)"
    print("PASS no look-ahead in signals")


def test_backtest_and_ai():
    cfg = Config()
    p = cfg.strategy_params()
    d = generate_signals(make_synthetic(20000), p)
    bt = BTConfig()
    tr = run_backtest(d, "EURUSD", p, bt)
    s = summarize(tr, bt.initial_balance)
    assert s["trades"] > 50 and 0 < s["win_rate_%"] < 100
    assert abs(tr["balance"].iloc[-1] - (bt.initial_balance + tr["pnl"].sum())) < 1e-6
    filt = train_filter(d.iloc[:10000].reset_index(drop=True), p)
    assert filt is not None and 0.0 <= filt.val_auc <= 1.0
    print("PASS backtest + AI filter  (", filt.note, ")")


def test_bot_trades_and_sizes_risk():
    cfg = make_cfg()
    fake = FakeConnector(make_synthetic(20000))
    bot = TradingBot(cfg, log=lambda m: None, connector=fake)
    bot._connect()
    bot._prepare()
    for _ in range(4000):
        bot._cycle()
        if fake.orders:
            break
        fake.ptr += 1
    assert fake.orders, "bot never placed an order"
    o = fake.orders[0]
    assert (o["sl"] < o["price"]) == (o["dir"] == 1) and (o["tp"] > o["price"]) == (o["dir"] == 1)
    risk = o["lots"] * abs(o["price"] - o["sl"]) * 100000 / fake.equity * 100
    assert 0.2 <= risk <= 0.6, f"risk per trade was {risk:.2f}%"
    # time exit: after 60 bars the open position must be closed by the bot
    fake.ptr += 60
    bot._cycle()
    assert not fake.pos, "time exit did not close the position"
    print(f"PASS bot order + risk sizing ({risk:.2f}% risked) + time exit")


def test_refuses_real_account():
    cfg = make_cfg()
    fake = FakeConnector(make_synthetic(5000), demo=False)
    msgs = []
    bot = TradingBot(cfg, log=msgs.append, connector=fake)
    bot.run()
    assert fake.orders == [] and any("REAL-MONEY" in m for m in msgs)
    print("PASS refuses real-money account")


def test_kill_switch():
    cfg = make_cfg(max_drawdown_pct=10.0)
    fake = FakeConnector(make_synthetic(5000))
    bot = TradingBot(cfg, log=lambda m: None, connector=fake)
    bot._connect()
    bot._cycle()
    fake.equity = 850.0     # -15% from peak
    bot._cycle()
    assert bot._stop.is_set()
    print("PASS drawdown kill switch")


if __name__ == "__main__":
    for t in (test_position_size, test_walk_trade, test_no_lookahead, test_backtest_and_ai,
              test_bot_trades_and_sizes_risk, test_refuses_real_account, test_kill_switch):
        t()
    print("\nALL TESTS PASSED")
