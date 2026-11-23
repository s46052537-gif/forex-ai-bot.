"""The trading loop: reads prices, decides, enforces risk limits, sends orders."""
import csv
import datetime as dt
import json
import os
import threading

from config import LOG_DIR, TF_SECONDS
from strategy import generate_signals
from risk import position_size, daily_loss_hit, drawdown_hit
from ai_filter import train_filter

STATE_PATH = os.path.join(LOG_DIR, "state.json")
TRADES_CSV = os.path.join(LOG_DIR, "trades.csv")


class TradingBot:
    def __init__(self, cfg, log=None, connector=None):
        self.cfg = cfg
        self.p = cfg.strategy_params()
        self._ext_log = log
        self.mt = connector
        self._stop = threading.Event()
        self._thread = None
        self.filters = {}
        self.last_bar = {}
        self.day = None
        self.day_start_equity = None
        self.peak_equity = None
        self.trades_today = 0
        self._halt_logged = False
        self.status = {"running": False, "connected": False, "account": "", "balance": 0.0,
                       "equity": 0.0, "day_pnl": 0.0, "drawdown_pct": 0.0,
                       "positions": 0, "last_event": ""}
        os.makedirs(LOG_DIR, exist_ok=True)

    # ------------------------------------------------------------------ utils
    def log(self, msg):
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        try:
            with open(os.path.join(LOG_DIR, "bot.log"), "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
        (self._ext_log or print)(line)
        self.status["last_event"] = msg

    def _load_state(self, login):
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                s = json.load(f)
            if s.get("login") == login:
                return float(s.get("peak_equity", 0.0)) or None
        except (OSError, ValueError):
            pass
        return None

    def _save_state(self, login):
        try:
            with open(STATE_PATH, "w", encoding="utf-8") as f:
                json.dump({"login": login, "peak_equity": self.peak_equity}, f)
        except OSError:
            pass

    # -------------------------------------------------------------- lifecycle
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def run(self):
        self.status["running"] = True
        try:
            if self.mt is None:
                from mt5_connector import MT5Connector
                self.mt = MT5Connector()
            self._connect()
            self._prepare()
            errors = 0
            while not self._stop.is_set():
                try:
                    self._cycle()
                    errors = 0
                except Exception as e:  # keep running through temporary problems
                    errors += 1
                    self.log(f"Error ({errors}/10): {e}")
                    if errors >= 10:
                        self.log("Too many consecutive errors. Stopping for safety.")
                        break
                self._stop.wait(self.cfg.poll_seconds)
        except Exception as e:
            self.log(f"FATAL: {e}")
        finally:
            self.status["running"] = False
            self.status["connected"] = False
            try:
                self.mt.shutdown()
            except Exception:
                pass
            self.log("Bot stopped.")

    def _connect(self):
        c = self.cfg
        self.mt.connect(c.login, c.password, c.server, c.terminal_path)
        acc = self.mt.account()
        if not acc["demo"] and not c.allow_live:
            raise RuntimeError(
                "This is a REAL-MONEY account and allow_live is false in config.json. "
                "Refusing to trade. Use a demo account.")
        kind = "DEMO" if acc["demo"] else "REAL"
        self.status["connected"] = True
        self.status["account"] = f"{acc['login']} @ {acc['server']} ({kind})"
        self.log(f"Connected: {self.status['account']}  balance={acc['balance']:.2f} {acc['currency']}")
        saved_peak = self._load_state(acc["login"])
        self.peak_equity = max(acc["equity"], saved_peak or 0.0)

    def _prepare(self):
        for sym in self.cfg.symbols:
            self.mt.symbol_info(sym)  # raises a clear error if the name is wrong
        if not self.cfg.use_ai:
            return
        for sym in self.cfg.symbols:
            try:
                df = self.mt.rates(sym, self.cfg.timeframe, self.cfg.ai_train_bars)
                d = generate_signals(df.iloc[:-1].reset_index(drop=True), self.p)
                filt = train_filter(d, self.p, margin=self.cfg.ai_margin, min_auc=self.cfg.ai_min_auc)
            except Exception as e:
                self.log(f"[{sym}] AI training failed: {e}")
                filt = None
            if filt is None:
                self.log(f"[{sym}] AI filter: not enough history, running on rules only.")
            else:
                self.log(f"[{sym}] AI filter: {filt.note}")
            self.filters[sym] = filt

    # ------------------------------------------------------------------ cycle
    def _cycle(self):
        c = self.cfg
        acc = self.mt.account()
        eq = acc["equity"]
        today = dt.date.today()
        if self.day != today:
            self.day, self.day_start_equity = today, eq
            self.trades_today = 0
            self._halt_logged = False
        if eq > self.peak_equity:
            self.peak_equity = eq
            self._save_state(acc["login"])

        positions = self.mt.positions(magic=c.magic)
        self.status.update(balance=acc["balance"], equity=eq,
                           day_pnl=eq - self.day_start_equity,
                           drawdown_pct=100.0 * (1 - eq / self.peak_equity) if self.peak_equity else 0.0,
                           positions=len(positions))

        if drawdown_hit(eq, self.peak_equity, c.max_drawdown_pct):
            self.log(f"KILL SWITCH: drawdown reached {c.max_drawdown_pct}%. Bot stopped. "
                     "Delete logs/state.json to reset after you review what happened.")
            self._stop.set()
            return

        self._manage_time_exits(positions)
        halted = daily_loss_hit(eq, self.day_start_equity, c.max_daily_loss_pct)
        if halted and not self._halt_logged:
            self.log(f"Daily loss limit {c.max_daily_loss_pct}% reached: no new trades today.")
            self._halt_logged = True

        for sym in c.symbols:
            if self._stop.is_set():
                break
            self._process_symbol(sym, eq, positions, halted)

    def _manage_time_exits(self, positions):
        limit = self.p.max_bars_in_trade * TF_SECONDS[self.cfg.timeframe]
        for pos in positions:
            now = self.mt.tick(pos["symbol"])["time"]
            if (now - pos["time"]).total_seconds() > limit:
                res = self.mt.close_position(pos, self.cfg.deviation)
                self.log(f"[{pos['symbol']}] time exit -> {'closed' if res['ok'] else res['comment']}")

    def _process_symbol(self, sym, eq, positions, halted):
        c = self.cfg
        df = self.mt.rates(sym, c.timeframe, c.history_bars)
        df = df.iloc[:-1].reset_index(drop=True)          # drop the still-forming bar
        if len(df) < self.p.warmup + 5:
            return
        bar_time = df["time"].iloc[-1]
        if sym not in self.last_bar:                       # first look: wait for a fresh bar
            self.last_bar[sym] = bar_time
            self.log(f"[{sym}] watching. Waiting for the next closed {c.timeframe} bar.")
            return
        if self.last_bar[sym] == bar_time:
            return
        self.last_bar[sym] = bar_time

        d = generate_signals(df, self.p)
        sig = int(d["signal"].iloc[-1])
        if sig == 0:
            return
        side = "BUY" if sig == 1 else "SELL"
        self.log(f"[{sym}] signal {side} on bar {bar_time}")

        mine = [x for x in positions if x["symbol"] == sym]
        if halted:
            return self.log(f"[{sym}] skipped: daily loss limit active")
        if mine:
            return self.log(f"[{sym}] skipped: already has an open position")
        if len(positions) >= c.max_open_positions:
            return self.log(f"[{sym}] skipped: max open positions ({c.max_open_positions})")
        if self.trades_today >= c.max_trades_per_day:
            return self.log(f"[{sym}] skipped: max trades per day ({c.max_trades_per_day})")

        tick = self.mt.tick(sym)
        if not (c.trade_hour_start <= tick["time"].hour < c.trade_hour_end):
            return self.log(f"[{sym}] skipped: outside trading hours")

        info = self.mt.symbol_info(sym)
        atr = float(d["atr"].iloc[-1])
        spread_price = info["spread"] * info["point"]
        if spread_price > c.max_spread_atr_frac * atr:
            return self.log(f"[{sym}] skipped: spread too wide")

        prob_txt = ""
        filt = self.filters.get(sym)
        if c.use_ai and filt is not None and filt.usable:
            prob = float(filt.predict(d)[-1])
            if prob < filt.threshold:
                return self.log(f"[{sym}] skipped by AI filter (p={prob:.2f} < {filt.threshold:.2f})")
            prob_txt = f"{prob:.2f}"

        sl_dist, tp_dist = self.p.sl_atr * atr, self.p.tp_atr * atr
        min_dist = (info["stops_level"] + info["spread"]) * info["point"]
        if sl_dist < min_dist or tp_dist < min_dist:
            return self.log(f"[{sym}] skipped: stop distance below broker minimum")

        price = tick["ask"] if sig == 1 else tick["bid"]
        sl, tp = price - sig * sl_dist, price + sig * tp_dist
        unit_value = info["tick_value"] / info["tick_size"]
        lots = position_size(eq, c.risk_per_trade_pct, sl_dist, unit_value, info["vol_min"],
                             info["vol_step"], info["vol_max"], c.max_risk_overshoot)
        if lots <= 0:
            return self.log(f"[{sym}] skipped: smallest lot would risk too much for this account")

        res = self.mt.send_market(sym, sig, lots, sl, tp, c.magic, c.deviation)
        if res["ok"]:
            self.trades_today += 1
            self.log(f"[{sym}] {side} {lots} lots @ {price:.5f}  SL={sl:.5f} TP={tp:.5f}")
            self._record_trade(sym, side, lots, price, sl, tp, prob_txt)
        else:
            self.log(f"[{sym}] order rejected: {res['comment']} (code {res['retcode']})")

    def _record_trade(self, sym, side, lots, price, sl, tp, prob):
        new = not os.path.exists(TRADES_CSV)
        with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "symbol", "side", "lots", "price", "sl", "tp", "ai_prob"])
            w.writerow([dt.datetime.now().isoformat(timespec="seconds"), sym, side, lots,
                        price, sl, tp, prob])

    def close_all(self):
        """Close every position opened by this bot (used by the GUI button)."""
        n = 0
        for pos in self.mt.positions(magic=self.cfg.magic):
            res = self.mt.close_position(pos, self.cfg.deviation)
            n += 1 if res["ok"] else 0
        self.log(f"Closed {n} position(s).")
        return n
