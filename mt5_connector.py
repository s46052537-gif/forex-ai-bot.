"""Thin wrapper around the MetaTrader5 Python package (Windows only)."""
import datetime as dt

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:  # the package only exists on Windows
    mt5 = None


class MT5Error(RuntimeError):
    pass


def _naive_utc(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).replace(tzinfo=None)


class MT5Connector:
    def connect(self, login=0, password="", server="", path=""):
        if mt5 is None:
            raise MT5Error("The MetaTrader5 package is not installed. Run 1_install.bat "
                           "(Windows only, and MetaTrader 5 must be installed).")
        kwargs = {}
        if path:
            kwargs["path"] = path
        if login:
            kwargs.update(login=int(login), password=password, server=server)
        if not mt5.initialize(**kwargs):
            raise MT5Error(f"Could not connect to MetaTrader 5: {mt5.last_error()}. "
                           "Open MT5 and log in to your demo account first.")

    def shutdown(self):
        if mt5 is not None:
            mt5.shutdown()

    def account(self):
        a = mt5.account_info()
        if a is None:
            raise MT5Error(f"account_info failed: {mt5.last_error()}")
        return {"login": a.login, "server": a.server, "balance": a.balance,
                "equity": a.equity, "currency": a.currency, "leverage": a.leverage,
                "demo": a.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO}

    def symbol_info(self, symbol):
        si = mt5.symbol_info(symbol)
        if si is None:
            raise MT5Error(f"Symbol {symbol} not found. Check its exact name in MT5 "
                           "Market Watch (some brokers add a suffix, e.g. EURUSDm).")
        if not si.visible:
            mt5.symbol_select(symbol, True)
        return {"point": si.point, "digits": si.digits, "tick_size": si.trade_tick_size,
                "tick_value": si.trade_tick_value, "contract_size": si.trade_contract_size,
                "vol_min": si.volume_min, "vol_step": si.volume_step, "vol_max": si.volume_max,
                "spread": si.spread, "stops_level": si.trade_stops_level,
                "filling": si.filling_mode}

    def tick(self, symbol):
        t = mt5.symbol_info_tick(symbol)
        if t is None:
            raise MT5Error(f"No tick for {symbol}: {mt5.last_error()}")
        return {"bid": t.bid, "ask": t.ask, "time": _naive_utc(t.time)}

    def rates(self, symbol, timeframe, count):
        tf = getattr(mt5, "TIMEFRAME_" + timeframe)
        arr = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if arr is None or len(arr) == 0:
            raise MT5Error(f"No price history for {symbol}: {mt5.last_error()}")
        df = pd.DataFrame(arr)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df[["time", "open", "high", "low", "close", "tick_volume", "spread"]]

    def positions(self, magic=None):
        res = mt5.positions_get()
        out = []
        for p in res or []:
            if magic is not None and p.magic != magic:
                continue
            out.append({"ticket": p.ticket, "symbol": p.symbol,
                        "direction": 1 if p.type == mt5.POSITION_TYPE_BUY else -1,
                        "volume": p.volume, "price_open": p.price_open, "sl": p.sl,
                        "tp": p.tp, "profit": p.profit, "time": _naive_utc(p.time),
                        "magic": p.magic})
        return out

    def _fill_modes(self, flags):
        modes = []
        if flags & 2:
            modes.append(mt5.ORDER_FILLING_IOC)
        if flags & 1:
            modes.append(mt5.ORDER_FILLING_FOK)
        modes.append(mt5.ORDER_FILLING_RETURN)
        return modes

    def _send(self, request, flags):
        last = None
        for mode in self._fill_modes(flags):
            request["type_filling"] = mode
            res = mt5.order_send(request)
            if res is None:
                last = {"ok": False, "retcode": -1, "comment": str(mt5.last_error()), "ticket": 0}
                continue
            last = {"ok": res.retcode == mt5.TRADE_RETCODE_DONE, "retcode": res.retcode,
                    "comment": res.comment, "ticket": res.order,
                    "price": getattr(res, "price", 0.0)}
            if res.retcode != 10030:  # 10030 = unsupported filling mode -> try the next
                break
        return last

    def send_market(self, symbol, direction, lots, sl, tp, magic, deviation, comment="AI-bot"):
        info = self.symbol_info(symbol)
        t = self.tick(symbol)
        digits = info["digits"]
        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(lots),
               "type": mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL,
               "price": t["ask"] if direction == 1 else t["bid"],
               "sl": round(sl, digits), "tp": round(tp, digits), "deviation": int(deviation),
               "magic": int(magic), "comment": comment, "type_time": mt5.ORDER_TIME_GTC}
        return self._send(req, info["filling"])

    def close_position(self, pos, deviation=20):
        info = self.symbol_info(pos["symbol"])
        t = self.tick(pos["symbol"])
        closing_buy = pos["direction"] == -1
        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": pos["symbol"],
               "volume": float(pos["volume"]), "position": int(pos["ticket"]),
               "type": mt5.ORDER_TYPE_BUY if closing_buy else mt5.ORDER_TYPE_SELL,
               "price": t["ask"] if closing_buy else t["bid"], "deviation": int(deviation),
               "magic": int(pos["magic"]), "comment": "AI-bot close",
               "type_time": mt5.ORDER_TIME_GTC}
        return self._send(req, info["filling"])
