"""Settings for the Forex AI bot.

Edit config.json (created automatically) instead of this file.
"""
import json
import os
import sys
from dataclasses import dataclass, field, asdict, fields

# When packaged as an .exe, keep config/logs/data next to the .exe file.
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

TF_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}


@dataclass
class Config:
    # ---- Account (leave empty to use the account already logged in inside MT5)
    login: int = 0
    password: str = ""
    server: str = ""
    terminal_path: str = ""
    # SAFETY: the bot refuses to trade on a REAL account unless this is True.
    allow_live: bool = False

    # ---- Market
    symbols: list = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY"])
    timeframe: str = "H1"
    history_bars: int = 600

    # ---- Strategy (trend following: EMA cross / pullback + ADX filter)
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
    mode: str = "both"            # "cross", "pullback" or "both"
    max_bars_in_trade: int = 48

    # ---- Risk management
    risk_per_trade_pct: float = 0.5      # % of equity risked per trade
    max_daily_loss_pct: float = 3.0      # stop opening trades for the day
    max_drawdown_pct: float = 10.0       # kill switch: bot stops completely
    max_open_positions: int = 3
    max_trades_per_day: int = 4
    max_spread_atr_frac: float = 0.15    # skip if spread > 15% of ATR
    max_risk_overshoot: float = 2.0      # allow min lot if its risk <= 2x target
    trade_hour_start: int = 7            # broker server hour (inclusive)
    trade_hour_end: int = 20             # broker server hour (exclusive)

    # ---- AI filter (small logistic-regression model trained on history)
    use_ai: bool = True
    ai_train_bars: int = 20000
    ai_margin: float = 0.05
    ai_min_auc: float = 0.52

    # ---- Execution
    magic: int = 26112301
    deviation: int = 20
    poll_seconds: int = 15

    def strategy_params(self):
        from strategy import StrategyParams
        return StrategyParams(
            ema_fast=self.ema_fast, ema_slow=self.ema_slow, ema_trend=self.ema_trend,
            rsi_period=self.rsi_period, adx_period=self.adx_period,
            atr_period=self.atr_period, adx_min=self.adx_min,
            rsi_long_max=self.rsi_long_max, rsi_short_min=self.rsi_short_min,
            sl_atr=self.sl_atr, tp_atr=self.tp_atr, mode=self.mode,
            max_bars_in_trade=self.max_bars_in_trade,
        )


def load_config(path=CONFIG_PATH):
    cfg = Config()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        names = {f.name for f in fields(Config)}
        for key, value in raw.items():
            if key in names:
                setattr(cfg, key, value)
    return cfg


def save_config(cfg, path=CONFIG_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
