"""Risk management helpers (position sizing and safety limits)."""
import math


def round_down_step(x, step):
    if step <= 0:
        return x
    return math.floor(x / step + 1e-9) * step


def position_size(equity, risk_pct, sl_dist, unit_value, vol_min=0.01, vol_step=0.01,
                  vol_max=100.0, max_overshoot=2.0):
    """Lots so that hitting the stop loses about `risk_pct` of equity.

    sl_dist     : stop distance in PRICE units (e.g. 0.0015)
    unit_value  : account-currency profit per 1.0 price unit per 1 lot
    Returns 0.0 if even the minimum lot would risk too much.
    """
    if equity <= 0 or sl_dist <= 0 or unit_value <= 0:
        return 0.0
    risk_money = equity * risk_pct / 100.0
    lots = round_down_step(risk_money / (sl_dist * unit_value), vol_step)
    if lots < vol_min:
        if vol_min * sl_dist * unit_value <= risk_money * max_overshoot:
            lots = vol_min
        else:
            return 0.0
    lots = min(lots, vol_max)
    decimals = max(0, int(round(-math.log10(vol_step)))) if vol_step < 1 else 0
    return round(lots, decimals)


def value_per_unit(symbol, price, contract_size=100000.0):
    """Approximate account-currency (USD) value of 1.0 price move for 1 lot.

    Used by the backtester only. Live trading uses the broker's tick value.
    """
    s = symbol.upper()
    if s.endswith("JPY") or s.startswith("USD"):
        return contract_size / price
    return contract_size


def daily_loss_hit(equity, day_start_equity, limit_pct):
    return equity <= day_start_equity * (1.0 - limit_pct / 100.0)


def drawdown_hit(equity, peak_equity, limit_pct):
    return equity <= peak_equity * (1.0 - limit_pct / 100.0)
