"""Shared trade simulation used by the backtester and by the AI labelling step."""


def walk_trade(high, low, close, start, direction, sl, tp, max_bars):
    """Walk forward from bar `start` until SL / TP / time limit / end of data.

    Returns (exit_index, exit_price, reason).  If SL and TP are both touched in
    the same bar we assume the SL was hit first (conservative).
    """
    n = len(high)
    if start >= n:
        return n - 1, close[n - 1], "END"
    end = min(n, start + max_bars)
    for j in range(start, end):
        if direction == 1:
            hit_sl = low[j] <= sl
            hit_tp = high[j] >= tp
        else:
            hit_sl = high[j] >= sl
            hit_tp = low[j] <= tp
        if hit_sl:
            return j, sl, "SL"
        if hit_tp:
            return j, tp, "TP"
    last = end - 1
    if end - start < max_bars:
        return last, close[last], "END"
    return last, close[last], "TIME"
