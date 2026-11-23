"""AI filter: a small logistic-regression model (numpy only) that estimates the
probability that a strategy signal will reach take-profit before stop-loss.

It is trained on the symbol's own history and only used if it shows a real
edge on a held-out (later) part of that history. Otherwise it disables itself.
"""
import numpy as np
import pandas as pd

from trade_sim import walk_trade

FEATURES = ["dist_f", "slope", "trend", "rsi_dir", "adx", "atr_rel",
            "ret5", "ret20", "hour_sin", "hour_cos"]


def build_features(d):
    atr = d["atr"].replace(0.0, np.nan)
    dirn = d["signal"].astype(float)
    f = pd.DataFrame(index=d.index)
    f["dist_f"] = dirn * (d["close"] - d["ema_f"]) / atr
    f["slope"] = dirn * (d["ema_f"] - d["ema_s"]) / atr
    f["trend"] = dirn * (d["ema_s"] - d["ema_t"]) / atr
    f["rsi_dir"] = dirn * (d["rsi"] - 50.0) / 50.0
    f["adx"] = d["adx"] / 50.0
    f["atr_rel"] = d["atr"] / d["atr"].rolling(100, min_periods=20).mean()
    f["ret5"] = dirn * (d["close"] - d["close"].shift(5)) / atr
    f["ret20"] = dirn * (d["close"] - d["close"].shift(20)) / atr
    if "time" in d.columns:
        hour = d["time"].dt.hour.astype(float)
    else:
        hour = pd.Series(0.0, index=d.index)
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    f = f.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0)
    return f[FEATURES]


def label_signals(d, p, idx):
    """For each signal bar, 1 if the simulated trade ends in profit, else 0."""
    o = d["open"].to_numpy(float)
    h = d["high"].to_numpy(float)
    l = d["low"].to_numpy(float)
    c = d["close"].to_numpy(float)
    atr = d["atr"].to_numpy(float)
    sig = d["signal"].to_numpy(int)
    n = len(d)
    kept, y = [], []
    for i in idx:
        if i + 1 >= n:
            continue
        s = sig[i]
        entry = o[i + 1]
        sl = entry - s * p.sl_atr * atr[i]
        tp = entry + s * p.tp_atr * atr[i]
        xi, xp, why = walk_trade(h, l, c, i + 1, s, sl, tp, p.max_bars_in_trade)
        if why == "END":
            continue
        kept.append(i)
        y.append(1.0 if s * (xp - entry) > 0 else 0.0)
    return np.array(kept, dtype=int), np.array(y, dtype=float)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def auc_score(y, s):
    y = np.asarray(y)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


class LogisticFilter:
    def __init__(self):
        self.mu = None
        self.sd = None
        self.w = None
        self.b = 0.0
        self.threshold = 0.5
        self.val_auc = 0.5
        self.n_samples = 0
        self.usable = False
        self.note = ""

    def fit(self, X, y, l2=1.0, lr=0.1, iters=1500):
        n = len(y)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-9] = 1.0
        Z = (X - self.mu) / self.sd
        w = np.zeros(Z.shape[1])
        base = min(max(y.mean(), 1e-3), 1 - 1e-3)
        b = float(np.log(base / (1 - base)))
        for _ in range(iters):
            pr = _sigmoid(Z @ w + b)
            w -= lr * (Z.T @ (pr - y) / n + l2 * w / n)
            b -= lr * float(np.mean(pr - y))
        self.w, self.b = w, b
        return self

    def proba_matrix(self, X):
        return _sigmoid(((X - self.mu) / self.sd) @ self.w + self.b)

    def predict(self, d):
        """Probability for every row of d (only meaningful on signal rows)."""
        return self.proba_matrix(build_features(d).to_numpy(float))


def train_filter(d, p, min_samples=120, val_frac=0.25, margin=0.05, min_auc=0.52):
    """Train on the signal rows of d. Returns a LogisticFilter or None if too few signals."""
    idx = np.where(d["signal"].to_numpy() != 0)[0]
    kept, y = label_signals(d, p, idx)
    if len(kept) < min_samples:
        return None
    X = build_features(d).to_numpy(float)[kept]
    split = int(len(kept) * (1.0 - val_frac))

    check = LogisticFilter().fit(X[:split], y[:split])
    val_auc = auc_score(y[split:], check.proba_matrix(X[split:]))

    final = LogisticFilter().fit(X, y)
    final.val_auc = val_auc
    final.n_samples = len(kept)
    breakeven = p.sl_atr / (p.sl_atr + p.tp_atr)
    final.threshold = breakeven + margin
    final.usable = val_auc >= min_auc
    final.note = (f"samples={len(kept)} win_rate={y.mean():.2f} validation_AUC={val_auc:.3f} "
                  f"threshold={final.threshold:.2f} -> "
                  f"{'USABLE' if final.usable else 'NO EDGE (filter disabled)'}")
    return final
