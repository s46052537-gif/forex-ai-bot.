"""Download price history from MetaTrader 5 into CSV files for backtesting.

    py export_history.py                       (uses symbols/timeframe from config.json)
    py export_history.py --bars 50000
"""
import argparse
import os

from config import load_config, DATA_DIR
from mt5_connector import MT5Connector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=50000)
    a = ap.parse_args()
    cfg = load_config()
    mt = MT5Connector()
    mt.connect(cfg.login, cfg.password, cfg.server, cfg.terminal_path)
    os.makedirs(DATA_DIR, exist_ok=True)
    for sym in cfg.symbols:
        try:
            mt.symbol_info(sym)
            df = mt.rates(sym, cfg.timeframe, a.bars)
            path = os.path.join(DATA_DIR, f"{sym}_{cfg.timeframe}.csv")
            df.to_csv(path, index=False)
            print(f"{sym}: saved {len(df)} bars -> {path}")
        except Exception as e:
            print(f"{sym}: FAILED - {e}")
    mt.shutdown()


if __name__ == "__main__":
    main()
