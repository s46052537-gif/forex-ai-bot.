"""Simple control window for the bot (tkinter ships with Python on Windows)."""
import copy
import queue
import tkinter as tk
from tkinter import ttk, messagebox

from config import load_config, save_config
from bot import TradingBot


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Forex AI Bot  -  DEMO ACCOUNT RECOMMENDED")
        self.geometry("860x600")
        self.cfg = load_config()
        self.bot = None
        self.q = queue.Queue()
        self._build()
        self.after(300, self._tick)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # --------------------------------------------------------------------- UI
    def _build(self):
        top = ttk.LabelFrame(self, text="Settings")
        top.pack(fill="x", padx=10, pady=8)

        ttk.Label(top, text="Symbols (comma separated)").grid(row=0, column=0, padx=6, pady=6, sticky="w")
        self.e_symbols = ttk.Entry(top, width=34)
        self.e_symbols.insert(0, ",".join(self.cfg.symbols))
        self.e_symbols.grid(row=0, column=1, padx=6)

        ttk.Label(top, text="Timeframe").grid(row=0, column=2, padx=6)
        self.c_tf = ttk.Combobox(top, values=["M15", "M30", "H1", "H4"], width=6, state="readonly")
        self.c_tf.set(self.cfg.timeframe)
        self.c_tf.grid(row=0, column=3)

        ttk.Label(top, text="Risk % per trade").grid(row=1, column=0, padx=6, pady=6, sticky="w")
        self.e_risk = ttk.Entry(top, width=8)
        self.e_risk.insert(0, str(self.cfg.risk_per_trade_pct))
        self.e_risk.grid(row=1, column=1, padx=6, sticky="w")

        self.v_ai = tk.BooleanVar(value=self.cfg.use_ai)
        ttk.Checkbutton(top, text="Use AI filter", variable=self.v_ai).grid(row=1, column=2, columnspan=2)

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=10)
        self.b_start = ttk.Button(btns, text="Start", command=self.start)
        self.b_start.pack(side="left", padx=4)
        self.b_stop = ttk.Button(btns, text="Stop", command=self.stop, state="disabled")
        self.b_stop.pack(side="left", padx=4)
        ttk.Button(btns, text="Close all bot trades", command=self.close_all).pack(side="left", padx=4)
        ttk.Button(btns, text="Save settings", command=self.save).pack(side="left", padx=4)

        st = ttk.LabelFrame(self, text="Status")
        st.pack(fill="x", padx=10, pady=8)
        self.lbl = {}
        for r, (key, title) in enumerate([("state", "State"), ("account", "Account"),
                                          ("balance", "Balance"), ("equity", "Equity"),
                                          ("day_pnl", "Today P/L"), ("drawdown_pct", "Drawdown from peak"),
                                          ("positions", "Open trades")]):
            ttk.Label(st, text=title + ":").grid(row=r // 2, column=(r % 2) * 2, padx=8, pady=3, sticky="e")
            self.lbl[key] = ttk.Label(st, text="-", width=34)
            self.lbl[key].grid(row=r // 2, column=(r % 2) * 2 + 1, sticky="w")

        box = ttk.LabelFrame(self, text="Log")
        box.pack(fill="both", expand=True, padx=10, pady=8)
        self.txt = tk.Text(box, height=14, state="disabled", wrap="word")
        sb = ttk.Scrollbar(box, command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        self.txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    # ---------------------------------------------------------------- actions
    def _read_settings(self):
        cfg = copy.deepcopy(self.cfg)
        syms = [s.strip() for s in self.e_symbols.get().split(",") if s.strip()]
        if not syms:
            raise ValueError("Enter at least one symbol.")
        cfg.symbols = syms
        cfg.timeframe = self.c_tf.get()
        cfg.risk_per_trade_pct = float(self.e_risk.get())
        if not 0.05 <= cfg.risk_per_trade_pct <= 2.0:
            raise ValueError("Risk % must be between 0.05 and 2.0.")
        cfg.use_ai = self.v_ai.get()
        return cfg

    def start(self):
        try:
            cfg = self._read_settings()
        except ValueError as e:
            messagebox.showerror("Settings", str(e))
            return
        self.bot = TradingBot(cfg, log=self.q.put)
        self.bot.start()
        self.b_start.configure(state="disabled")
        self.b_stop.configure(state="normal")

    def stop(self):
        if self.bot:
            self.bot.stop()
            self.q.put("Stopping...")

    def close_all(self):
        if not self.bot or not self.bot.status.get("connected"):
            messagebox.showinfo("Close trades", "The bot is not connected.")
            return
        if messagebox.askyesno("Close trades", "Close every trade opened by this bot now?"):
            try:
                self.bot.close_all()
            except Exception as e:
                messagebox.showerror("Close trades", str(e))

    def save(self):
        try:
            self.cfg = self._read_settings()
            save_config(self.cfg)
            messagebox.showinfo("Saved", "Settings saved to config.json")
        except ValueError as e:
            messagebox.showerror("Settings", str(e))

    # ---------------------------------------------------------------- refresh
    def _tick(self):
        while True:
            try:
                line = self.q.get_nowait()
            except queue.Empty:
                break
            self.txt.configure(state="normal")
            self.txt.insert("end", line + "\n")
            self.txt.see("end")
            self.txt.configure(state="disabled")
        if self.bot:
            s = self.bot.status
            self.lbl["state"].configure(text="RUNNING" if s["running"] else "STOPPED")
            self.lbl["account"].configure(text=s["account"] or "-")
            self.lbl["balance"].configure(text=f"{s['balance']:.2f}")
            self.lbl["equity"].configure(text=f"{s['equity']:.2f}")
            self.lbl["day_pnl"].configure(text=f"{s['day_pnl']:+.2f}")
            self.lbl["drawdown_pct"].configure(text=f"{s['drawdown_pct']:.2f}%")
            self.lbl["positions"].configure(text=str(s["positions"]))
            if not s["running"]:
                self.b_start.configure(state="normal")
                self.b_stop.configure(state="disabled")
        self.after(500, self._tick)

    def _on_close(self):
        if self.bot:
            self.bot.stop()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
