# ── Set your Alpaca PAPER keys here (or as environment variables) ───────────
# Either export ALPACA_API_KEY / ALPACA_SECRET_KEY in your terminal,
# or replace the two os.environ.get(...) lines below with your keys directly:
#     API_KEY = "PK..."
#     SECRET_KEY = "..."
# PAPER TRADING ONLY — the trading client is hard-coded to paper=True.
# ────────────────────────────────────────────────────────────────────────────

import os
import queue
import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import ttk

API_KEY = "PKN6FYCWZJP3SBCFBJRJTXX2HW"
SECRET_KEY = "HxgLAUTkmirjvpJxbKYLFjqmRQ4GeiwFUDXiCdBUoxvy"

INITIAL = 100_000
TRADING_DAYS = 252
PROB_THRESHOLD = 0.40     # long if P(up) > 0.60, else flat
TRAIN_FRAC = 0.70         # time-ordered train/test split
TARGET_NOTIONAL = 10_000  # paper order size in dollars


# ============================================================================
#  DATA
# ============================================================================
class DataConnector:
    def __init__(self, api_key, secret_key):
        if not api_key or not secret_key:
            raise RuntimeError("Missing API keys. Set ALPACA_API_KEY and "
                               "ALPACA_SECRET_KEY (or paste them in the code).")
        self.data = StockHistoricalDataClient(api_key, secret_key)
        self.trading = TradingClient(api_key, secret_key, paper=True)  # PAPER ONLY

    def get_daily(self, symbol, years=5):
        end = datetime.now(timezone.utc) - timedelta(minutes=20)
        start = end - timedelta(days=int(years * 365.25) + 10)
        req = StockBarsRequest(symbol_or_symbols=symbol.upper(),
                               timeframe=TimeFrame(1, TimeFrameUnit.Day),
                               start=start, end=end,
                               feed=DataFeed.IEX, adjustment=Adjustment.ALL)
        df = self.data.get_stock_bars(req).df
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]


# ============================================================================
#  INDICATORS / FEATURES
# ============================================================================
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def true_range(h, l, c):
    p = c.shift(1)
    return pd.concat([h - l, (h - p).abs(), (l - p).abs()], axis=1).max(axis=1)


def adx(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    tr = true_range(h, l, c).ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / tr
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def stochastic_k(h, l, c, n=14):
    ll, hh = l.rolling(n).min(), h.rolling(n).max()
    return 100 * (c - ll) / (hh - ll).replace(0, np.nan)


def williams_r(h, l, c, n=14):
    hh, ll = h.rolling(n).max(), l.rolling(n).min()
    return -100 * (hh - c) / (hh - ll).replace(0, np.nan)


def cmf(h, l, c, v, n=20):
    mult = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    mfv = mult * v
    return mfv.rolling(n).sum() / v.rolling(n).sum().replace(0, np.nan)


def build_features(df):
    """Return a scale-friendly, mostly-stationary feature matrix + target."""
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    f = pd.DataFrame(index=df.index)

    # trend
    f["sma_ratio"] = c / c.rolling(20).mean() - 1
    f["ema_ratio"] = c / ema(c, 20) - 1
    macd = ema(c, 12) - ema(c, 26)
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    f["adx"] = adx(h, l, c)
    # momentum
    f["rsi"] = rsi(c)
    f["stoch_k"] = stochastic_k(h, l, c)
    f["williams_r"] = williams_r(h, l, c)
    # volatility
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    f["bb_pctb"] = (c - (mid - 2 * sd)) / (4 * sd).replace(0, np.nan)
    f["atr_pct"] = true_range(h, l, c).ewm(alpha=1 / 14, adjust=False).mean() / c
    # volume
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f["obv_chg"] = obv.diff() / v.rolling(20).mean()
    f["cmf"] = cmf(h, l, c, v)
    # returns & rolling stats
    f["log_ret"] = np.log(c / c.shift(1))
    f["roll_mean"] = f["log_ret"].rolling(10).mean()
    f["roll_std"] = f["log_ret"].rolling(10).std()

    target = (c.shift(-1) > c).astype(int)   # next-day up?
    data = f.copy()
    data["target"] = target
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    return data.drop(columns="target"), data["target"]


# ============================================================================
#  ML PIPELINE
# ============================================================================
def train_pipeline(X, y):
    n = len(X)
    split = int(n * TRAIN_FRAC)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler().fit(Xtr)
    pca = PCA(n_components=0.80).fit(scaler.transform(Xtr))   # keep >=80% variance
    Ztr = pca.transform(scaler.transform(Xtr))
    Zte = pca.transform(scaler.transform(Xte))

    model = RandomForestClassifier(n_estimators=300, max_depth=4,
                                   min_samples_leaf=20, random_state=42,
                                   class_weight="balanced_subsample").fit(Ztr, ytr)

    proba = model.predict_proba(Zte)[:, 1]
    signal = pd.Series((proba > PROB_THRESHOLD).astype(int), index=Xte.index)
    acc = accuracy_score(yte, (proba > 0.5).astype(int))
    return {"scaler": scaler, "pca": pca, "model": model, "signal": signal,
            "test_index": Xte.index, "test_acc": acc,
            "n_components": pca.n_components_,
            "explained": pca.explained_variance_ratio_,
            "feature_cols": list(X.columns)}


def latest_signal(df, scaler, pca, model, feature_cols):
    X, _ = build_features(df)
    row = X[feature_cols].iloc[[-1]]
    z = pca.transform(scaler.transform(row))
    proba = float(model.predict_proba(z)[0, 1])
    return (1 if proba > PROB_THRESHOLD else 0), proba, X.index[-1]


# ============================================================================
#  BACKTEST + METRICS
# ============================================================================
def run_backtest(close, positions, initial=INITIAL):
    ret = close.pct_change().fillna(0)
    strat = positions.shift(1).fillna(0) * ret
    equity = initial * (1 + strat).cumprod()
    trades, pos, price, idx = [], positions.to_numpy(), close.to_numpy(), close.index
    ep = None
    for i in range(1, len(pos)):
        if pos[i] == 1 and pos[i - 1] == 0:
            ep, ed = price[i], idx[i]
        elif pos[i] == 0 and pos[i - 1] == 1 and ep is not None:
            trades.append({"entry": ed, "exit": idx[i], "ret": price[i] / ep - 1,
                           "pnl": (price[i] / ep - 1)})
            ep = None
    if ep is not None:
        trades.append({"entry": ed, "exit": idx[-1], "ret": price[-1] / ep - 1,
                       "pnl": price[-1] / ep - 1})
    return {"equity": equity, "returns": strat, "trades": trades}


def metrics(res, initial=INITIAL):
    eq, ret, trades = res["equity"], res["returns"], res["trades"]
    total = eq.iloc[-1] / initial - 1
    years = max(len(eq) / TRADING_DAYS, 1e-9)
    cagr = (eq.iloc[-1] / initial) ** (1 / years) - 1
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = ret.mean() / ret.std() * np.sqrt(TRADING_DAYS) if ret.std() > 0 else 0.0
    dn = ret[ret < 0].std()
    sortino = ret.mean() / dn * np.sqrt(TRADING_DAYS) if dn and dn > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min()
    wins = sum(1 for t in trades if t["ret"] > 0)
    wr = wins / len(trades) if trades else 0.0
    return {"Total Return": total, "CAGR": cagr, "Volatility": vol,
            "Sharpe": sharpe, "Sortino": sortino, "Max Drawdown": dd,
            "Win Rate": wr, "Trades": len(trades)}


def analyze(df):
    X, y = build_features(df)
    trained = train_pipeline(X, y)
    test_close = df["close"].reindex(trained["test_index"])
    ml = run_backtest(test_close, trained["signal"])
    bh = run_backtest(test_close, pd.Series(1.0, index=trained["test_index"]))
    return {"trained": trained, "test_close": test_close,
            "ml": ml, "bh": bh,
            "ml_stats": metrics(ml), "bh_stats": metrics(bh)}


# ============================================================================
#  UI
# ============================================================================
DARK, PANEL, FG, MUTED = "#0d1117", "#161b22", "#c9d1d9", "#8b949e"


class MLApp:
    def __init__(self, root, connector):
        self.root = root
        self.conn = connector
        self.queue = queue.Queue()
        self.out = None
        self.symbol = None

        root.title("ML Trading Signal Terminal — Alpaca (PAPER)")
        root.configure(bg=DARK)
        root.geometry("1180x820")

        bar = tk.Frame(root, bg=DARK)
        bar.pack(fill="x", padx=12, pady=(12, 6))
        tk.Label(bar, text="Ticker:", fg=FG, bg=DARK, font=("Consolas", 12)).pack(side="left")
        self.symbol_var = tk.StringVar(value="AAPL")
        e = tk.Entry(bar, textvariable=self.symbol_var, width=9,
                     font=("Consolas", 14, "bold"), justify="center")
        e.pack(side="left", padx=8)
        e.bind("<Return>", lambda ev: self.train())
        tk.Button(bar, text="Train & Backtest  ▶", command=self.train,
                  font=("Consolas", 11, "bold"), bg="#238636", fg="white",
                  relief="flat", padx=12, pady=4).pack(side="left")
        for t in ("AAPL", "MSFT", "SPY", "QQQ", "NVDA"):
            tk.Button(bar, text=t, command=lambda s=t: self._quick(s),
                      font=("Consolas", 10), bg=PANEL, fg=FG, relief="flat",
                      padx=8).pack(side="left", padx=2)
        self.status = tk.Label(bar, text="Train a model, then submit a paper order.",
                               fg=MUTED, bg=DARK, font=("Consolas", 10))
        self.status.pack(side="right")

        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True, padx=12, pady=8)
        self.tab_bt = tk.Frame(self.nb, bg=DARK)
        self.tab_dd = tk.Frame(self.nb, bg=DARK)
        self.tab_pca = tk.Frame(self.nb, bg=DARK)
        self.tab_paper = tk.Frame(self.nb, bg=DARK)
        self.nb.add(self.tab_bt, text="Backtest")
        self.nb.add(self.tab_dd, text="Drawdowns")
        self.nb.add(self.tab_pca, text="PCA / Model")
        self.nb.add(self.tab_paper, text="Paper Trade")

        self.fig_bt = self._canvas(self.tab_bt)
        self.bt_text = tk.Text(self.tab_bt, height=9, bg=PANEL, fg=FG,
                               font=("Consolas", 10), relief="flat")
        self.bt_text.pack(fill="x", padx=6, pady=6)

        self.fig_dd = self._canvas(self.tab_dd)
        self.fig_pca = self._canvas(self.tab_pca)
        self._build_paper_tab()
        self.root.after(150, self._drain)

    def _canvas(self, parent):
        fig = Figure(figsize=(10, 4.6), facecolor=DARK)
        c = FigureCanvasTkAgg(fig, master=parent)
        c.get_tk_widget().pack(fill="both", expand=True)
        c.draw()
        return {"fig": fig, "canvas": c}

    def _build_paper_tab(self):
        top = tk.Frame(self.tab_paper, bg=DARK)
        top.pack(fill="x", padx=6, pady=6)
        self.paper_btn = tk.Button(top, text="Compute Live Signal & Submit PAPER Order",
                                   command=self.paper_trade, state="disabled",
                                   font=("Consolas", 11, "bold"), bg="#1f6feb",
                                   fg="white", relief="flat", padx=12, pady=4)
        self.paper_btn.pack(side="left")
        tk.Label(top, text="  PAPER TRADING ONLY — no real money.", fg="#d29922",
                 bg=DARK, font=("Consolas", 10, "bold")).pack(side="left")
        self.log = tk.Text(self.tab_paper, bg="#010409", fg="#3fb950",
                           font=("Consolas", 10), relief="flat", wrap="word")
        self.log.pack(fill="both", expand=True, padx=6, pady=6)

    def _logline(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.insert("end", f"[{ts}] {msg}\n")
        self.log.see("end")

    def _style(self, ax):
        ax.set_facecolor(DARK)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color("#30363d")
        ax.grid(True, color="#21262d", linewidth=0.5)

    # ---- train ----
    def _quick(self, s):
        self.symbol_var.set(s)
        self.train()

    def train(self):
        sym = self.symbol_var.get().upper().strip()
        if not sym:
            return
        self.symbol = sym
        self.status.config(text=f"Downloading {sym}, building features, training…",
                           fg="#d29922")
        threading.Thread(target=self._train_worker, args=(sym,), daemon=True).start()

    def _train_worker(self, sym):
        try:
            df = self.conn.get_daily(sym, years=5)
            if df is None or df.empty or len(df) < 300:
                self.queue.put(("error", f"Not enough data for {sym}."))
                return
            out = analyze(df)
            self.queue.put(("trained", sym, out))
        except Exception as e:
            self.queue.put(("error", str(e)))

    # ---- paper trade ----
    def paper_trade(self):
        if not self.out:
            return
        self.nb.select(self.tab_paper)
        self._logline(f"--- Live signal run for {self.symbol} (PAPER) ---")
        threading.Thread(target=self._paper_worker, daemon=True).start()

    def _paper_worker(self):
        try:
            t = self.out["trained"]
            df = self.conn.get_daily(self.symbol, years=1)
            sig, proba, asof = latest_signal(df, t["scaler"], t["pca"],
                                             t["model"], t["feature_cols"])
            price = float(df["close"].iloc[-1])
            self.queue.put(("log", f"Data as of {asof.date()} | last close ${price:,.2f}"))
            self.queue.put(("log", f"Model P(up)={proba:.3f}  ->  "
                                   f"signal = {'LONG' if sig else 'FLAT'} "
                                   f"(threshold {PROB_THRESHOLD})"))

            trading = self.conn.trading
            acct = trading.get_account()
            self.queue.put(("log", f"Paper account equity ${float(acct.equity):,.2f}, "
                                   f"buying power ${float(acct.buying_power):,.2f}"))

            held = 0.0
            try:
                pos = trading.get_open_position(self.symbol)
                held = float(pos.qty)
            except Exception:
                held = 0.0
            self.queue.put(("log", f"Current position: {held:g} shares"))

            if sig == 1:
                if held > 0:
                    self.queue.put(("log", "Signal LONG and already holding — no order."))
                else:
                    qty = max(1, int(TARGET_NOTIONAL / price))
                    order = MarketOrderRequest(symbol=self.symbol, qty=qty,
                                               side=OrderSide.BUY,
                                               time_in_force=TimeInForce.DAY)
                    o = trading.submit_order(order)
                    self.queue.put(("log", f"BUY submitted: {qty} {self.symbol} "
                                           f"(order id {o.id}, status {o.status})"))
            else:
                if held > 0:
                    o = trading.close_position(self.symbol)
                    self.queue.put(("log", f"SELL/close submitted for {self.symbol} "
                                           f"(order id {getattr(o,'id','n/a')})"))
                else:
                    self.queue.put(("log", "Signal FLAT and no position — no order."))
            self.queue.put(("log", "Done. Check your Alpaca paper dashboard for the order."))
        except Exception as e:
            self.queue.put(("log", f"ERROR: {e}"))

    # ---- queue ----
    def _drain(self):
        try:
            while True:
                m = self.queue.get_nowait()
                if m[0] == "trained":
                    _, sym, out = m
                    self.symbol, self.out = sym, out
                    self._render()
                    self.paper_btn.config(state="normal")
                    a, b = self.out["ml_stats"], self.out["bh_stats"]
                    self.status.config(
                        text=(f"{sym} trained | test acc {out['trained']['test_acc']*100:.1f}% | "
                              f"ML {a['Total Return']*100:+.1f}% vs B&H {b['Total Return']*100:+.1f}%"),
                        fg="#3fb950")
                elif m[0] == "log":
                    self._logline(m[1])
                elif m[0] == "error":
                    self.status.config(text=f"Error: {m[1]}", fg="#f85149")
        except queue.Empty:
            pass
        self.root.after(150, self._drain)

    # ---- render ----
    def _render(self):
        self._draw_backtest()
        self._draw_drawdown()
        self._draw_pca()
        self._fill_metrics()
        
    def _draw_backtest(self):
        fig = self.fig_bt["fig"]
        fig.clear()
        ax = fig.add_axes([0.08, 0.12, 0.88, 0.78])
        self._style(ax)
        ml, bh = self.out["ml"]["equity"], self.out["bh"]["equity"]
        ax.plot(ml.index, ml, color="#3fb950", linewidth=1.2, label="ML Signal")
        ax.plot(bh.index, bh, color="#8b949e", linewidth=1.0, label="Buy & Hold")
        ax.axhline(INITIAL, color="#30363d", linewidth=0.6, linestyle="--")
        ax.set_title(f"{self.symbol} — ML Signal vs Buy & Hold (out-of-sample test)",
                     color=FG, fontsize=11, loc="left")
        ax.legend(fontsize=9, loc="upper left", facecolor=PANEL,
                  edgecolor="#30363d", labelcolor=FG)
        self.fig_bt["canvas"].draw()

    def _draw_drawdown(self):
        fig = self.fig_dd["fig"]
        fig.clear()
        ax = fig.add_axes([0.08, 0.12, 0.88, 0.78])
        self._style(ax)
        for name, key, color in (("ML Signal", "ml", "#3fb950"),
                                  ("Buy & Hold", "bh", "#8b949e")):
            eq = self.out[key]["equity"]
            dd = (eq / eq.cummax() - 1) * 100
            ax.plot(dd.index, dd, color=color, linewidth=1.0, label=name)
        ax.set_title(f"{self.symbol} — Drawdowns (%), out-of-sample test",
                     color=FG, fontsize=11, loc="left")
        ax.legend(fontsize=9, loc="lower left", facecolor=PANEL,
                  edgecolor="#30363d", labelcolor=FG)
        self.fig_dd["canvas"].draw()

    def _draw_pca(self):
        fig = self.fig_pca["fig"]
        fig.clear()
        ax = fig.add_axes([0.08, 0.12, 0.88, 0.78])
        self._style(ax)
        ev = self.out["trained"]["explained"]
        x = np.arange(1, len(ev) + 1)
        ax.bar(x, ev * 100, color="#1f6feb", label="per component")
        ax.plot(x, np.cumsum(ev) * 100, color="#d29922", marker="o",
                linewidth=1.0, label="cumulative")
        ax.axhline(80, color="#3fb950", linewidth=0.7, linestyle="--")
        ax.set_xlabel("Principal component", color=MUTED, fontsize=8)
        ax.set_ylabel("Variance explained (%)", color=MUTED, fontsize=8)
        ax.set_title(f"PCA — {self.out['trained']['n_components']} components kept "
                     f"(≥80% variance)", color=FG, fontsize=11, loc="left")
        ax.legend(fontsize=8, loc="center right", facecolor=PANEL,
                  edgecolor="#30363d", labelcolor=FG)
        self.fig_pca["canvas"].draw()

    def _fill_metrics(self):
        a, b = self.out["ml_stats"], self.out["bh_stats"]
        t = self.out["trained"]
        self.bt_text.delete("1.0", "end")
        lines = []
        lines.append(f"Model: Random Forest on {t['n_components']} PCA components "
                     f"(from {len(t['feature_cols'])} features) | "
                     f"test accuracy {t['test_acc']*100:.1f}%")
        lines.append(f"Signal: LONG when P(next-day up) > {PROB_THRESHOLD}, else FLAT "
                     f"| out-of-sample test window: {len(self.out['test_close'])} days")
        lines.append("")
        hdr = f"{'Metric':<16}{'ML Signal':>14}{'Buy & Hold':>14}"
        lines.append(hdr)
        rows = [("Total Return", "Total Return", "{:+.1%}"),
                ("CAGR", "CAGR", "{:+.1%}"),
                ("Volatility", "Volatility", "{:.1%}"),
                ("Sharpe", "Sharpe", "{:.2f}"),
                ("Sortino", "Sortino", "{:.2f}"),
                ("Max Drawdown", "Max Drawdown", "{:.1%}"),
                ("Win Rate", "Win Rate", "{:.0%}"),
                ("Trades", "Trades", "{:d}")]
        for label, key, fmt in rows:
            lines.append(f"{label:<16}{fmt.format(a[key]):>14}{fmt.format(b[key]):>14}")
        self.bt_text.insert("1.0", "\n".join(lines))


def main():
    try:
        conn = DataConnector(API_KEY, SECRET_KEY)
    except RuntimeError as e:
        print(e)
        return
    root = tk.Tk()
    MLApp(root, conn)
    root.mainloop()


if __name__ == "__main__":
    main()
