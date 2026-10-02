import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yfinance as yf

from model import Kronos, KronosPredictor, KronosTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("ticker", nargs="?", default="BTC-USD")
parser.add_argument("--pred-len", type=int, default=24, help="hourly steps to forecast")
parser.add_argument("--runs", type=int, default=20)
parser.add_argument("--lookback", type=int, default=400, help="hourly candles of history")
parser.add_argument("--json", action="store_true", help="print the result as JSON instead of saving a PNG")
args = parser.parse_args()

LOOKBACK = args.lookback
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- data ---
raw = yf.download(args.ticker, period="60d", interval="1h", auto_adjust=False, progress=False)
if raw.empty:
    raise SystemExit(f"No data returned for {args.ticker}")
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
raw = raw.rename(columns=str.lower).dropna(subset=["open", "high", "low", "close"])
df = raw.tail(LOOKBACK)
if len(df) < LOOKBACK:
    print(f"Warning: only {len(df)} candles available")
df = df[["open", "high", "low", "close", "volume"]].copy()
df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)

idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
x_ts = pd.Series(idx, name="timestamps")
def future_timestamps(idx, n):
    """Next n candle times. 24/7 markets: every hour. Equities: only weekdays at observed bar times."""
    if idx.to_series().diff().max() <= pd.Timedelta(hours=1):
        return pd.date_range(idx[-1] + pd.Timedelta(hours=1), periods=n, freq="h")
    bar_times = {(t.hour, t.minute) for t in idx}
    out, t = [], idx[-1]
    while len(out) < n:
        t += pd.Timedelta(minutes=30)
        if t.weekday() < 5 and (t.hour, t.minute) in bar_times:
            out.append(t)
    return pd.DatetimeIndex(out)


y_ts = pd.Series(future_timestamps(idx, args.pred_len))
df = df.reset_index(drop=True)

# --- model ---
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
print(f"Device: {device}")

# --- 20 independent single-sample runs ---
paths = []
for i in range(args.runs):
    pred = predictor.predict(
        df=df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=args.pred_len,
        T=1.0, top_p=0.9, sample_count=1, verbose=False,
    )
    paths.append(pred["close"].values)
    print(f"run {i + 1}/{args.runs} done")
paths = np.array(paths)  # (runs, pred_len)

lo, med, hi = np.percentile(paths, [5, 50, 95], axis=0)

if args.json:
    # Machine-readable result for the BusinessBasics Forecast tab (no PNG).
    import json

    n = min(40, len(idx))
    iso = lambda t: pd.Timestamp(t).isoformat()
    print("RESULT_JSON " + json.dumps({
        "ticker": args.ticker, "model": "Kronos-small", "runs": args.runs, "predLen": args.pred_len,
        "lastClose": float(df["close"].iloc[-1]),
        "history": {"t": [iso(t) for t in idx[-n:]], "close": [round(float(v), 4) for v in df["close"].values[-n:]]},
        "forecast": {"t": [iso(t) for t in y_ts], "lo": np.round(lo, 4).tolist(), "med": np.round(med, 4).tolist(),
                     "hi": np.round(hi, 4).tolist(), "paths": np.round(paths, 4).tolist()},
    }))
    raise SystemExit(0)

# --- plot ---
# x axis = candle position (not wall-clock), so overnight/weekend gaps don't distort the chart
PLOT_HIST = min(40, len(idx))
fmt_t = lambda t: pd.Timestamp(t).strftime("%a %b %d %H:%M")
last_c = float(df["close"].iloc[-1])
hist_t = list(idx[-PLOT_HIST:])
fut_t = list(y_ts)
all_t = hist_t + fut_t
hx = np.arange(PLOT_HIST)
x0 = PLOT_HIST - 1                      # position of the last real candle
fx = np.arange(PLOT_HIST, PLOT_HIST + len(fut_t))

fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(hx, df["close"].values[-PLOT_HIST:], color="black", lw=2, label="History (close)")
ax.plot(fx, paths.T, color="tab:blue", alpha=0.15, lw=0.8)
ax.fill_between(fx, lo, hi, color="tab:blue", alpha=0.2, label="5–95% range")
ax.plot([x0, fx[0]], [last_c, med[0]], color="tab:blue", lw=2.5)
ax.plot(fx, med, color="tab:blue", lw=2.5, label="Median forecast")
ax.axvline(x0, color="gray", ls="--", lw=1)
ax.axhline(last_c, color="gray", ls=":", lw=1)
ax.axvspan(x0, fx[-1], color="tab:blue", alpha=0.04)


def mark(x, p, t, text, color, dx, dy, marker, ha="center"):
    ax.scatter([x], [p], color=color, s=90, zorder=5, marker=marker, edgecolor="white")
    ax.annotate(f"{text}\n${p:,.2f}\n{fmt_t(t)}", (x, p), xytext=(dx, dy), textcoords="offset points",
                fontsize=10, color=color, fontweight="bold", ha=ha,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color),
                arrowprops=dict(arrowstyle="-", color=color), zorder=6)


hi_i, lo_i = int(np.argmax(hi)), int(np.argmin(lo))
mark(x0, last_c, hist_t[-1], "START (last close)", "black", -70, 90, "s")
mark(fx[-1], med[-1], fut_t[-1], "END (median)", "tab:blue", 90, 0, "D", ha="left")
mark(fx[hi_i], hi[hi_i], fut_t[hi_i], "HIGH (95th pct)", "tab:green", 60, 50, "^")
mark(fx[lo_i], lo[lo_i], fut_t[lo_i], "LOW (5th pct)", "tab:red", 60, -50, "v")

ax.set_title(f"{args.ticker} Kronos-small forecast — {args.runs} runs, next {args.pred_len} candles\n"
             f"{fmt_t(fut_t[0])} → {fmt_t(fut_t[-1])}  |  median change {med[-1] - last_c:+.2f} "
             f"({(med[-1] / last_c - 1) * 100:+.2f}%)", fontsize=14)
ax.set_ylabel("Price (USD)")
ax.set_xlabel("Time (market candles; overnight/weekend gaps removed)")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.2f}"))
ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(14))
ticks = list(range(0, len(all_t), 4))
ax.set_xticks(ticks)
ax.set_xticklabels([pd.Timestamp(all_t[i]).strftime("%b %d\n%H:%M") for i in ticks], fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, fx[-1] + 12)
ax.margins(y=0.2)
ax.legend(loc="upper left")
out = f"forecast_{args.ticker.replace('/', '_')}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
print(f"Start  {fmt_t(hist_t[-1])}  ${last_c:,.2f}")
print(f"End    {fmt_t(fut_t[-1])}  ${med[-1]:,.2f} (median), 5–95%: ${lo[-1]:,.2f}–${hi[-1]:,.2f}")
print(f"High   {fmt_t(fut_t[hi_i])}  ${hi[hi_i]:,.2f} (95th pct)")
print(f"Low    {fmt_t(fut_t[lo_i])}  ${lo[lo_i]:,.2f} (5th pct)")
