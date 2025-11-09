#!/usr/bin/env python3
"""
Plot Bitcoin price with multiple sentiment indices.

Looks for these CSVs in the current folder (skips if missing):
  - alt_fear_greed.csv
  - cmc_fear_greed.csv
  - cnn_fgi.csv
  - santiment_btc_sentiment.csv
  - augmento_bull_bear.csv

Downloads BTC-USD daily price with yfinance and plots:
  - Top panel: BTC price (Close, adjusted)
  - Bottom panel: sentiment series (scaled to 0..100 when needed)

Output:
  - btc_sentiment_dashboard.png
"""

import os
from datetime import datetime, timezone

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# -------------------------- helpers --------------------------

def read_csv_if_exists(path: str):
    """Read CSV if present. Return DataFrame indexed by UTC-normalised date, else None."""
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found.")
        return None
    df = pd.read_csv(path)
    # Prefer 'date' column; else fall back to 'timestamp'
    date_col = "date" if "date" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    if not date_col:
        print(f"[WARN] {path} has no 'date' or 'timestamp'; skipping.")
        return None

    # Parse to UTC-normalised daily index
    dt_series = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    dt_series = dt_series.dt.tz_convert("UTC").dt.normalize()
    df = df.assign(_date=dt_series).dropna(subset=["_date"]).set_index("_date").sort_index()
    return df


def normalize_0_100(series: pd.Series) -> pd.Series:
    """Scale a numeric series to 0..100. Flat series -> 50."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return s
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(50.0, index=s.index)
    return (s - lo) / (hi - lo) * 100.0


def extract_price_series(btc_df: pd.DataFrame) -> pd.Series:
    """
    Return a single 'Close' price series from a yfinance DataFrame.
    Handles both single-index and MultiIndex columns.
    Assumes auto_adjust=True, so 'Close' is adjusted already.
    """
    if btc_df.empty:
        raise SystemExit("BTC price download returned empty DataFrame.")

    if isinstance(btc_df.columns, pd.MultiIndex):
        # Common yfinance layout: levels ['Price','Ticker'] with ('Close','BTC-USD')
        col_names = list(btc_df.columns.names or [])
        # Try selecting by the level named 'Price' if present
        if "Price" in col_names:
            s = btc_df.xs("Close", axis=1, level="Price")
        else:
            # Fallback: try selecting 'Close' from level 0, then level -1
            try:
                s = btc_df.xs("Close", axis=1, level=0)
            except Exception:
                s = btc_df.xs("Close", axis=1, level=-1)

        # If we still have a DataFrame (multiple tickers), pick BTC-USD if present, else first column
        if isinstance(s, pd.DataFrame):
            if "BTC-USD" in s.columns:
                s = s["BTC-USD"]
            else:
                s = s.iloc[:, 0].squeeze()
    else:
        # Single-index columns: prefer 'Close', else 'Adj Close'
        if "Close" in btc_df.columns:
            s = btc_df["Close"]
        elif "Adj Close" in btc_df.columns:
            s = btc_df["Adj Close"]
        else:
            raise KeyError(f"No 'Close' or 'Adj Close' in BTC columns: {btc_df.columns}")

    s.index = pd.to_datetime(s.index, utc=True).normalize()
    return s



# -------------------------- load sentiment files --------------------------

series = {}  # name -> Series (indexed by date)

# Alternative.me FGI
alt = read_csv_if_exists("alt_fear_greed.csv")
if alt is not None and "value" in alt.columns:
    series["Alt FGI"] = pd.to_numeric(alt["value"], errors="coerce")

# CoinMarketCap FGI
cmc = read_csv_if_exists("cmc_fear_greed.csv")
if cmc is not None and "value" in cmc.columns:
    series["CMC FGI"] = pd.to_numeric(cmc["value"], errors="coerce")

# CNN Stock Market FGI
cnn = read_csv_if_exists("cnn_fgi.csv")
if cnn is not None and "value" in cnn.columns:
    series["CNN Stock FGI"] = pd.to_numeric(cnn["value"], errors="coerce")

# Santiment (whatever metric you saved to santiment_btc_sentiment.csv)
san = read_csv_if_exists("santiment_btc_sentiment.csv")
if san is not None and "value" in san.columns:
    series["Santiment (weighted)"] = pd.to_numeric(san["value"], errors="coerce")

# Augmento Bull & Bear (index 0..1 -> 0..100), or compute from bull/bear
aug = read_csv_if_exists("augmento_bull_bear.csv")
if aug is not None:
    if "index" in aug.columns:
        series["Augmento Bull/Bear (idx)"] = pd.to_numeric(aug["index"], errors="coerce") * 100.0
    elif set(["bull", "bear"]).issubset(aug.columns):
        bb = pd.to_numeric(aug["bull"], errors="coerce")
        br = pd.to_numeric(aug["bear"], errors="coerce")
        idx = (bb / (bb + br)).where((bb + br) > 0) * 100.0
        series["Augmento Bull/Bear (idx)"] = idx

if not series:
    raise SystemExit("No sentiment CSVs found — put this script next to your CSV files and retry.")


# -------------------------- combine & fetch price --------------------------

sent_df = pd.DataFrame(series).sort_index()
# Daily sampling; mean for any duplicate days; keep daily cadence
sent_df = sent_df.resample("1D").mean()

start_date = sent_df.index.min() if len(sent_df.index) else pd.Timestamp("2015-01-01", tz="UTC")
end_date = pd.Timestamp(datetime.now(timezone.utc)).normalize()

print(f"[INFO] Downloading BTC-USD from {start_date.date()} to {end_date.date()}...")
btc = yf.download(
    "BTC-USD",
    start=start_date.tz_localize(None),
    end=end_date.tz_localize(None),
    interval="1d",
    auto_adjust=True,   # modern yfinance default; ensures 'Close' is adjusted
    progress=False
)
price = extract_price_series(btc)

# Align on intersection of dates
df = sent_df.join(price.rename("BTC-USD"), how="inner").dropna(subset=["BTC-USD"]).sort_index()
if df.empty:
    raise SystemExit("No overlapping dates between BTC price and sentiment series.")


# -------------------------- normalise sentiment panel --------------------------

norm_panel = {}
for name in series.keys():
    if name not in df.columns:
        continue
    # If it's known to already be a 0..100 gauge (FGIs, Augmento idx), use as-is;
    # otherwise scale to 0..100 for comparability.
    if any(k in name.lower() for k in ["fgi", "greed", "aug"]):
        norm_panel[name] = df[name]
    else:
        norm_panel[name] = normalize_0_100(df[name])

sent_norm = pd.DataFrame(norm_panel)


# -------------------------- plot --------------------------

plt.figure(figsize=(14, 8))

# Top: BTC price
ax1 = plt.subplot(2, 1, 1)
ax1.plot(df.index, df["BTC-USD"], label="BTC-USD")
ax1.set_title("Bitcoin Price (BTC-USD)")
ax1.set_ylabel("Price (USD)")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper left")
ax1.set_xmargin(0)

# Bottom: Sentiment (0..100)
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
for col in sent_norm.columns:
    ax2.plot(sent_norm.index, sent_norm[col], label=col)
ax2.set_title("Sentiment Indices (normalised to 0–100 where needed)")
ax2.set_ylabel("Sentiment (0–100)")
ax2.set_xlabel("Date")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 100)
# Light bands similar to FGI interpretation (no colors specified)
ax2.axhspan(0, 25, alpha=0.05)     # Extreme Fear
ax2.axhspan(25, 45, alpha=0.05)    # Fear
ax2.axhspan(55, 75, alpha=0.05)    # Greed
ax2.axhspan(75, 100, alpha=0.05)   # Extreme Greed
ax2.legend(ncols=2, loc="upper left", framealpha=0.85)
ax2.set_xmargin(0)

df.to_csv("btc_sentiment_merged.csv", index_label="date")
plt.tight_layout()
OUT_PATH = "btc_sentiment_dashboard.png"
plt.savefig(OUT_PATH, dpi=160)
print(f"[DONE] Saved plot -> {OUT_PATH}")
