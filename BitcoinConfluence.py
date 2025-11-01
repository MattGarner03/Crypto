# BitcoinConfluence.py
# deps: pip install -U ccxt yfinance pandas numpy scipy matplotlib

import math
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import ccxt
import yfinance as yf
from scipy.signal import argrelextrema

warnings.filterwarnings("ignore")

# ----------------- Parameters -----------------
START  = "2017-01-01"     # CME starts 2017; Bybit spot/perp has later start, it's fine
END    = None
SPLIT  = "2021-01-01"     # In-sample / Out-of-sample split (edit as you like)

CTX    = 250              # bars for volume profile (≈ 1y of daily bars)
VA_PCT = 0.70             # value area percent

FEE    = 0.0005           # fee per side
SLIP   = 0.0005           # slippage per side
COST   = FEE + SLIP

SWING_WIN   = 60          # days to detect last swing high/low
ATR_WIN     = 14
ENTRY_SCORE = 3           # confluence needed to enter (sum of booleans)
ATR_STOP_MULT = 1.5       # stop distance in ATR if structural stop not available

# ----------------- Data -----------------
def fetch_bybit_daily(symbols: list[str]) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Bybit via CCXT for given symbols.
    Returns DataFrame with MultiIndex columns: (symbol, field).
    """
    ex = ccxt.bybit({"enableRateLimit": True})
    ex.load_markets()
    frames = []
    for sym in symbols:
        if sym not in ex.markets:
            continue
        ohlcv = ex.fetch_ohlcv(sym, timeframe="1d", since=None, limit=2000)
        if not ohlcv:
            continue
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        # detect ms vs s
        unit = "ms" if df["ts"].max() > 10**12 else "s"
        df["ts"] = pd.to_datetime(df["ts"], unit=unit, utc=True)
        df.set_index("ts", inplace=True)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert(None)  # make tz-naive
        df.columns = pd.MultiIndex.from_product([[sym], df.columns])
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()

def pick_btc_symbol(exframe: pd.DataFrame) -> str:
    """Prefer USD perp; else USD spot; else USDT."""
    candidates = ["BTC/USD:USD", "BTC/USD", "BTC/USDT"]
    for c in candidates:
        if (c, "close") in exframe.columns:
            return c
    # fallback to any BTC/* we found
    symbols = list({c[0] for c in exframe.columns})
    for s in symbols:
        if s.startswith("BTC/"):
            return s
    raise RuntimeError("No BTC symbols found in Bybit data.")

def get_cme_daily(start, end) -> pd.DataFrame:
    """Yahoo CME Bitcoin futures (BTC=F)."""
    df = yf.download("BTC=F", start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("Could not fetch BTC=F (CME) from Yahoo.")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df.rename(columns=str.lower)

# ----------------- Indicators -----------------
def rolling_atr(df: pd.DataFrame, win: int = ATR_WIN) -> pd.Series:
    """Pandas-native ATR (no numpy arrays)."""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(win, min_periods=1).mean()

def volume_profile(ctx_close: pd.Series, ctx_vol: pd.Series, bins: int = CTX, va_pct: float = VA_PCT):
    """
    Simple close-price/volume histogram profile over context window.
    Returns POC, VAL, VAH, centers, hist.
    """
    lo, hi = ctx_close.min(), ctx_close.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        centers = np.array([float(lo)])
        hist = np.array([float(ctx_vol.sum())])
        poc_px = float(lo)
        return poc_px, np.nan, np.nan, centers, hist

    edges = np.linspace(lo, hi, bins + 1)
    hist, edges = np.histogram(ctx_close.values, bins=edges, weights=ctx_vol.values)
    centers = (edges[:-1] + edges[1:]) / 2.0

    if hist.sum() <= 0:
        return float(centers[0]), np.nan, np.nan, centers, hist

    poc_px = float(centers[hist.argmax()])

    # build symmetric VA around POC covering va_pct of volume
    idx = int(hist.argmax())
    L = R = idx
    cum = hist[idx]
    total = hist.sum()
    while cum < va_pct * total and (L > 0 or R < len(hist) - 1):
        left = hist[L - 1] if L > 0 else -np.inf
        right = hist[R + 1] if R < len(hist) - 1 else -np.inf
        if right >= left:
            R += 1; cum += hist[R]
        else:
            L -= 1; cum += hist[L]
    VAL = float(centers[L])
    VAH = float(centers[R])
    return poc_px, VAL, VAH, centers, hist

def detect_npocs(poc: pd.Series, high: pd.Series, low: pd.Series, ref_close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    NPOCs using daily bars:
      - Yesterday's POC becomes 'open' from today.
      - Remove an open POC as soon as today's [low, high] trades through it.
      - For each day, return nearest open NPOC strictly above/below today's **close**.
    """
    idx = poc.index
    open_levels: list[float] = []
    npoc_above = np.full(len(idx), np.nan, dtype=float)
    npoc_below = np.full(len(idx), np.nan, dtype=float)

    prev_p = np.nan
    for i in range(len(idx)):
        lo = float(low.iloc[i]); hi = float(high.iloc[i]); c = float(ref_close.iloc[i])

        # remove filled levels
        if open_levels:
            open_levels = [lvl for lvl in open_levels if not (lo <= lvl <= hi)]

        # add yesterday's poc
        if np.isfinite(prev_p):
            open_levels.append(float(prev_p))

        # record nearest above/below
        if open_levels:
            above = [lvl for lvl in open_levels if lvl > c]
            below = [lvl for lvl in open_levels if lvl < c]
            npoc_above[i] = min(above) if above else np.nan
            npoc_below[i] = max(below) if below else np.nan
        else:
            npoc_above[i] = np.nan
            npoc_below[i] = np.nan

        # store today's poc for tomorrow
        prev_p = float(poc.iloc[i]) if np.isfinite(poc.iloc[i]) else np.nan

    return pd.Series(npoc_above, index=idx, name="npoc_above"), pd.Series(npoc_below, index=idx, name="npoc_below")

def last_swings(df: pd.DataFrame, win: int = SWING_WIN) -> tuple[int, int]:
    """Indices of last swing low and last swing high using local extrema."""
    h = df["high"].values; l = df["low"].values
    highs = argrelextrema(h, np.greater_equal, order=win)[0]
    lows  = argrelextrema(l, np.less_equal,   order=win)[0]
    hi_idx = int(highs[-1]) if len(highs) else int(np.argmax(h))
    lo_idx = int(lows[-1])  if len(lows)  else int(np.argmin(l))
    return lo_idx, hi_idx

def anchored_vwap(close: pd.Series, vol: pd.Series, anchor_idx: int) -> pd.Series:
    """AVWAP anchored at anchor_idx (inclusive)."""
    pv = (close * vol).copy()
    v  = vol.copy()
    pv.iloc[:anchor_idx] = 0.0
    v.iloc[:anchor_idx] = 0.0
    vw = pv.cumsum() / v.cumsum()
    vw.iloc[:anchor_idx] = np.nan
    return vw

def golden_pocket_from_swing(lo_px: float, hi_px: float, uptrend: bool) -> tuple[float, float]:
    """Return (gp_low, gp_high) bounds for the 61.8–65% retracement."""
    hi, lo = (hi_px, lo_px)
    span = hi - lo
    if uptrend:
        # pullback in uptrend measured from hi → lo
        gp_low  = hi - 0.65 * span
        gp_high = hi - 0.618 * span
    else:
        gp_low  = lo + 0.618 * span
        gp_high = lo + 0.65  * span
    return float(min(gp_low, gp_high)), float(max(gp_low, gp_high))

def cme_unfilled_gaps(cme_df: pd.DataFrame) -> list[tuple[pd.Timestamp, float, float, str]]:
    """List gaps: (date, hi, lo, type='up' or 'down'). We treat them as magnets/targets."""
    gaps = []
    for i in range(1, len(cme_df)):
        prev = cme_df.iloc[i-1]; cur = cme_df.iloc[i]
        if cur["low"] > prev["high"]:   # gap up
            gaps.append((cme_df.index[i], prev["high"], cur["low"], "up"))
        elif cur["high"] < prev["low"]: # gap down
            gaps.append((cme_df.index[i], cur["high"], prev["low"], "down"))
    return gaps

# ----------------- Build feature set -----------------
def build_btc_frame() -> pd.DataFrame:
    # try Bybit USD perp → USD spot → USDT spot
    bybit = fetch_bybit_daily(["BTC/USD:USD", "BTC/USD", "BTC/USDT"])
    if bybit.empty:
        raise RuntimeError("Bybit data unavailable via CCXT.")
    sym = pick_btc_symbol(bybit)
    df = bybit[sym].copy()
    df = df.loc[df.index >= pd.Timestamp(START)]
    if END:
        df = df.loc[df.index <= pd.Timestamp(END)]
    df = df.rename(columns=str)  # ensure 'open','high','low','close','volume' as strings

    # Ensure correct column order
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    # ATR
    df["atr"] = rolling_atr(df, ATR_WIN)

    # Rolling volume profile POC/VAL/VAH
    poc, val, vah = [], [], []
    for i in range(len(df)):
        j0 = max(0, i - CTX + 1)
        ctx = df.iloc[j0:i+1]
        p_, vL_, vH_, _, _ = volume_profile(ctx["close"], ctx["volume"], bins=CTX, va_pct=VA_PCT)
        poc.append(p_); val.append(vL_); vah.append(vH_)
    df["poc"], df["val"], df["vah"] = poc, val, vah

    # NPOCs (use close as reference for "nearest above/below")
    npoc_up, npoc_dn = detect_npocs(df["poc"], df["high"], df["low"], df["close"])
    df["npoc_above"], df["npoc_below"] = npoc_up, npoc_dn

    # Swings & Anchored VWAPs
    lo_i, hi_i = last_swings(df, SWING_WIN)
    df["avwap_from_lo"] = anchored_vwap(df["close"], df["volume"], lo_i)
    df["avwap_from_hi"] = anchored_vwap(df["close"], df["volume"], hi_i)

    # Golden pocket
    lo_px, hi_px = float(df.iloc[lo_i]["low"]), float(df.iloc[hi_i]["high"])
    uptrend = hi_i > lo_i
    gp_low, gp_high = golden_pocket_from_swing(lo_px, hi_px, uptrend)
    df["gp_low"], df["gp_high"], df["swing_dir"] = gp_low, gp_high, (1 if uptrend else -1)

    # Daily prev levels
    d_prev = df.shift(1)
    df["y_high"], df["y_low"], df["y_open"] = d_prev["high"], d_prev["low"], d_prev["open"]

    # Weekly (Fri close week)
    wk = df.resample("W-FRI").agg({"open":"first","high":"max","low":"min","close":"last"})
    for col in ["open","high","low","close"]:
        df[f"w_{col}"] = wk[col].shift().reindex(df.index).ffill()

    # Monthly (month-end)
    mo = df.resample("ME").agg({"open":"first","high":"max","low":"min","close":"last"})
    for col in ["open","high","low","close"]:
        df[f"m_{col}"] = mo[col].shift().reindex(df.index).ffill()

    # CME gaps -> nearest mid above/below
    cme = get_cme_daily(START, END)
    gaps = cme_unfilled_gaps(cme)
    gap_above, gap_below = [], []
    for ts, row in df.iterrows():
        mids_up   = [ (hi+lo)/2 for (t,hi,lo,typ) in gaps if typ=="up"   and ts>=t and row["close"] < lo ]
        mids_down = [ (hi+lo)/2 for (t,hi,lo,typ) in gaps if typ=="down" and ts>=t and row["close"] > hi ]
        gap_above.append(min(mids_up) if mids_up else np.nan)
        gap_below.append(max(mids_down) if mids_down else np.nan)
    df["cme_mid_above"], df["cme_mid_below"] = gap_above, gap_below

    # Drop earliest rows until ATR exists
    return df.dropna(subset=["atr"])

# ----------------- Signals (confluence) -----------------
def confluence_signals(df: pd.DataFrame, tol: float = 0.003) -> pd.DataFrame:
    s = pd.DataFrame(index=df.index)

    # NPOC touch & reject
    s["touch_npoc_long"]  = df["npoc_below"].notna() & (df["low"]  <= df["npoc_below"]*(1+tol)) & (df["close"] > df["npoc_below"])
    s["touch_npoc_short"] = df["npoc_above"].notna() & (df["high"] >= df["npoc_above"]*(1-tol)) & (df["close"] < df["npoc_above"])

    # AVWAP support/resistance
    s["avwap_support"] = df["avwap_from_lo"].notna() & (df["low"]  <= df["avwap_from_lo"]*(1+tol)) & (df["close"] > df["avwap_from_lo"])
    s["avwap_resist"]  = df["avwap_from_hi"].notna() & (df["high"] >= df["avwap_from_hi"]*(1-tol)) & (df["close"] < df["avwap_from_hi"])

    # Value area re-entries
    s["reenter_value_long"]  = df["val"].notna() & (df["close"].shift(1) < df["val"].shift(1)) & (df["close"] > df["val"])
    s["reenter_value_short"] = df["vah"].notna() & (df["close"].shift(1) > df["vah"].shift(1)) & (df["close"] < df["vah"])

    # Golden pocket (directional)
    gp_hit = df["gp_low"].notna() & df["gp_high"].notna() & (df["low"] <= df["gp_high"]) & (df["high"] >= df["gp_low"])
    s["golden_long"]  = gp_hit & (df["swing_dir"]==1)  & (df["close"] > df["gp_low"])
    s["golden_short"] = gp_hit & (df["swing_dir"]==-1) & (df["close"] < df["gp_high"])

    # CME target existence (as magnet)
    s["cme_up"]   = df["cme_mid_above"].notna()
    s["cme_down"] = df["cme_mid_below"].notna()

    return s.fillna(False).astype(bool)

# ----------------- Backtest -----------------
def backtest(df: pd.DataFrame, sigs: pd.DataFrame) -> pd.Series:
    """
    Daily, next-bar execution. Position ∈ {-1, 0, +1}.
    Stop/target:
      Long: stop = min(VAL, NPOC_below, entry - 1.5*ATR)
            target = max(VAH, CME_mid_above) or entry + 2*ATR
      Short: mirrored.
    Returns a daily returns Series aligned to df.index[1:].
    """
    pos = 0
    entry_px = stop_px = target_px = None
    rets = []

    for i in range(1, len(df)):
        y, t = df.iloc[i-1], df.iloc[i]          # decide on y, execute on t
        sy   = sigs.iloc[i-1]
        atr  = y["atr"]

        # evaluate exit if in position on today's range
        if pos != 0 and stop_px is not None and target_px is not None:
            hit_stop = (t["low"] <= stop_px) if pos>0 else (t["high"] >= stop_px)
            hit_tgt  = (t["high"] >= target_px) if pos>0 else (t["low"] <= target_px)
            if hit_stop or hit_tgt:
                exit_px = stop_px if hit_stop else target_px
                pnl = (exit_px / entry_px - 1.0) * pos
                rets.append(pnl - COST)
                pos = 0; entry_px = stop_px = target_px = None
                continue

        # build confluence scores
        long_score = int(sy["touch_npoc_long"])  + int(sy["avwap_support"]) + int(sy["reenter_value_long"]) \
                   + int(sy["golden_long"])     + int(sy["cme_up"])
        short_score = int(sy["touch_npoc_short"]) + int(sy["avwap_resist"]) + int(sy["reenter_value_short"]) \
                    + int(sy["golden_short"])    + int(sy["cme_down"])

        # entries when flat
        if pos == 0:
            if long_score >= ENTRY_SCORE and np.isfinite(y["val"]):
                pos = +1
                entry_px = float(t["open"])  # next bar open
                structural = []
                if np.isfinite(y["val"]): structural.append(float(y["val"]))
                if np.isfinite(y["npoc_below"]): structural.append(float(y["npoc_below"]))
                if structural:
                    stop_px = min(min(structural), entry_px - ATR_STOP_MULT*atr)
                else:
                    stop_px = entry_px - ATR_STOP_MULT*atr
                t_candidates = []
                if np.isfinite(y["vah"]): t_candidates.append(float(y["vah"]))
                if np.isfinite(y["cme_mid_above"]): t_candidates.append(float(y["cme_mid_above"]))
                target_px = max(t_candidates) if t_candidates else entry_px + 2*atr
                rets.append(0.0)
                continue

            if short_score >= ENTRY_SCORE and np.isfinite(y["vah"]):
                pos = -1
                entry_px = float(t["open"])
                structural = []
                if np.isfinite(y["vah"]): structural.append(float(y["vah"]))
                if np.isfinite(y["npoc_above"]): structural.append(float(y["npoc_above"]))
                if structural:
                    stop_px = max(max(structural), entry_px + ATR_STOP_MULT*atr)
                else:
                    stop_px = entry_px + ATR_STOP_MULT*atr
                t_candidates = []
                if np.isfinite(y["val"]): t_candidates.append(float(y["val"]))
                if np.isfinite(y["cme_mid_below"]): t_candidates.append(float(y["cme_mid_below"]))
                target_px = min(t_candidates) if t_candidates else entry_px - 2*atr
                rets.append(0.0)
                continue

            rets.append(0.0)
        else:
            # mark-to-market while in position (no extra turnover unless exit/flip)
            day_ret = (t["close"] / y["close"] - 1.0) * pos
            rets.append(day_ret)

    return pd.Series(rets, index=df.index[1:], name="ret")

# ----------------- Stats & Run -----------------
def stats_ser(r: pd.Series) -> pd.Series:
    r = r.dropna()
    if r.empty:
        return pd.Series({"Total Return": np.nan, "CAGR": np.nan, "Sharpe": np.nan, "MaxDD": np.nan})
    ann = 252
    eq = (1 + r).cumprod()
    total = float(eq.iloc[-1] - 1)
    cagr  = float((1 + r).prod() ** (ann / len(r)) - 1)
    sharpe = float(np.sqrt(ann) * r.mean() / (r.std() + 1e-12))
    mdd   = float((eq / eq.cummax() - 1).min())
    return pd.Series({"Total Return": total, "CAGR": cagr, "Sharpe": sharpe, "MaxDD": mdd})

def run():
    df = build_btc_frame()
    sigs = confluence_signals(df)
    r = backtest(df, sigs)

    ins = r.loc[:pd.Timestamp(SPLIT) - pd.Timedelta(days=1)]
    oos = r.loc[pd.Timestamp(SPLIT):]

    print(f"Rows: {len(df):,}  Start: {df.index[0].date()}  End: {df.index[-1].date()}")
    print("\n=== BTC Confluence Strategy ===")
    print(f"In-sample (< {SPLIT})");  print(stats_ser(ins).to_frame("Value"))
    print(f"\nOut-of-sample (≥ {SPLIT})"); print(stats_ser(oos).to_frame("Value"))

    # Save artifacts
    (1 + r).cumprod().to_frame("Equity").to_csv("btc_confluence_equity.csv")
    sigs.to_csv("btc_confluence_signals.csv")
    df.to_csv("btc_confluence_features.csv")
    print("\nSaved: btc_confluence_equity.csv, btc_confluence_signals.csv, btc_confluence_features.csv")

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        import traceback; traceback.print_exc()
