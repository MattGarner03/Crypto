# btc_confluence_250d.py
import math, warnings, numpy as np, pandas as pd
import ccxt, yfinance as yf
from scipy.signal import argrelextrema
from datetime import datetime
warnings.filterwarnings("ignore")

# ----------------- Params -----------------
START  = "2017-01-01"     # CME starts 2017; feel free to push earlier for spot
END    = None
SPLIT  = "2021-01-01"     # IS/OOS split (edit freely)
CTX    = 250              # context window for volume profile (≈ 1Y of trading days)
VA_PCT = 0.70             # value area percent
FEE    = 0.0005           # fee per side
SLIP   = 0.0005           # slippage per side
COST   = FEE + SLIP

SWING_WIN = 60            # lookback to find last swing H/L for anchors
ATR_WIN   = 14
ENTRY_SCORE = 3           # confluence needed to enter
RISK_R    = 1.0           # risk per trade in R units (position sizing here is 1x notional; you can adapt)

# ----------------- Data -----------------


def fetch_bybit_daily(symbols):
    ex = ccxt.bybit({"enableRateLimit": True})
    ex.load_markets()
    frames = []
    for sym in symbols:
        if sym not in ex.markets:
            continue
        ohlcv = ex.fetch_ohlcv(sym, timeframe="1d", since=None, limit=2000)
        if not ohlcv:
            continue

        df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])

        # robust epoch handling (ms vs s)
        unit = "ms" if df["ts"].max() > 10**12 else "s"
        df["ts"] = pd.to_datetime(df["ts"], unit=unit, utc=True)

        # make it the index first, THEN tz_convert on the index
        df.set_index("ts", inplace=True)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert(None)  # tz-aware -> tz-naive

        # multiindex columns per symbol
        df.columns = pd.MultiIndex.from_product([[sym], df.columns])
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()



def pick_spot_from_bybit(exframe):
    # Prefer USD perp, else linear, else anything crypto/USD/USDT we got
    for cand in ["BTC/USD:USD","BTC/USD","BTC/USDT"]:
        if (cand, "close") in exframe.columns:
            return cand
    # fallback to any BTC/* symbol
    for col in exframe.columns.get_level_values(0).unique():
        if col.startswith("BTC/"):
            return col
    raise RuntimeError("No BTC symbol found in Bybit data.")

def get_cme_daily(start, end):
    # Yahoo CME Bitcoin futures
    df = yf.download("BTC=F", start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("Could not fetch BTC=F from Yahoo for CME gaps.")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df.rename(columns=str.lower)

# ----------------- Indicators -----------------
def rolling_atr(df, win=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(win, min_periods=1).mean()


def volume_profile(ctx_close, ctx_vol, bins=CTX, va_pct=VA_PCT):
    lo, hi = ctx_close.min(), ctx_close.max()
    if lo == hi: 
        p_bins = np.array([lo, lo+1e-9])
        hist = np.array([ctx_vol.sum()])
        centers = np.array([lo])
    else:
        p_bins = np.linspace(lo, hi, bins+1)
        hist, edges = np.histogram(ctx_close, bins=p_bins, weights=ctx_vol)
        centers = (edges[:-1] + edges[1:]) / 2
    # POC
    poc_px = centers[hist.argmax()]
    # Value area: smallest symmetric band around POC covering va_pct of volume
    tot = hist.sum()
    if tot == 0: 
        return poc_px, np.nan, np.nan, centers, hist
    idx = hist.argmax()
    cum = hist[idx]
    L, R = idx, idx
    while cum < va_pct * tot and (L > 0 or R < len(hist)-1):
        left  = hist[L-1] if L>0 else -np.inf
        right = hist[R+1] if R<len(hist)-1 else -np.inf
        if right >= left:
            R += 1; cum += hist[R]
        else:
            L -= 1; cum += hist[L]
    val = centers[L]; vah = centers[R]
    return float(poc_px), float(val), float(vah), centers, hist

def detect_npocs(poc_series, high, low):
    """
    An NPOC forms on day t at price P if later days never traded through P.
    We mark it as 'open' until touched (low<=P<=high), then it's removed.
    We return a series with the nearest open NPOC above/below each day.
    """
    open_levels = []  # (price, day_index)
    near_up, near_dn = [], []
    for i in range(len(poc_series)):
        p = poc_series.iloc[i]
        # resolve touches
        price_low, price_high = low.iloc[i], high.iloc[i]
        open_levels = [(lvl, day) for (lvl, day) in open_levels if not (price_low <= lvl <= price_high)]
        # add today's POC (yesterday's POC becomes candidate)
        if not np.isnan(p):
            open_levels.append((p, i))
        # nearest above/below for today
        if open_levels:
            diffs = np.array([lvl - poc_series.index[i]*0 + 0 for lvl,_ in open_levels])  # dummy just to keep structure
            prices = np.array([lvl for (lvl,_) in open_levels])
            above = prices[prices >= low.iloc[i]]
            below = prices[prices <= high.iloc[i]]
        # compute nearest levels
        lvls = np.array([lvl for (lvl,_) in open_levels]) if open_levels else np.array([])
        if lvls.size:
            na = lvls[lvls >= low.iloc[i]]
            nb = lvls[lvls <= high.iloc[i]]
        else:
            na, nb = np.array([]), np.array([])"/l"
        near_up.append(na.min() if na.size else np.nan)
        near_dn.append(nb.max() if nb.size else np.nan)
    return pd.Series(near_up, index=poc_series.index, name="npoc_above"), \
           pd.Series(near_dn, index=poc_series.index, name="npoc_below")

def anchored_vwap(close, vol, anchor_idx):
    """Classic cumulative PV / V anchored at index anchor_idx."""
    pv = (close * vol).copy()
    pv.iloc[:anchor_idx] = 0
    v  = vol.copy()
    v.iloc[:anchor_idx] = 0
    vw = pv.cumsum() / v.cumsum()
    vw.iloc[:anchor_idx] = np.nan
    return vw

def last_swings(df, win=SWING_WIN):
    """Return indices of last swing low and last swing high using rolling local extrema."""
    h = df["high"].values; l = df["low"].values
    # local extrema
    highs = argrelextrema(h, np.greater_equal, order=win)[0]
    lows  = argrelextrema(l, np.less_equal,   order=win)[0]
    hi_idx = highs[-1] if len(highs) else np.argmax(h)
    lo_idx = lows[-1]  if len(lows)  else np.argmin(l)
    return int(lo_idx), int(hi_idx)

def golden_pocket(level_a, level_b):
    """Return (fib_618, fib_65) between two prices."""
    hi, lo = max(level_a, level_b), min(level_a, level_b)
    span = hi - lo
    return hi - 0.618*span, hi - 0.65*span  # for pullback to an up-move
# For down-move, logic mirrored in code.

def cme_unfilled_gaps(cme_df):
    """Return list of gaps (start_date, hi_gap, lo_gap) that remain unfilled till later."""
    gaps = []
    for i in range(1, len(cme_df)):
        prev = cme_df.iloc[i-1]; cur = cme_df.iloc[i]
        # gap up if cur.low > prev.high; gap down if cur.high < prev.low
        if cur["low"] > prev["high"]:
            gaps.append((cme_df.index[i], prev["high"], cur["low"], "up"))
        elif cur["high"] < prev["low"]:
            gaps.append((cme_df.index[i], cur["high"], prev["low"], "down"))
    # we do NOT mark fills here; in backtest we check if spot traded into gap
    return gaps

# ----------------- Build feature set -----------------
def build_btc_frame():
    bybit = fetch_bybit_daily(["BTC/USD:USD","BTC/USD","BTC/USDT"])
    if bybit.empty: 
        raise RuntimeError("Bybit data not available via ccxt.")
    sym = pick_spot_from_bybit(bybit)
    df = bybit[sym].copy()
    df = df.loc[df.index >= pd.Timestamp(START)]
    if END:
        df = df.loc[df.index <= pd.Timestamp(END)]
    df.columns = ["open","high","low","close","volume"]

    # ATR
    df["atr"] = rolling_atr(df)

    # rolling volume profile POC/VAL/VAH
    poc, val, vah = [], [], []
    for i in range(len(df)):
        j0 = max(0, i-CTX+1)
        ctx = df.iloc[j0:i+1]
        p, vL, vH, _, _ = volume_profile(ctx["close"], ctx["volume"], bins=CTX, va_pct=VA_PCT)
        poc.append(p); val.append(vL); vah.append(vH)
    df["poc"], df["val"], df["vah"] = poc, val, vah

    # NPOCs
    npoc_up, npoc_dn = detect_npocs(df["poc"], df["high"], df["low"])
    df["npoc_above"], df["npoc_below"] = npoc_up, npoc_dn

    # swings and anchored VWAPs
    lo_i, hi_i = last_swings(df)
    df["avwap_from_lo"] = anchored_vwap(df["close"], df["volume"], lo_i)
    df["avwap_from_hi"] = anchored_vwap(df["close"], df["volume"], hi_i)

    # golden pockets from active swing
    lo_px, hi_px = df.iloc[lo_i]["low"], df.iloc[hi_i]["high"]
    if hi_i > lo_i:
        gp_lo, gp_hi = golden_pocket(hi_px, lo_px)  # pullback in uptrend
        df["gp_low"], df["gp_high"], df["swing_dir"] = gp_hi, gp_lo, 1
    else:
        # downtrend pullback
        span = hi_px - lo_px
        gp_l = lo_px + 0.618*span
        gp_h = lo_px + 0.65*span
        df["gp_low"], df["gp_high"], df["swing_dir"] = gp_l, gp_h, -1

    # D/W/M levels
    d_prev = df.shift(1)
    df["y_high"], df["y_low"], df["y_open"] = d_prev["high"], d_prev["low"], d_prev["open"]
    # weekly
    wk = df.resample("W-FRI").agg({"open":"first","high":"max","low":"min","close":"last"})
    for col in ["open","high","low","close"]:
        df[f"w_{col}"] = wk[col].shift().reindex(df.index).ffill()
    # monthly
    mo = df.resample("ME").agg({"open":"first","high":"max","low":"min","close":"last"})
    for col in ["open","high","low","close"]:
        df[f"m_{col}"] = mo[col].shift().reindex(df.index).ffill()

    # CME gaps
    cme = get_cme_daily(START, END)
    gaps = cme_unfilled_gaps(cme)
    # nearest gap above/below each day (by mid-gap)
    gap_above = []; gap_below = []
    for ts, row in df.iterrows():
        mids_up   = [ (hi+lo)/2 for (t,hi,lo,typ) in gaps if typ=="up"   and ts>=t and row["close"] < lo ]
        mids_down = [ (hi+lo)/2 for (t,hi,lo,typ) in gaps if typ=="down" and ts>=t and row["close"] > hi ]
        gap_above.append(min(mids_up) if mids_up else np.nan)
        gap_below.append(max(mids_down) if mids_down else np.nan)
    df["cme_mid_above"], df["cme_mid_below"] = gap_above, gap_below

    return df.dropna(subset=["atr"])

# ----------------- Confluence & Rules -----------------
def confluence_signals(df, tol=0.003):
    """
    Create binary signals used in scoring:
      - touch_npoc_reject_[long/short]
      - avwap_support/resistance
      - reenter_value_[long/short] (VAL/VAH)
      - golden_pocket_hit_[long/short]
      - cme_target_[up/down] exists
    """
    s = pd.DataFrame(index=df.index)

    # NPOC touches and reject
    s["touch_npoc_long"]  = (df["low"]  <= df["npoc_below"]*(1+tol)) & (df["close"] > df["npoc_below"])
    s["touch_npoc_short"] = (df["high"] >= df["npoc_above"]*(1-tol)) & (df["close"] < df["npoc_above"])

    # Anchored VWAP support/resistance (from last swing)
    s["avwap_support"]    = (df["low"]  <= df["avwap_from_lo"]*(1+tol)) & (df["close"] > df["avwap_from_lo"])
    s["avwap_resist"]     = (df["high"] >= df["avwap_from_hi"]*(1-tol)) & (df["close"] < df["avwap_from_hi"])

    # Value area re-entries
    s["reenter_value_long"]  = (df["close"].shift(1) < df["val"].shift(1)) & (df["close"] > df["val"])
    s["reenter_value_short"] = (df["close"].shift(1) > df["vah"].shift(1)) & (df["close"] < df["vah"])

    # Golden pocket touches relative to swing direction
    gp_hit = (df["low"] <= df["gp_high"]) & (df["high"] >= df["gp_low"])
    s["golden_long"]  = gp_hit & (df["swing_dir"]==1) & (df["close"] > df["gp_low"])
    s["golden_short"] = gp_hit & (df["swing_dir"]==-1) & (df["close"] < df["gp_high"])

    # CME targets exist
    s["cme_up"]   = df["cme_mid_above"].notna()
    s["cme_down"] = df["cme_mid_below"].notna()

    return s.fillna(False).astype(bool)

def backtest(df, sigs):
    """Daily next-bar execution with ATR stop and target to nearest structural level / CME mid."""
    pos = 0  # -1, 0, +1
    entry_px = stop_px = target_px = None
    rets = []
    for i in range(1, len(df)):
        row_y, row = df.iloc[i-1], df.iloc[i]   # decide using yesterday's info, execute today
        sig_y = sigs.iloc[i-1]
        atr = row_y["atr"]

        # Build confluence scores
        long_score = sum([
            sig_y["touch_npoc_long"], sig_y["avwap_support"],
            sig_y["reenter_value_long"], sig_y["golden_long"],
            sig_y["cme_up"]
        ])
        short_score = sum([
            sig_y["touch_npoc_short"], sig_y["avwap_resist"],
            sig_y["reenter_value_short"], sig_y["golden_short"],
            sig_y["cme_down"]
        ])

        # exits on stop/target
        if pos != 0 and stop_px is not None and target_px is not None:
            # intraday stop/target on today's range (approx with close location)
            hit_stop  = (row["low"] <= stop_px) if pos>0 else (row["high"] >= stop_px)
            hit_tgt   = (row["high"] >= target_px) if pos>0 else (row["low"] <= target_px)
            if hit_stop or hit_tgt:
                exit_px = stop_px if hit_stop else target_px
                pnl = (exit_px/entry_px - 1.0) * pos
                rets.append(pnl - COST)
                pos, entry_px, stop_px, target_px = 0, None, None, None
                continue

        # if flat, consider entries
        if pos == 0:
            if long_score >= ENTRY_SCORE and not np.isnan(row_y["val"]):
                pos = +1
                entry_px = row["open"]  # next bar open
                # Stop a bit beyond VAL or 1.5 ATR below entry
                structural = min(row_y["val"], row_y["npoc_below"]) if not np.isnan(row_y["npoc_below"]) else row_y["val"]
                stop_px = min(structural, entry_px - 1.5*atr)
                # Target: VAH or nearest gap above
                t_candidates = [x for x in [row_y["vah"], row_y["cme_mid_above"]] if not np.isnan(x)]
                target_px = max(t_candidates) if t_candidates else entry_px + 2*atr
            elif short_score >= ENTRY_SCORE and not np.isnan(row_y["vah"]):
                pos = -1
                entry_px = row["open"]
                structural = max(row_y["vah"], row_y["npoc_above"]) if not np.isnan(row_y["npoc_above"]) else row_y["vah"]
                stop_px = max(structural, entry_px + 1.5*atr)
                t_candidates = [x for x in [row_y["val"], row_y["cme_mid_below"]] if not np.isnan(x)]
                target_px = min(t_candidates) if t_candidates else entry_px - 2*atr
            rets.append(0.0)
        else:
            # manage open position: simple daily mark-to-market
            day_ret = (row["close"]/df.iloc[i-1]["close"] - 1.0) * pos
            rets.append(day_ret - COST*abs(pos!=0)*0.0)  # no extra turnover unless flip/close

    r = pd.Series(rets, index=df.index[1:])
    return r

# ----------------- Run -----------------
def run():
    df = build_btc_frame()
    sigs = confluence_signals(df)
    r = backtest(df, sigs)

    def stats(x):
        x = x.dropna()
        ann = 252
        eq = (1+x).cumprod()
        out = {
            "Total Return": float(eq.iloc[-1]-1),
            "CAGR": float((1+x).prod()**(ann/len(x))-1) if len(x)>0 else np.nan,
            "Sharpe": float(np.sqrt(ann)*x.mean()/(x.std()+1e-12)) if x.std()>0 else np.nan,
            "MaxDD": float((eq/eq.cummax()-1).min()) if len(eq)>0 else np.nan
        }
        return pd.Series(out)

    ins = r.loc[:pd.Timestamp(SPLIT)-pd.Timedelta(days=1)]
    oos = r.loc[pd.Timestamp(SPLIT):]

    print(f"Rows: {len(df):,}  Start: {df.index[0].date()}  End: {df.index[-1].date()}")
    print("\n=== BTC Confluence Strategy ===")
    print("In-sample (< {0}):".format(SPLIT));  print(stats(ins).to_frame("Value"))
    print("\nOut-of-sample (≥ {0}):".format(SPLIT)); print(stats(oos).to_frame("Value"))

    # Save equity + signals
    equity = (1+r).cumprod()
    equity.to_frame("Equity").to_csv("btc_confluence_equity.csv")
    sigs.to_csv("btc_confluence_signals.csv")
    df.to_csv("btc_confluence_features.csv")
    print("\nSaved: btc_confluence_equity.csv, btc_confluence_signals.csv, btc_confluence_features.csv")

if __name__ == "__main__":
    run()
