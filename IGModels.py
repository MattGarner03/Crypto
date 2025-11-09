# IGModels.py
# deps: pip install -U yfinance pandas numpy
import numpy as np
import pandas as pd
import yfinance as yf

START = "2004-01-01"
SPLIT = "2015-01-01"   # out-of-sample starts this date (inclusive)
END   = None           # to latest
INIT_EQUITY = 100_000

FEE_BPS   = 5e-4       # 5 bps per change
SLIP_BPS  = 5e-4       # 5 bps per change
COST_PER_CHANGE = FEE_BPS + SLIP_BPS

ANN = 252              # trading days/year
TSMOM_TARGET_VOL = 0.10
TSMOM_VOL_WIN = 20     # rolling daily stdev window
SKIP_M = 21            # ~1 trading month

# ---------- Helpers ----------
def get_close(ticker: str, start: str, end: str | None, auto_adjust: bool = True) -> pd.Series:
    """
    Return tz-naive adjusted Close as 1D Series.
    yfinance>=0.2.43 sets auto_adjust=True by default, making 'Close' already adjusted.
    Handles single-level and MultiIndex column shapes.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=auto_adjust, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned empty frame for {ticker}")

    cols = df.columns
    if not isinstance(cols, pd.MultiIndex):
        # case-insensitive 'Close'
        key = next((c for c in cols if str(c).lower() == "close"), None)
        if key is None:
            raise KeyError(f"No 'Close' in columns {list(cols)} for {ticker}")
        s = df[key]
    else:
        # MultiIndex can be (field,ticker) or (ticker,field)
        lev0 = list(map(str, cols.get_level_values(0)))
        lev1 = list(map(str, cols.get_level_values(1)))
        if "Close" in lev0 and ticker in lev1:
            s = df.loc[:, ("Close", ticker)]
        elif "Close" in lev1 and ticker in lev0:
            s = df.loc[:, (ticker, "Close")]
        else:
            cands = [c for c in cols if isinstance(c, tuple) and any(str(x).lower()=="close" for x in c)]
            if not cands:
                raise KeyError(f"No 'Close' field in MultiIndex columns for {ticker}: {list(cols)}")
            s = df[cands[0]]

    s = s.squeeze()
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    s.name = ticker
    return s.dropna()


def stats_from_returns(r: pd.Series) -> pd.Series:
    r = r.dropna()
    if r.empty or r.std() == 0:
        return pd.Series({"Total Return": np.nan, "CAGR": np.nan, "Sharpe": np.nan, "MaxDD": np.nan})
    eq = (1 + r).cumprod()
    total = float(eq.iloc[-1] - 1)
    cagr = float((1 + r).prod() ** (ANN / len(r)) - 1)
    sharpe = float(np.sqrt(ANN) * r.mean() / (r.std() + 1e-12))
    dd = float((eq / eq.cummax() - 1).min())
    return pd.Series({"Total Return": total, "CAGR": cagr, "Sharpe": sharpe, "MaxDD": dd})


def month_markers(idx: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Boolean Series (indexed by idx) for first/last trading day of each month. Pandas 2.2 safe."""
    period = idx.to_period("M")
    first_arr = ~np.asarray(period.duplicated())
    first = pd.Series(first_arr, index=idx)

    next_period = period.shift(-1)
    last_arr = np.asarray(period != next_period)
    if len(last_arr) > 0:
        last_arr[-1] = True  # final index is last of its month segment
    last = pd.Series(last_arr, index=idx)
    return first, last


def build_hold_mask(entries: pd.Series, exits: pd.Series) -> pd.Series:
    """Stateful expansion from entry to exit (inclusive). Non-overlapping assumed."""
    e = entries.fillna(False).to_numpy()
    x = exits.fillna(False).to_numpy()
    hold = np.zeros(len(e), dtype=bool)
    on = False
    for i in range(len(e)):
        if e[i]:
            on = True
        hold[i] = on
        if x[i]:
            on = False
    return pd.Series(hold, index=entries.index)


# ---------- Strategy A: TLT Calendar (returns-only) ----------
def calendar_returns(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Short on first 5 trading days of each month; long from 7 trading days before month-end to last day.
    Returns:
      strat_ret (Series), weight (Series in {-1,0,+1} lagged for next-day trading)
    """
    idx = close.index
    first, last = month_markers(idx)

    short_entries = first
    short_exits   = first.shift(5, fill_value=False)

    long_entries  = last.shift(-6, fill_value=False)  # 7 trading days incl. entry
    long_exits    = last

    short_hold = build_hold_mask(short_entries, short_exits)
    long_hold  = build_hold_mask(long_entries,  long_exits)

    weight_raw = long_hold.astype(int) - short_hold.astype(int)  # +1 long, -1 short, 0 cash
    weight = weight_raw.shift(1).fillna(0.0)  # trade next day (no look-ahead)

    ret = close.pct_change().fillna(0.0)
    turnover = weight.diff().abs().fillna(weight.abs())  # initial build + changes
    strat_ret = (weight * ret) - turnover * COST_PER_CHANGE
    return strat_ret, weight


# ---------- Strategy B: Single-asset TSMOM (12-1 & 6-1) w/ 10% vol targeting (returns-only) ----------
def tsmom_returns(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    ret = close.pct_change().fillna(0.0)

    m12 = close.pct_change(252) - close.pct_change(SKIP_M)
    m06 = close.pct_change(126) - close.pct_change(SKIP_M)
    signal = np.sign((m12.fillna(0.0) + m06.fillna(0.0))).shift(1).fillna(0.0)  # trade next day

    daily_vol = ret.rolling(TSMOM_VOL_WIN).std()
    ann_vol = daily_vol * np.sqrt(ANN)
    weight = (signal * (TSMOM_TARGET_VOL / (ann_vol + 1e-12))).clip(-1.0, 1.0).fillna(0.0)

    turnover = weight.diff().abs().fillna(weight.abs())
    strat_ret = (weight * ret) - turnover * COST_PER_CHANGE
    return strat_ret, weight


def print_block(title: str, ins: pd.Series, oos: pd.Series):
    ins_stats = stats_from_returns(ins)
    oos_stats = stats_from_returns(oos)
    print(f"\n=== {title} ===")
    print(f"In-sample (< {SPLIT}):")
    print(ins_stats.to_frame("Value"))
    print(f"\nOut-of-sample (≥ {SPLIT}):")
    print(oos_stats.to_frame("Value"))


def main():
    print(f"yfinance version: {yf.__version__}")

    # Data
    tlt  = get_close("TLT", START, END, auto_adjust=True)
    aapl = get_close("AAPL", START, END, auto_adjust=True)
    print(f"TLT rows: {len(tlt):,}, AAPL rows: {len(aapl):,}")

    # Strategies (returns-only)
    cal_ret, cal_w = calendar_returns(tlt)
    tsm_ret, tsm_w = tsmom_returns(aapl)

    # Split
    ins = slice(START, SPLIT)
    oos = slice(SPLIT, END)

    # Stats
    print_block("TLT Calendar (returns-only)", cal_ret.loc[ins],  cal_ret.loc[oos])
    print_block("AAPL TSMOM (12-1 & 6-1, 10% vol tgt)", tsm_ret.loc[ins], tsm_ret.loc[oos])

    # Save equity curves
    (1 + cal_ret).cumprod().mul(INIT_EQUITY).to_frame("Calendar_Equity").to_csv("calendar_equity.csv")
    (1 + tsm_ret).cumprod().mul(INIT_EQUITY).to_frame("TSMOM_Equity").to_csv("tsmom_equity.csv")
    print("\nSaved: calendar_equity.csv, tsmom_equity.csv")

if __name__ == "__main__":
    main()
