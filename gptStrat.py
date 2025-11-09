# top2_relmom_cash.py
# deps: pip install -U yfinance pandas numpy matplotlib

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

START = "2004-01-01"
SPLIT = "2015-01-01"
END   = None

INIT_EQUITY = 100_000

FEE_BPS   = 5e-4          # 5 bps per change
SLIP_BPS  = 5e-4
COST_PER_CHANGE = FEE_BPS + SLIP_BPS

ANN = 252
MOM_WINDOW = 6            # months
SKIP_1M = 1               # exclude most recent month

RISK_TICKERS = ["SPY", "QQQ", "EFA", "EEM"]
DEF_TICKER   = "SHY"      # cash as defensive
BENCH_6040_DEF = "IEF"    # for 60/40 benchmark

# ---------- Helpers ----------
def get_closes(tickers, start, end, auto_adjust=True) -> pd.DataFrame:
    """Return tz-naive adjusted Close DataFrame with columns in requested order."""
    df = yf.download(" ".join(tickers), start=start, end=end, auto_adjust=auto_adjust, progress=False)
    if df.empty:
        raise RuntimeError("yfinance returned empty data.")
    if isinstance(df.columns, pd.MultiIndex):
        lev0 = list(map(str, df.columns.get_level_values(0)))
        lev1 = list(map(str, df.columns.get_level_values(1)))
        if "Close" in lev1:      # (Ticker, Field)
            px = df.xs("Close", axis=1, level=1)
        elif "Close" in lev0:    # (Field, Ticker)
            px = df.xs("Close", axis=1, level=0)
        else:
            cands = [c for c in df.columns if isinstance(c, tuple) and any(str(x).lower()=="close" for x in c)]
            if not cands:
                raise KeyError("No 'Close' in columns.")
            px = df[cands]
            new_cols = []
            for c in px.columns:
                new_cols.append(str(c[1]) if str(c[0]).lower()=="close" else str(c[0]))
            px.columns = new_cols
    else:
        ckey = next((c for c in df.columns if str(c).lower()=="close"), None)
        if ckey is None:
            raise KeyError(f"No 'Close' column in {list(df.columns)}")
        px = df[[ckey]]
        px.columns = [tickers[0]]

    if getattr(px.index, "tz", None) is not None:
        px.index = px.index.tz_localize(None)
    cols_present = [t for t in tickers if t in px.columns]
    return px[cols_present].dropna(how="all")

def stats_from_returns(r: pd.Series) -> pd.Series:
    r = r.dropna()
    if r.empty or r.std() == 0:
        return pd.Series({"Total Return": np.nan, "CAGR": np.nan, "Sharpe": np.nan, "MaxDD": np.nan})
    eq = (1 + r).cumprod()
    return pd.Series({
        "Total Return": float(eq.iloc[-1] - 1),
        "CAGR": float((1 + r).prod() ** (ANN / len(r)) - 1),
        "Sharpe": float(np.sqrt(ANN) * r.mean() / (r.std() + 1e-12)),
        "MaxDD": float((eq / eq.cummax() - 1).min())
    })

def compute_drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0

# ---------- Strategy ----------
def month_end_choices(px: pd.DataFrame) -> pd.DataFrame:
    """
    On each month-end:
      - compute 6-1 momentum for risk + SHY,
      - pick up to top-2 risk assets whose momentum > SHY,
      - return a monthly weight DataFrame (sum to 1.0; if none -> 100% SHY).
    """
    px_m = px.resample("M").last()

    # 6-month momentum excluding the last month: pct_change(6) then shift(1)
    mom6 = px_m.pct_change(MOM_WINDOW).shift(SKIP_1M)

    risk_mom = mom6[RISK_TICKERS]
    shy_mom  = mom6[DEF_TICKER]

    w_m = pd.DataFrame(0.0, index=px_m.index, columns=px.columns)
    for dt in px_m.index:
        if pd.isna(shy_mom.get(dt)):
            continue
        rm = risk_mom.loc[dt].dropna()
        if rm.empty:
            continue
        # Filter risk assets that beat SHY
        winners = rm[rm > shy_mom.loc[dt]]
        if len(winners) == 0:
            w_m.loc[dt, DEF_TICKER] = 1.0
        else:
            top2 = winners.sort_values(ascending=False).index[:2]
            w_m.loc[dt, top2] = 1.0 / len(top2)
    return w_m

def expand_to_daily(px: pd.DataFrame, w_m: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill monthly weights to daily and shift 1 day for next-day execution."""
    w_d = w_m.reindex(px.resample("D").last().index).ffill()
    w_d = w_d.reindex(px.index).ffill()
    return w_d.shift(1).fillna(0.0)

def backtest_top2_relmom_cash(px: pd.DataFrame):
    w_m = month_end_choices(px)
    w_d = expand_to_daily(px, w_m)
    ret_d = px.pct_change().fillna(0.0)
    gross = (w_d * ret_d).sum(axis=1)
    turnover = w_d.diff().abs().sum(axis=1).fillna(w_d.sum(axis=1))
    net = gross - turnover * COST_PER_CHANGE
    return net, w_m, w_d

def daily_rebalanced_benchmark(px: pd.DataFrame, w: dict) -> pd.Series:
    ret = px[list(w.keys())].pct_change().fillna(0.0)
    return (ret * pd.Series(w)).sum(axis=1)

def print_block(title, r: pd.Series, split):
    ins = r.loc[:pd.Timestamp(split) - pd.Timedelta(days=1)]
    oos = r.loc[pd.Timestamp(split):]
    print(f"\n=== {title} ===")
    print("In-sample (< {0}):".format(split))
    print(stats_from_returns(ins).to_frame("Value"))
    print("\nOut-of-sample (≥ {0}):".format(split))
    print(stats_from_returns(oos).to_frame("Value"))

def main():
    tickers = RISK_TICKERS + [DEF_TICKER, BENCH_6040_DEF]
    px = get_closes(tickers, START, END, auto_adjust=True)

    strat_ret, w_m, w_d = backtest_top2_relmom_cash(px)
    spy_ret  = px["SPY"].pct_change().fillna(0.0)
    bench6040 = daily_rebalanced_benchmark(px, {"SPY": 0.60, BENCH_6040_DEF: 0.40})

    # Align all series
    idx = strat_ret.index
    spy_ret = spy_ret.reindex(idx).fillna(0.0)
    bench6040 = bench6040.reindex(idx).fillna(0.0)

    print(f"yfinance version: {yf.__version__}")
    print(f"Rows: {len(px):,} | Start: {px.index[0].date()} | End: {px.index[-1].date()}")

    print_block("Top-2 Relative Momentum (6-1, cash fallback)", strat_ret, SPLIT)
    print_block("SPY buy & hold", spy_ret, SPLIT)
    print_block("60/40 SPY/IEF (daily rebalanced)", bench6040, SPLIT)

    # Equity & drawdowns
    strat_eq = (1 + strat_ret).cumprod() * INIT_EQUITY
    spy_eq   = (1 + spy_ret).cumprod() * INIT_EQUITY
    b6040_eq = (1 + bench6040).cumprod() * INIT_EQUITY

    strat_dd = compute_drawdown(strat_eq)
    spy_dd   = compute_drawdown(spy_eq)
    b6040_dd = compute_drawdown(b6040_eq)

    # Plot: equity (log) + OOS shading
    split_dt = pd.Timestamp(SPLIT)
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    strat_eq.plot(ax=ax1, label="Top-2 RelMom (6-1, cash)")
    spy_eq.plot(ax=ax1, label="SPY")
    b6040_eq.plot(ax=ax1, label="60/40 SPY/IEF")
    ax1.set_yscale("log")
    ax1.set_title("Equity Curves (Log Scale)")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Equity (log)")
    ax1.axvspan(split_dt, strat_eq.index[-1], alpha=0.15)
    ax1.legend()
    fig1.tight_layout(); fig1.savefig("top2_relmom_equity.png", dpi=150)

    # Plot: drawdowns
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    strat_dd.plot(ax=ax2, label="Top-2 RelMom (6-1, cash)")
    spy_dd.plot(ax=ax2, label="SPY")
    b6040_dd.plot(ax=ax2, label="60/40 SPY/IEF")
    ax2.axhline(0, linewidth=1)
    ax2.set_title("Drawdowns")
    ax2.set_xlabel("Date"); ax2.set_ylabel("Drawdown")
    ax2.axvspan(split_dt, strat_dd.index[-1], alpha=0.15)
    ax2.legend()
    fig2.tight_layout(); fig2.savefig("top2_relmom_drawdown.png", dpi=150)

    # Save logs
    strat_eq.to_frame("Equity").to_csv("top2_relmom_equity.csv")
    spy_eq.to_frame("SPY_Equity").to_csv("spy_equity.csv")
    b6040_eq.to_frame("Bench_60_40_Equity").to_csv("bench_60_40_equity.csv")
    w_m.to_csv("top2_relmom_alloc_monthly.csv")
    w_d.to_csv("top2_relmom_weights_daily.csv")

    print("\nSaved:")
    print(" - top2_relmom_equity.png")
    print(" - top2_relmom_drawdown.png")
    print(" - top2_relmom_equity.csv")
    print(" - spy_equity.csv")
    print(" - bench_60_40_equity.csv")
    print(" - top2_relmom_alloc_monthly.csv")
    print(" - top2_relmom_weights_daily.csv")

if __name__ == "__main__":
    main()
