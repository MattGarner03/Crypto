"""
Fetch OHLCV history for the top spot coins on Binance across multiple intervals.

This script:
1. Queries CoinGecko for the top market-cap crypto assets (excludes stables).
2. Maps each asset to a preferred Binance spot pair (USDT first, then FDUSD, BUSD, USDC).
3. Downloads full OHLCV history via Binance's public klines API for the requested intervals.
4. Saves each asset's history to ./data_{tf}/{symbol}.csv (e.g., data/BTCUSDT.csv, data_1h/BTCUSDT.csv).
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import requests

BASE_URL = "https://api.binance.com"
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
DATA_ROOT = Path(__file__).resolve().parent
PREFERRED_QUOTES: tuple[str, ...] = ("USDT", "FDUSD", "BUSD", "USDC")
STABLE_SYMBOLS: set[str] = {"usdt", "usdc", "busd", "tusd", "usdp", "usdd", "dai", "gusd"}

TIMEFRAME_CONFIG = {
    "1d": {"interval": "1d", "dir": DATA_ROOT / "data"},
    "1h": {"interval": "1h", "dir": DATA_ROOT / "data_1h"},
    "4h": {"interval": "4h", "dir": DATA_ROOT / "data_4h"},
}


def _to_millis(iso_date: str | None) -> int | None:
    if not iso_date:
        return None
    dt = datetime.fromisoformat(iso_date).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _binance_pairs() -> dict[str, list[dict]]:
    resp = requests.get(f"{BASE_URL}/api/v3/exchangeInfo", timeout=30)
    resp.raise_for_status()
    symbols = resp.json().get("symbols", [])
    by_base: dict[str, list[dict]] = defaultdict(list)
    for sym in symbols:
        if sym.get("status") != "TRADING":
            continue
        by_base[sym["baseAsset"]].append(sym)
    return by_base


def _top_market_cap_assets(limit: int = 10, oversample: int = 10) -> list[dict]:
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": limit + oversample,
        "page": 1,
        "sparkline": "false",
    }
    resp = requests.get(COINGECKO_URL, params=params, timeout=30)
    resp.raise_for_status()
    coins = resp.json()
    selected: list[dict] = []
    for coin in coins:
        sym = coin.get("symbol", "").lower()
        if sym in STABLE_SYMBOLS:
            continue
        selected.append(coin)
        if len(selected) >= limit:
            break
    return selected


def _map_to_binance_symbol(base_asset: str, listings: dict[str, list[dict]]) -> str | None:
    candidates = listings.get(base_asset.upper())
    if not candidates:
        return None
    for quote in PREFERRED_QUOTES:
        for entry in candidates:
            if entry.get("quoteAsset") == quote and entry.get("status") == "TRADING":
                return entry["symbol"]
    # fallback to the first trading pair if preferred quotes unavailable
    return candidates[0]["symbol"] if candidates else None


def _fetch_ohlcv(binance_symbol: str, interval: str, start: str | None = None) -> pd.DataFrame:
    start_ms = _to_millis(start) if start else None
    limit = 1000
    cursor = start_ms
    rows: list[list] = []
    while True:
        params = {"symbol": binance_symbol, "interval": interval, "limit": limit}
        if cursor is not None:
            params["startTime"] = cursor
        resp = requests.get(f"{BASE_URL}/api/v3/klines", params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][6] + 1  # next open_time starts after last close_time
        if len(batch) < limit:
            break
        time.sleep(0.2)  # avoid hitting rate limits

    if not rows:
        return pd.DataFrame()

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["trades"] = df["trades"].astype(int)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def fetch_and_store(
    limit: int = 10,
    start: str = "2017-01-01",
    timeframes: Sequence[str] = ("1d", "1h", "4h"),
) -> dict[tuple[str, str], Path]:
    for tf in timeframes:
        cfg = TIMEFRAME_CONFIG.get(tf)
        if not cfg:
            raise ValueError(f"Unsupported timeframe '{tf}'. Supported: {', '.join(TIMEFRAME_CONFIG)}")
        cfg["dir"].mkdir(parents=True, exist_ok=True)

    listings = _binance_pairs()
    top_assets = _top_market_cap_assets(limit=limit)
    saved: dict[tuple[str, str], Path] = {}
    for asset in top_assets:
        base = asset["symbol"].upper()
        symbol = _map_to_binance_symbol(base, listings)
        if not symbol:
            print(f"[skip] No Binance spot symbol found for {base}")
            continue
        for tf in timeframes:
            cfg = TIMEFRAME_CONFIG[tf]
            print(f"[info] Fetching {symbol} interval={cfg['interval']} …")
            df = _fetch_ohlcv(symbol, cfg["interval"], start=start)
            if df.empty:
                print(f"[warn] No data returned for {symbol} @ {cfg['interval']}, skipping.")
                continue
            out_path = cfg["dir"] / f"{symbol}.csv"
            df.to_csv(out_path, index=False)
            saved[(symbol, tf)] = out_path
            print(f"[done] Saved {symbol} ({tf}) -> {out_path}")
    return saved


def main() -> None:
    saved = fetch_and_store()
    if not saved:
        print("No datasets were saved.")
    else:
        print("Completed. Saved datasets:")
        for (symbol, tf), path in saved.items():
            print(f"  {symbol} [{tf}]: {path}")


if __name__ == "__main__":
    main()
