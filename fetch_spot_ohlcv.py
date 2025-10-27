#!/usr/bin/env python3
"""Download hourly OHLCV data for top Binance spot symbols."""

from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping

import requests

BASE_URL = "https://api.binance.com"
INTERVAL = "1h"
TOP_LIMIT = 50
QUOTE_ASSET = "USDT"
DATA_DIR = "spot_data"
KLINE_LIMIT = 1000
REQUEST_TIMEOUT = 20
MAX_RETRIES = 5
BACKOFF_SECONDS = 1.5


session = requests.Session()
session.headers.update({"User-Agent": "Binance-OHLCV-Fetcher/1.0"})


def binance_get(path: str, params: Mapping[str, object]) -> Any:
    """Execute a GET request with basic retry and rate-limit handling."""
    url = f"{BASE_URL}{path}"
    for attempt in range(MAX_RETRIES):
        response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return response.json()

        if response.status_code in (418, 429):
            # Hit the rate limit; back off before retrying.
            sleep_seconds = BACKOFF_SECONDS * (attempt + 1)
            time.sleep(sleep_seconds)
            continue

        response.raise_for_status()

    raise RuntimeError(f"Exceeded retry limit for {path} with params {params}")


def is_spot_symbol(meta: Mapping[str, Any]) -> bool:
    """Return True if the symbol metadata indicates spot trading is allowed."""
    if meta.get("status") != "TRADING":
        return False

    if meta.get("isSpotTradingAllowed"):
        return True

    permissions = meta.get("permissions") or []
    if "SPOT" in permissions:
        return True

    for permission_set in meta.get("permissionSets") or []:
        if "SPOT" in permission_set:
            return True

    return False


def get_tradable_spot_symbols() -> Dict[str, Dict[str, Any]]:
    """Return exchange metadata for tradable spot symbols."""
    data = binance_get("/api/v3/exchangeInfo", params={})
    symbols = {
        symbol["symbol"]: symbol
        for symbol in data["symbols"]
        if is_spot_symbol(symbol)
    }
    if not symbols:
        raise RuntimeError("No tradable spot symbols found in exchange info.")
    return symbols


def get_top_spot_symbols(limit: int) -> List[str]:
    """Select top spot symbols quoted in the configured quote asset."""
    symbol_meta = get_tradable_spot_symbols()
    tickers = binance_get("/api/v3/ticker/24hr", params={})

    ranked: List[Dict[str, Any]] = []
    for ticker in tickers:
        symbol = ticker["symbol"]
        meta = symbol_meta.get(symbol)
        if not meta:
            continue
        if meta.get("quoteAsset") != QUOTE_ASSET:
            continue
        if any(keyword in symbol for keyword in ("UP", "DOWN", "BEAR", "BULL")):
            # Exclude leveraged tokens that sometimes slip into the list.
            continue

        try:
            quote_volume = float(ticker["quoteVolume"])
        except (KeyError, TypeError, ValueError):
            continue

        ranked.append({"symbol": symbol, "quote_volume": quote_volume})

    ranked.sort(key=lambda entry: entry["quote_volume"], reverse=True)
    top_symbols = [entry["symbol"] for entry in ranked[:limit]]
    if len(top_symbols) < limit:
        print(
            f"Warning: Requested top {limit} symbols but only found {len(top_symbols)}.",
            file=sys.stderr,
        )
    return top_symbols


def iter_klines(symbol: str) -> Iterable[List[Any]]:
    """Yield pages of klines for a symbol starting from the earliest data point."""
    start_time = 0
    while True:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": start_time,
            "limit": KLINE_LIMIT,
        }
        data = binance_get("/api/v3/klines", params=params)
        if not data:
            break

        yield data

        last_open_time = data[-1][0]
        # Move one millisecond past the last open time to avoid duplicates.
        start_time = last_open_time + 1

        if len(data) < KLINE_LIMIT:
            break

        time.sleep(0.5)


def write_klines(symbol: str) -> None:
    """Download all klines for a symbol and write them to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, f"{symbol}.csv")
    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])

        row_count = 0
        for batch in iter_klines(symbol):
            for entry in batch:
                timestamp = datetime.utcfromtimestamp(entry[0] / 1000).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                writer.writerow(
                    [timestamp, entry[1], entry[2], entry[3], entry[4], entry[5]]
                )
                row_count += 1

        print(f"{symbol}: wrote {row_count} rows to {filepath}")


def main() -> None:
    try:
        symbols = get_top_spot_symbols(TOP_LIMIT)
    except Exception as exc:
        print(f"Failed to load symbol list: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching hourly OHLCV for {len(symbols)} symbols...")
    for index, symbol in enumerate(symbols, start=1):
        print(f"[{index}/{len(symbols)}] {symbol}")
        try:
            write_klines(symbol)
        except Exception as exc:
            print(f"  Error fetching {symbol}: {exc}", file=sys.stderr)
            time.sleep(2)


if __name__ == "__main__":
    main()
