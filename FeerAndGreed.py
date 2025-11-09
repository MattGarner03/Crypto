#!/usr/bin/env python3
"""
Download Crypto Fear & Greed Index from:
  1) CoinMarketCap (Pro API, needs key)
  2) Alternative.me (free)
Outputs:
  - cmc_fear_greed.csv
  - alt_fear_greed.csv
  - fear_greed_merged.csv (aligned by date)
"""

import csv
import time
import requests
import datetime as dt
from collections import defaultdict

# ---------------- Settings ----------------
CMC_API_KEY = "d6c4212cfd1a4e2cb7b39c29349edb4d"  # <-- your key (as requested)
CMC_URL = "https://pro-api.coinmarketcap.com/v3/fear-and-greed/historical"
ALT_URL = "https://api.alternative.me/fng/"
PAGE_LIMIT = 100         # CMC returns up to 100 rows per request
BACKOFF_SECONDS = 0.25   # be polite to APIs
# ------------------------------------------


def iso_and_date_from_ts(ts_raw):
    """
    Convert a raw timestamp (unix seconds or ISO string) to:
      - iso string (UTC, with 'Z')
      - date string YYYY-MM-DD
    """
    if ts_raw is None:
        return "", ""
    # try UNIX seconds first
    try:
        ts_int = int(ts_raw)
        dt_ = dt.datetime.utcfromtimestamp(ts_int)
        return dt_.isoformat() + "Z", dt_.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        pass
    # assume already ISO-8601 string
    if isinstance(ts_raw, str):
        date_only = ts_raw[:10] if len(ts_raw) >= 10 else ""
        # normalise trailing Z
        iso = ts_raw if ts_raw.endswith("Z") else ts_raw + ("Z" if "T" in ts_raw else "")
        return iso, date_only
    return "", ""


def fetch_all_cmc():
    """Fetch all pages from the CMC Fear & Greed historical endpoint."""
    headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
    start = 1
    all_rows = []

    while True:
        params = {"start": start, "limit": PAGE_LIMIT}
        r = requests.get(CMC_URL, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        data = payload.get("data", [])
        if not data:
            break
        all_rows.extend(data)
        start += PAGE_LIMIT
        time.sleep(BACKOFF_SECONDS)

    # Normalise and sort by date ascending
    norm = []
    for d in all_rows:
        iso_ts, ymd = iso_and_date_from_ts(d.get("timestamp"))
        norm.append({
            "timestamp": iso_ts,
            "date": ymd,
            "value": d.get("value"),
            "classification": d.get("value_classification"),
            "source": "coinmarketcap",
        })
    norm.sort(key=lambda x: x["date"])
    return norm


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def fetch_all_alt():
    """Fetch full Alternative.me history (limit=0)."""
    r = requests.get(ALT_URL, params={"limit": 0}, timeout=30)
    r.raise_for_status()
    js = r.json()
    data = js.get("data", [])

    norm = []
    for d in data:
        iso_ts, ymd = iso_and_date_from_ts(d.get("timestamp"))
        # Alternative.me returns 'value' as string
        try:
            val = int(d.get("value"))
        except (TypeError, ValueError):
            val = d.get("value")
        norm.append({
            "timestamp": iso_ts,
            "date": ymd,
            "value": val,
            "classification": d.get("value_classification"),
            "source": "alternative.me",
        })
    norm.sort(key=lambda x: x["date"])
    return norm


def merge_by_date(cmc_rows, alt_rows):
    """
    Merge into a single table keyed by date:
      date, alt_value, alt_classification, cmc_value, cmc_classification
    """
    by_date = defaultdict(dict)
    for r in alt_rows:
        by_date[r["date"]].update({
            "date": r["date"],
            "alt_value": r["value"],
            "alt_classification": r["classification"],
        })
    for r in cmc_rows:
        by_date[r["date"]].update({
            "date": r["date"],
            "cmc_value": r["value"],
            "cmc_classification": r["classification"],
        })

    merged = list(by_date.values())
    merged.sort(key=lambda x: x["date"])
    return merged


def main():
    print("Fetching Alternative.me ...")
    alt_rows = fetch_all_alt()
    save_csv(
        "alt_fear_greed.csv",
        alt_rows,
        ["timestamp", "date", "value", "classification", "source"],
    )
    print(f"[ALT] Saved {len(alt_rows)} rows -> alt_fear_greed.csv")

    print("Fetching CoinMarketCap ...")
    cmc_rows = fetch_all_cmc()
    save_csv(
        "cmc_fear_greed.csv",
        cmc_rows,
        ["timestamp", "date", "value", "classification", "source"],
    )
    print(f"[CMC] Saved {len(cmc_rows)} rows -> cmc_fear_greed.csv")

    print("Merging by date ...")
    merged = merge_by_date(cmc_rows, alt_rows)
    save_csv(
        "fear_greed_merged.csv",
        merged,
        ["date", "alt_value", "alt_classification", "cmc_value", "cmc_classification"],
    )
    print(f"[MERGE] Saved {len(merged)} rows -> fear_greed_merged.csv")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
