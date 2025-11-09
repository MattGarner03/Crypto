#!/usr/bin/env python3
"""
Sentiment downloader for:
- Alternative.me Fear & Greed (free)
- CoinMarketCap Fear & Greed (uses your key)
- CNN Stock Market Fear & Greed (no key)
- Santiment Weighted Sentiment (uses key)
- Augmento Bull & Bear Index (uses key)

Outputs CSVs alongside a merged-by-date file for F&G.
"""

import csv
import time
import random
import requests
import datetime as dt
from collections import defaultdict

# ====== YOUR KEYS ======
CMC_API_KEY = "d6c4212cfd1a4e2cb7b39c29349edb4d"
AUGMENTO_API_KEY = "PASTE_AUGMENTO_API_KEY_HERE"
SANTIMENT_API_KEY = "phe5zyohhzm2glyq_nxj26byqfiixogye"
LUNARCRUSH_API_KEY = ""  # not used
# =======================

# ==== Endpoints ====
ALT_URL = "https://api.alternative.me/fng/"
CMC_URL = "https://pro-api.coinmarketcap.com/v3/fear-and-greed/historical"
CNN_FGI_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
SANTIMENT_URL = "https://api.santiment.net/graphql"
AUG_BASE = "https://api.augmento.ai/v0.1"
# ===================


# ---------- Utilities ----------
def iso_and_date(ts_raw):
    """Return (ISO-8601 Z, YYYY-MM-DD) from unix seconds or ISO string."""
    if ts_raw is None:
        return "", ""
    try:
        t = int(ts_raw)
        d = dt.datetime.fromtimestamp(t, dt.UTC)
        return d.isoformat().replace("+00:00", "Z"), d.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        if isinstance(ts_raw, str):
            date_only = ts_raw[:10] if len(ts_raw) >= 10 else ""
            iso = ts_raw
            if "T" in iso and not iso.endswith("Z") and "+00:00" not in iso:
                iso = iso + "Z"
            return iso, date_only
    return "", ""


def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


# ---------- Alternative.me ----------
def fetch_alt():
    r = requests.get(ALT_URL, params={"limit": 0}, timeout=30)
    r.raise_for_status()
    data = (r.json() or {}).get("data", [])
    out = []
    for d in data:
        ts_iso, ymd = iso_and_date(d.get("timestamp"))
        val = int(d.get("value", 0))
        out.append({
            "timestamp": ts_iso,
            "date": ymd,
            "value": val,
            "classification": d.get("value_classification"),
            "source": "alternative.me"
        })
    out.sort(key=lambda x: x["date"])
    return out


# ---------- CoinMarketCap ----------
def fetch_cmc():
    start, limit, out = 1, 100, []
    headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
    while True:
        r = requests.get(CMC_URL, params={"start": start, "limit": limit},
                         headers=headers, timeout=30)
        r.raise_for_status()
        data = (r.json() or {}).get("data", [])
        if not data:
            break
        for d in data:
            ts_iso, ymd = iso_and_date(d.get("timestamp"))
            out.append({
                "timestamp": ts_iso,
                "date": ymd,
                "value": d.get("value"),
                "classification": d.get("value_classification"),
                "source": "coinmarketcap"
            })
        start += limit
        time.sleep(0.25)
    out.sort(key=lambda x: x["date"])
    return out


# ---------- CNN FGI (stock market) ----------
_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
]


def _get_json_with_retries(url, params=None, max_tries=5, base_sleep=0.8):
    last_err = None
    for i in range(max_tries):
        try:
            headers = {
                "User-Agent": random.choice(_USER_AGENTS),
                "Accept": "application/json,text/plain,*/*",
                "Referer": "https://money.cnn.com/data/fear-and-greed/",
                "Cache-Control": "no-cache",
            }
            params = {**(params or {}), "_": str(int(time.time() * 1000))}
            r = requests.get(url, headers=headers, params=params, timeout=20)
            if r.status_code in (418, 403, 429, 503):
                raise requests.HTTPError(f"{r.status_code} from {url}", response=r)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(base_sleep * (2 ** i) + random.random() * 0.25)
    raise last_err


def fetch_cnn_fgi(start_date_iso=None):
    try:
        js = _get_json_with_retries(CNN_FGI_URL)
    except Exception:
        if not start_date_iso:
            start_date_iso = "2018-01-01"
        js = _get_json_with_retries(f"{CNN_FGI_URL}/{start_date_iso}")

    hist = js.get("fear_and_greed_historical") or js.get("fear_and_greed_historical_values") or []
    out = []
    for p in hist:
        ts_iso, ymd = iso_and_date(p.get("x"))
        out.append({"timestamp": ts_iso, "date": ymd, "value": p.get("y"), "source": "cnn_fgi"})
    out.sort(key=lambda x: x["date"])
    return out


# ---------- Santiment ----------
def fetch_santiment_weighted_sentiment(slug="bitcoin",
                                       start_iso="2018-01-01T00:00:00Z",
                                       end_iso=None,
                                       interval="1d"):
    if not SANTIMENT_API_KEY:
        print("[SANTIMENT] Skipped (no key)")
        return []
    if not end_iso:
        end_iso = dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")

    query = """
    query WeightedSentiment($slug: String!, $from: DateTime!, $to: DateTime!, $interval: String!) {
      getMetric(metric: "sentiment_weighted_1d") {
        timeseriesData(slug: $slug, from: $from, to: $to, interval: $interval) {
          datetime
          value
        }
      }
    }
    """
    variables = {"slug": slug, "from": start_iso, "to": end_iso, "interval": interval}
    headers = {"Authorization": f"Apikey {SANTIMENT_API_KEY}"}
    r = requests.post(SANTIMENT_URL, json={"query": query, "variables": variables}, headers=headers, timeout=60)
    r.raise_for_status()
    points = (r.json().get("data") or {}).get("getMetric", {}).get("timeseriesData", [])
    out = []
    for p in points:
        ts_iso, ymd = iso_and_date(p["datetime"])
        out.append({"timestamp": ts_iso, "date": ymd, "value": p["value"], "source": "santiment"})
    out.sort(key=lambda x: x["date"])
    return out


# ---------- Augmento Bull & Bear ----------
def augmento_topics():
    r = requests.get(f"{AUG_BASE}/topics", timeout=30)
    r.raise_for_status()
    return r.json() or {}


def find_bull_bear_indices(topics_dict):
    bullish_keys = {"optimistic", "bullish", "positive"}
    bearish_keys = {"bearish", "negative", "pessimistic"}
    bulls, bears = set(), set()
    for idx_str, name in topics_dict.items():
        n = name.lower()
        if any(k in n for k in bullish_keys):
            bulls.add(int(idx_str))
        if any(k in n for k in bearish_keys):
            bears.add(int(idx_str))
    return sorted(bulls), sorted(bears)


def augmento_fetch(source, coin, bin_size, start_iso, end_iso, start_ptr, count_ptr, headers):
    params = {
        "source": source,
        "coin": coin,
        "bin_size": bin_size,
        "start_datetime": start_iso,
        "end_datetime": end_iso,
        "start_ptr": start_ptr,
        "count_ptr": count_ptr,
    }
    r = requests.get(f"{AUG_BASE}/events/aggregated", params=params, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json() or []


def fetch_augmento_bull_bear(coin="bitcoin", bin_size="24H", start_iso="2018-01-01T00:00:00Z", end_iso=None):
    if not end_iso:
        end_iso = dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")
    topics = augmento_topics()
    bull_ids, bear_ids = find_bull_bear_indices(topics)
    headers = {"Api-Key": AUGMENTO_API_KEY} if AUGMENTO_API_KEY else {}

    sources = ["twitter", "reddit", "bitcointalk"]
    count_ptr = 1000
    data = defaultdict(lambda: {"bull": 0, "bear": 0})

    for src in sources:
        start_ptr = 0
        while True:
            slab = augmento_fetch(src, coin, bin_size, start_iso, end_iso, start_ptr, count_ptr, headers)
            if not slab:
                break
            for step in slab:
                dt_iso = step.get("datetime")
                counts = step.get("counts", [])
                bull_sum = sum(counts[i] for i in bull_ids if i < len(counts))
                bear_sum = sum(counts[i] for i in bear_ids if i < len(counts))
                data[dt_iso]["bull"] += bull_sum
                data[dt_iso]["bear"] += bear_sum
            start_ptr += count_ptr
            time.sleep(0.15)

    out = []
    for ts_iso, vals in data.items():
        _, ymd = iso_and_date(ts_iso)
        bull, bear = vals["bull"], vals["bear"]
        total = bull + bear
        index = bull / total if total else None
        out.append({"timestamp": ts_iso, "date": ymd, "bull": bull, "bear": bear, "index": index, "source": "augmento"})
    out.sort(key=lambda x: x["date"])
    return out


# ---------- Merge ----------
def merge_fg(cmc, alt):
    by = defaultdict(dict)
    for r in alt:
        by[r["date"]]["date"] = r["date"]
        by[r["date"]]["alt_value"] = r["value"]
        by[r["date"]]["alt_class"] = r.get("classification")
    for r in cmc:
        by[r["date"]]["date"] = r["date"]
        by[r["date"]]["cmc_value"] = r["value"]
        by[r["date"]]["cmc_class"] = r.get("classification")
    merged = list(by.values())
    merged.sort(key=lambda x: x["date"])
    return merged


# ---------- Main ----------
def main():
    # Alternative.me
    alt = fetch_alt()
    write_csv("alt_fear_greed.csv", alt, ["timestamp", "date", "value", "classification", "source"])
    print(f"[ALT] {len(alt)} rows")

    # CMC
    cmc = fetch_cmc()
    write_csv("cmc_fear_greed.csv", cmc, ["timestamp", "date", "value", "classification", "source"])
    print(f"[CMC] {len(cmc)} rows")

    # Merge
    merged = merge_fg(cmc, alt)
    write_csv("fear_greed_merged.csv", merged, ["date", "alt_value", "alt_class", "cmc_value", "cmc_class"])
    print(f"[MERGED] {len(merged)} rows")

    # CNN
    try:
        cnn = fetch_cnn_fgi()
        write_csv("cnn_fgi.csv", cnn, ["timestamp", "date", "value", "source"])
        print(f"[CNN] {len(cnn)} rows")
    except Exception as e:
        print(f"[CNN] Skipped ({e})")

    # Santiment
    try:
        san = fetch_santiment_weighted_sentiment()
        write_csv("santiment_btc_sentiment.csv", san, ["timestamp", "date", "value", "source"])
        print(f"[SANTIMENT] {len(san)} rows")
    except Exception as e:
        print(f"[SANTIMENT] Skipped ({e})")

    # Augmento
    try:
        aug = fetch_augmento_bull_bear()
        write_csv("augmento_bull_bear.csv", aug, ["timestamp", "date", "bull", "bear", "index", "source"])
        print(f"[AUGMENTO] {len(aug)} rows")
    except Exception as e:
        print(f"[AUGMENTO] Skipped ({e})")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
