import san
import pandas as pd

# Your API key
san.ApiConfig.api_key = "phe5zyohhzm2glyq_nxj26byqfiixogye"

def save_santiment_csv(
    metric="sentiment_weighted_total_1d",  # fixed metric name
    slug="bitcoin",
    from_date="2018-01-01",
    to_date="utc_now",
    interval="1d",
    out_path="santiment_btc_sentiment.csv",
):
    df = san.get(
        metric,
        slug=slug,
        from_date=from_date,
        to_date=to_date,
        interval=interval,
    )

    # Handle empty / unexpected frames
    if df is None or len(df) == 0:
        print(f"[SANTIMENT] No data returned for metric={metric}, slug={slug}.")
        return

    # Ensure we have a datetime series (either a column or the index)
    if "datetime" in df.columns:
        ts = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    else:
        # Many sanpy endpoints return DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        else:
            # Try common fallbacks
            for candidate in ("time", "date"):
                if candidate in df.columns:
                    ts = pd.to_datetime(df[candidate], utc=True, errors="coerce")
                    break
            else:
                raise KeyError("No datetime column or DatetimeIndex found in Santiment DataFrame.")

    # Ensure we have a 'value' series
    if "value" in df.columns:
        vals = df["value"]
    else:
        # Some metrics might name it differently; if there’s only one numeric column, use it
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) == 1:
            vals = df[numeric_cols[0]]
        else:
            raise KeyError("Could not identify the 'value' column in Santiment DataFrame.")

    # Build output frame
    out = pd.DataFrame({
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date": ts.strftime("%Y-%m-%d"),
        "value": vals.values,
        "metric": metric,
        "slug": slug,
        "source": "santiment",
    })

    # Drop any rows where timestamp failed to parse
    out = out.dropna(subset=["timestamp", "date"])

    out.to_csv(out_path, index=False)
    print(f"[SANTIMENT] Wrote {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    # Weighted sentiment (daily)
    save_santiment_csv()

    # Example you tried earlier (works too)
    save_santiment_csv(
        metric="daily_active_addresses",
        slug="bitcoin",
        from_date="2024-01-01",
        to_date="2024-01-31",
        interval="1d",
        out_path="santiment_btc_daa_jan2024.csv",
    )
