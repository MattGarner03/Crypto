"""Collect Binance USD-M futures datasets across multiple public endpoints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import requests

from data_pull import BinanceData, top_market_cap_spot_symbols

FUTURES_BASE_URL = "https://fapi.binance.com"
OPTIONS_BASE_URL = "https://eapi.binance.com"
USER_AGENT = "binance-futures-collector/0.1"


def _http_get(
    session: requests.Session,
    base_url: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    clean_params = {k: v for k, v in (params or {}).items() if v is not None}
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    response = session.get(url, params=clean_params, timeout=15)
    response.raise_for_status()
    return response.json()


def _signed_params(client: BinanceData, params: Dict[str, Any]) -> Dict[str, Any]:
    if not client.api_secret:
        raise RuntimeError("Binance API secret required for signed futures endpoint.")
    return client._sign_params(params)


def _ensure_session(client: BinanceData) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    if client.api_key:
        session.headers["X-MBX-APIKEY"] = client.api_key
    session.headers["Accept"] = "application/json"
    return session


def _coerce_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _coerce_timestamp(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], unit="ms", utc=True)


def _ms_to_iso(value: Any) -> Any:
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return value
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _fetch_ratio_table(
    session: requests.Session,
    path: str,
    symbol: str,
    period: str,
    max_rows: Optional[int],
    numeric_cols: Sequence[str],
) -> pd.DataFrame:
    aggregated: list[Dict[str, Any]] = []
    next_end: Optional[int] = None
    last_earliest: Optional[int] = None
    per_request_limit = 500
    iterations = 0

    while True:
        iterations += 1
        if iterations > 1000:
            break

        request_limit = per_request_limit
        if max_rows is not None:
            remaining = max_rows - len(aggregated)
            if remaining <= 0:
                break
            request_limit = max(1, min(request_limit, remaining))

        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "period": period,
            "limit": request_limit,
        }
        if next_end is not None:
            params["endTime"] = next_end

        batch = _http_get(session, FUTURES_BASE_URL, path, params)
        if not batch:
            break

        aggregated = batch + aggregated if aggregated else batch

        earliest = aggregated[0].get("timestamp")
        if earliest is None:
            break

        if len(batch) < request_limit:
            break

        next_candidate = batch[0].get("timestamp")
        if next_candidate is None:
            break
        next_candidate -= 1

        if last_earliest is not None and earliest >= last_earliest:
            break
        last_earliest = earliest

        if max_rows is not None and len(aggregated) >= max_rows:
            break

        if next_end is not None and next_candidate >= next_end:
            break
        next_end = next_candidate

    frame = pd.DataFrame(aggregated)
    if frame.empty:
        return frame

    frame = frame.drop_duplicates(subset="timestamp", keep="last")
    frame.sort_values("timestamp", inplace=True)
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.tail(max_rows)
    frame.reset_index(drop=True, inplace=True)

    _coerce_numeric(frame, numeric_cols)
    _coerce_timestamp(frame, ["timestamp"])

    frame["symbol"] = symbol.upper()
    frame["period"] = period
    columns = ["symbol", "period"] + [col for col in frame.columns if col not in {"symbol", "period"}]
    return frame[columns]


def fetch_global_long_short_ratio(
    session: requests.Session,
    symbol: str,
    period: str,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    numeric = ["longShortRatio", "longAccount", "shortAccount"]
    return _fetch_ratio_table(
        session,
        "futures/data/globalLongShortAccountRatio",
        symbol,
        period,
        limit,
        numeric,
    )


def fetch_top_trader_account_ratio(
    session: requests.Session,
    symbol: str,
    period: str,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    numeric = ["longShortRatio", "longAccount", "shortAccount"]
    return _fetch_ratio_table(
        session,
        "futures/data/topLongShortAccountRatio",
        symbol,
        period,
        limit,
        numeric,
    )


def fetch_top_trader_position_ratio(
    session: requests.Session,
    symbol: str,
    period: str,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    numeric = ["longShortRatio", "longPosition", "shortPosition"]
    return _fetch_ratio_table(
        session,
        "futures/data/topLongShortPositionRatio",
        symbol,
        period,
        limit,
        numeric,
    )


def fetch_taker_long_short_ratio(
    session: requests.Session,
    symbol: str,
    period: str,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    frame = _fetch_ratio_table(
        session,
        "futures/data/takerlongshortRatio",
        symbol,
        period,
        limit,
        ["buySellRatio", "buyVol", "sellVol"],
    )
    return frame


def fetch_force_orders(
    client: BinanceData,
    session: requests.Session,
    symbol: str,
    limit: int,
) -> pd.DataFrame:
    params = {"symbol": symbol.upper(), "limit": limit}
    payload = _http_get(
        session,
        FUTURES_BASE_URL,
        "fapi/v1/forceOrders",
        _signed_params(client, params),
    )
    frame = pd.DataFrame(payload)
    if frame.empty:
        return frame
    numeric_columns = [
        "price",
        "avgPrice",
        "origQty",
        "executedQty",
        "executedQuotationQty",
        "balance",
        "pnl",
    ]
    _coerce_numeric(frame, [col for col in numeric_columns if col in frame.columns])
    _coerce_timestamp(frame, ["time", "updateTime"])
    frame.insert(0, "symbol", symbol.upper())
    return frame


def fetch_options_open_interest(
    session: requests.Session,
    underlying: str,
    limit: int,
) -> pd.DataFrame:
    payload = _http_get(
        session,
        OPTIONS_BASE_URL,
        "eapi/v1/openInterest",
        {"underlying": underlying.upper(), "limit": limit},
    )
    frame = pd.DataFrame(payload)
    if frame.empty:
        return frame
    _coerce_numeric(frame, ["sumOpenInterest", "sumOpenInterestUsd"])
    _coerce_timestamp(frame, ["timestamp"])
    frame.insert(0, "underlying", underlying.upper())
    return frame


def collect_futures_market_data(
    client: BinanceData,
    symbol: str,
    *,
    output_root: Path,
    futures_kline_intervals: Optional[Sequence[str]] = None,
    futures_kline_limit: int = 1000,
    premium_kline_intervals: Optional[Sequence[str]] = None,
    premium_kline_limit: int = 720,
    funding_limit: int = 1000,
    open_interest_periods: Optional[Sequence[str]] = None,
    open_interest_limit: int = 500,
    ratio_periods: Optional[Sequence[str]] = None,
    ratio_limit: Optional[int] = None,
    taker_periods: Optional[Sequence[str]] = None,
    taker_limit: Optional[int] = None,
    force_orders_limit: int = 100,
    include_options_data: bool = False,
    options_limit: int = 200,
) -> None:
    symbol = symbol.upper()
    output_root.mkdir(parents=True, exist_ok=True)
    symbol_dir = output_root / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    futures_session = _ensure_session(client)
    options_session = _ensure_session(client) if include_options_data else None

    def _write_json(filename: str, payload: Any) -> None:
        if payload is None:
            return
        path = symbol_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _write_df(filename: str, frame: Optional[pd.DataFrame]) -> None:
        if frame is None or frame.empty:
            return
        path = symbol_dir / filename
        frame.to_csv(path, index=False)

    fetch_time = datetime.now(timezone.utc).isoformat()
    summary: Dict[str, Any] = {"symbol": symbol, "fetched_at": fetch_time}

    try:
        mark_price = client.fut_mark_price(symbol)
        _write_json("mark_price_snapshot.json", mark_price)
        summary["mark_price"] = {
            "markPrice": float(mark_price.get("markPrice", "nan")),
            "indexPrice": float(mark_price.get("indexPrice", "nan")),
            "lastFundingRate": float(mark_price.get("lastFundingRate", "nan")),
            "nextFundingTime": _ms_to_iso(mark_price.get("nextFundingTime")),
        }
    except Exception as exc:
        summary["mark_price_error"] = str(exc)

    try:
        open_interest = client.fut_open_interest(symbol)
        _write_json("open_interest_snapshot.json", open_interest)
        summary["open_interest"] = {
            "openInterest": float(open_interest.get("openInterest", "nan")),
        }
    except Exception as exc:
        summary["open_interest_error"] = str(exc)

    try:
        funding_history = client.fut_funding_history(symbol, limit=funding_limit)
        _write_df("funding_rate_history.csv", funding_history)
    except Exception as exc:
        summary["funding_history_error"] = str(exc)

    futures_intervals = futures_kline_intervals or ("1m", "1h", "1d")
    for interval in futures_intervals:
        try:
            klines = client.fut_klines(symbol, interval=interval, limit=futures_kline_limit)
        except Exception as exc:
            klines = pd.DataFrame([{"error": str(exc)}])
        _write_df(f"futures_klines_{interval}.csv", klines)

    premium_intervals = premium_kline_intervals or ("5m", "1h", "1d")
    for interval in premium_intervals:
        try:
            premium_klines = client.fut_premium_index_klines(
                symbol,
                interval=interval,
                limit=premium_kline_limit,
            )
        except Exception as exc:
            premium_klines = pd.DataFrame([{"error": str(exc)}])
        _write_df(f"premium_index_klines_{interval}.csv", premium_klines)

        try:
            index_klines = client.fut_index_price_klines(
                symbol,
                interval=interval,
                limit=premium_kline_limit,
            )
        except Exception as exc:
            index_klines = pd.DataFrame([{"error": str(exc)}])
        _write_df(f"index_price_klines_{interval}.csv", index_klines)

    oi_periods = open_interest_periods or ("5m", "1h", "1d")
    for period in oi_periods:
        try:
            oi_hist = client.fut_open_interest_hist(
                symbol,
                period=period,
                limit=open_interest_limit,
            )
        except Exception as exc:
            oi_hist = pd.DataFrame([{"error": str(exc)}])
        _write_df(f"open_interest_hist_{period}.csv", oi_hist)

    ratio_periods = ratio_periods or ("5m", "15m", "1h", "4h", "1d")
    for period in ratio_periods:
        try:
            global_ratio = fetch_global_long_short_ratio(
                futures_session,
                symbol,
                period,
                ratio_limit,
            )
        except Exception as exc:
            global_ratio = pd.DataFrame([{"error": str(exc)}])
        _write_df(f"global_long_short_accounts_{period}.csv", global_ratio)

        try:
            top_accounts = fetch_top_trader_account_ratio(
                futures_session,
                symbol,
                period,
                ratio_limit,
            )
        except Exception as exc:
            top_accounts = pd.DataFrame([{"error": str(exc)}])
        _write_df(f"top_trader_accounts_{period}.csv", top_accounts)

        try:
            top_positions = fetch_top_trader_position_ratio(
                futures_session,
                symbol,
                period,
                ratio_limit,
            )
        except Exception as exc:
            top_positions = pd.DataFrame([{"error": str(exc)}])
        _write_df(f"top_trader_positions_{period}.csv", top_positions)

    taker_periods = taker_periods or ("5m", "15m", "1h", "4h", "1d")
    for period in taker_periods:
        try:
            taker_ratio = fetch_taker_long_short_ratio(
                futures_session,
                symbol,
                period,
                taker_limit,
            )
        except Exception as exc:
            taker_ratio = pd.DataFrame([{"error": str(exc)}])
        _write_df(f"taker_long_short_ratio_{period}.csv", taker_ratio)

    if force_orders_limit > 0:
        if not client.api_key or not client.api_secret:
            summary["force_orders_note"] = "Skipped force orders: API key/secret required."
        else:
            try:
                force_orders = fetch_force_orders(client, futures_session, symbol, force_orders_limit)
            except Exception as exc:
                force_orders = pd.DataFrame([{"error": str(exc)}])
            else:
                summary["force_orders_rows"] = len(force_orders)
            _write_df("force_orders.csv", force_orders)

    if include_options_data and options_session is not None:
        try:
            options_oi = fetch_options_open_interest(options_session, symbol, options_limit)
        except Exception as exc:
            options_oi = pd.DataFrame([{"error": str(exc)}])
        _write_df("options_open_interest.csv", options_oi)

    summary_path = symbol_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    client = BinanceData()
    symbols = top_market_cap_spot_symbols(quote_asset="USDT", limit=20)
    if not symbols:
        raise SystemExit("Unable to determine symbols to collect.")

    output_dir = Path("data") / "futures_top20"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting futures market data for {len(symbols)} symbols: {', '.join(symbols)}")
    for symbol in symbols:
        print(f"[{symbol}] Gathering futures datasets...")
        collect_futures_market_data(
            client,
            symbol,
            output_root=output_dir,
            futures_kline_intervals=("1m", "1h", "1d"),
            futures_kline_limit=1000,
            premium_kline_intervals=("5m", "1h", "1d"),
            premium_kline_limit=720,
            funding_limit=1000,
            open_interest_periods=("5m", "1h", "1d"),
            open_interest_limit=500,
            ratio_periods=("5m", "15m", "1h", "4h", "1d"),
            ratio_limit=None,
            taker_periods=("5m", "15m", "1h", "4h", "1d"),
            taker_limit=None,
            force_orders_limit=100,
            include_options_data=False,
        )

    print("Futures data collection complete. Files stored under:", output_dir.resolve())
