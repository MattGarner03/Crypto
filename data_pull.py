# binance_data.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
import time
import hashlib
import hmac
from urllib.parse import urlencode
import pandas as pd
from dotenv import load_dotenv

try:
    from binance.spot import Spot as SpotClient
    from binance.um_futures import UMFutures as FuturesClient
    _USING_CONNECTOR = True
except ImportError:
    SpotClient = None  # type: ignore
    FuturesClient = None  # type: ignore
    import requests  # type: ignore

    _USING_CONNECTOR = False

Number = Union[int, float]
TsLike = Union[int, float, str, datetime]

_SYMBOL_SUFFIX_BLACKLIST = (
    "UP",
    "DOWN",
    "BULL",
    "BEAR",
    "2L",
    "2S",
    "3L",
    "3S",
    "4L",
    "4S",
    "5L",
    "5S",
    "USD",
    "USDC",
    "USDT",
    "BUSD",
    "FDUSD",
)

_ENV_DEFAULT_PATH = Path(__file__).resolve().parent / ".env"


def _load_api_credentials(
    api_key: Optional[str], api_secret: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    """
    Load Binance API credentials from arguments or environment variables.

    Environment variables are populated from the nearest .env file if present.
    Keys passed explicitly to BinanceData override the environment.
    """
    if api_key and api_secret:
        return api_key, api_secret

    load_dotenv(
        dotenv_path=_ENV_DEFAULT_PATH if _ENV_DEFAULT_PATH.exists() else None,
        override=False,
    )

    return (
        api_key or os.getenv("BINANCE_API_KEY"),
        api_secret or os.getenv("BINANCE_API_SECRET"),
    )


def _to_millis(ts: Optional[TsLike]) -> Optional[int]:
    """Convert ISO string / datetime / seconds-ms int/float to Binance ms timestamp."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        # Heuristic: treat >= 10^12 as already ms
        return int(ts if ts >= 1_000_000_000_000 else ts * 1000)
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return int(ts.timestamp() * 1000)
    return int(pd.to_datetime(ts, utc=True).value // 1_000_000)


def _df_klines(raw: List[List[Any]]) -> pd.DataFrame:
    cols = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_base_volume",
        "taker_quote_volume", "ignore"
    ]
    df = pd.DataFrame(raw, columns=cols)
    num_cols = [
        "open", "high", "low", "close", "volume", "quote_asset_volume",
        "taker_base_volume", "taker_quote_volume"
    ]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


def _retry(call, *, tries=3, backoff=0.8):
    """Tiny retry helper for transient HTTP 429/5xx."""
    for i in range(tries):
        try:
            return call()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(backoff * (2 ** i))


@dataclass
class BinanceData:
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    # Use the lower-load CDN for public Spot data by default:
    spot_base_url: str = "https://data-api.binance.vision"

    def __post_init__(self):
        # Fill missing credentials from environment/.env if needed.
        self.api_key, self.api_secret = _load_api_credentials(self.api_key, self.api_secret)

        self._using_connector = _USING_CONNECTOR

        if self._using_connector:
            self.spot_public = SpotClient(base_url=self.spot_base_url)
            self.spot_auth = (
                SpotClient(api_key=self.api_key, api_secret=self.api_secret)
                if self.api_key and self.api_secret else None
            )
            self.um = FuturesClient(api_key=self.api_key, api_secret=self.api_secret)
        else:
            # HTTP fallback using requests for environments without binance-connector.
            self._http_user_agent = "binance-data/0.1"
            self._spot_http_base = f"{self.spot_base_url.rstrip('/')}/api"
            self._futures_http_base = "https://fapi.binance.com"

            self._spot_session = requests.Session()
            self._futures_session = requests.Session()

            self._spot_session.headers.update({"User-Agent": self._http_user_agent})
            self._futures_session.headers.update({"User-Agent": self._http_user_agent})

            if self.api_key:
                # Set API key header for endpoints that support/require it.
                self._spot_session.headers["X-MBX-APIKEY"] = self.api_key
                self._futures_session.headers["X-MBX-APIKEY"] = self.api_key

            # Expose placeholders to keep attribute names consistent with connector branch.
            self.spot_public = None
            self.spot_auth = None
            self.um = None

    # -------- HTTP helpers for fallback --------
    def _http_spot_get(self, path: str, params: Dict[str, Any]) -> Any:
        if self._using_connector:
            raise RuntimeError("HTTP helper should not be used when connector is available")
        clean_params = {k: v for k, v in params.items() if v is not None}
        url = f"{self._spot_http_base.rstrip('/')}/{path.lstrip('/')}"
        response = self._spot_session.get(url, params=clean_params, timeout=10)
        response.raise_for_status()
        return response.json()

    def _http_futures_get(self, path: str, params: Dict[str, Any], *, signed: bool = False) -> Any:
        if self._using_connector:
            raise RuntimeError("HTTP helper should not be used when connector is available")
        clean_params = {k: v for k, v in params.items() if v is not None}
        if signed:
            clean_params = self._sign_params(clean_params)
        url = f"{self._futures_http_base.rstrip('/')}/{path.lstrip('/')}"
        response = self._futures_session.get(url, params=clean_params, timeout=10)
        response.raise_for_status()
        return response.json()

    def _sign_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_secret:
            raise RuntimeError("Binance API secret required for signed endpoint")
        payload = dict(params)
        payload["timestamp"] = int(time.time() * 1000)
        query = urlencode(sorted(payload.items()), doseq=True)
        signature = hmac.new(self.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
        payload["signature"] = signature
        return payload

    # ---------- Spot ----------
    def spot_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start: Optional[TsLike] = None,
        end: Optional[TsLike] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        params = dict(symbol=symbol.upper(), interval=interval, limit=limit)
        st = _to_millis(start); et = _to_millis(end)
        if st:
            params["startTime"] = st
        if et:
            params["endTime"] = et

        if self._using_connector:
            raw = _retry(lambda: self.spot_public.klines(**params))
        else:
            raw = _retry(lambda: self._http_spot_get("v3/klines", params))
        return _df_klines(raw)

    def spot_depth(self, symbol: str, limit: int = 1000) -> Dict[str, Any]:
        symbol = symbol.upper()
        if self._using_connector:
            return _retry(lambda: self.spot_public.depth(symbol=symbol, limit=limit))
        return _retry(lambda: self._http_spot_get("v3/depth", {"symbol": symbol, "limit": limit}))

    def spot_avg_price(self, symbol: str) -> Dict[str, Any]:
        symbol = symbol.upper()
        if self._using_connector:
            return _retry(lambda: self.spot_public.avg_price(symbol=symbol))
        return _retry(lambda: self._http_spot_get("v3/avgPrice", {"symbol": symbol}))

    def spot_trades(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        symbol = symbol.upper()
        if self._using_connector:
            raw = _retry(lambda: self.spot_public.trades(symbol=symbol, limit=limit))
        else:
            raw = _retry(lambda: self._http_spot_get("v3/trades", {"symbol": symbol, "limit": limit}))
        df = pd.DataFrame(raw)
        if not df.empty:
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            if "quantity" in df.columns:
                df.rename(columns={"quantity": "qty"}, inplace=True)
            if "T" in df.columns and "time" not in df.columns:
                df.rename(columns={"T": "time"}, inplace=True)
            if "qty" in df.columns:
                df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        return df

    def spot_agg_trades(
        self,
        symbol: str,
        *,
        start: Optional[TsLike] = None,
        end: Optional[TsLike] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        symbol = symbol.upper()
        params: Dict[str, Any] = {"symbol": symbol, "limit": limit}
        start_ms = _to_millis(start)
        end_ms = _to_millis(end)
        if start_ms:
            params["startTime"] = start_ms
        if end_ms:
            params["endTime"] = end_ms

        if self._using_connector:
            raw = _retry(lambda: self.spot_public.agg_trades(**params))
        else:
            raw = _retry(lambda: self._http_spot_get("v3/aggTrades", params))

        df = pd.DataFrame(raw)
        if not df.empty:
            rename_map = {"p": "price", "q": "qty", "T": "time", "f": "first_id", "l": "last_id", "a": "aggregate_id"}
            df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
            if "price" in df.columns:
                df["price"] = pd.to_numeric(df["price"], errors="coerce")
            if "qty" in df.columns:
                df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        return df

    def spot_book_ticker(self, symbol: str) -> Dict[str, Any]:
        symbol = symbol.upper()
        if self._using_connector:
            return _retry(lambda: self.spot_public.book_ticker(symbol=symbol))
        return _retry(lambda: self._http_spot_get("v3/ticker/bookTicker", {"symbol": symbol}))

    def spot_ticker_price(self, symbol: str) -> Dict[str, Any]:
        symbol = symbol.upper()
        if self._using_connector:
            return _retry(lambda: self.spot_public.ticker_price(symbol=symbol))
        return _retry(lambda: self._http_spot_get("v3/ticker/price", {"symbol": symbol}))

    def spot_ticker_24hr(self, symbol: str) -> Dict[str, Any]:
        symbol = symbol.upper()
        if self._using_connector:
            return _retry(lambda: self.spot_public.ticker_24hr(symbol=symbol))
        return _retry(lambda: self._http_spot_get("v3/ticker/24hr", {"symbol": symbol}))

    def spot_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        if self._using_connector:
            return _retry(lambda: self.spot_public.exchange_info(**params))
        return _retry(lambda: self._http_spot_get("v3/exchangeInfo", params))

    # ---------- USDⓈ-M Futures ----------
    def fut_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start: Optional[TsLike] = None,
        end: Optional[TsLike] = None,
        limit: int = 1500,
    ) -> pd.DataFrame:
        params = dict(symbol=symbol.upper(), interval=interval, limit=limit)
        st = _to_millis(start); et = _to_millis(end)
        if st:
            params["startTime"] = st
        if et:
            params["endTime"] = et

        if self._using_connector:
            raw = _retry(lambda: self.um.klines(**params))
        else:
            raw = _retry(lambda: self._http_futures_get("fapi/v1/klines", params))
        return _df_klines(raw)

    def fut_mark_price(self, symbol: str) -> Dict[str, Any]:
        symbol = symbol.upper()
        if self._using_connector:
            return _retry(lambda: self.um.mark_price(symbol=symbol))
        return _retry(lambda: self._http_futures_get("fapi/v1/premiumIndex", {"symbol": symbol}))

    def fut_open_interest(self, symbol: str) -> Dict[str, Any]:
        symbol = symbol.upper()
        if self._using_connector:
            return _retry(lambda: self.um.open_interest(symbol=symbol))
        return _retry(lambda: self._http_futures_get("fapi/v1/openInterest", {"symbol": symbol}))

    def fut_funding_history(
        self,
        symbol: str,
        start: Optional[TsLike] = None,
        end: Optional[TsLike] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        params = dict(symbol=symbol.upper(), limit=limit)
        st = _to_millis(start); et = _to_millis(end)
        if st:
            params["startTime"] = st
        if et:
            params["endTime"] = et

        if self._using_connector:
            raw = _retry(lambda: self.um.funding_rate(**params))
        else:
            raw = _retry(lambda: self._http_futures_get("fapi/v1/fundingRate", params))
        df = pd.DataFrame(raw)
        if not df.empty:
            df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
            df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        return df

    def fut_index_price_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start: Optional[TsLike] = None,
        end: Optional[TsLike] = None,
        limit: int = 720,
    ) -> pd.DataFrame:
        params = dict(symbol=symbol.upper(), interval=interval, limit=limit)
        st = _to_millis(start); et = _to_millis(end)
        if st:
            params["startTime"] = st
        if et:
            params["endTime"] = et

        if self._using_connector:
            raw = _retry(lambda: self.um.index_price_klines(**params))
        else:
            http_params = dict(params)
            http_params["pair"] = http_params.pop("symbol")
            raw = _retry(lambda: self._http_futures_get("fapi/v1/indexPriceKlines", http_params))
        return _df_klines(raw)

    def fut_premium_index_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start: Optional[TsLike] = None,
        end: Optional[TsLike] = None,
        limit: int = 720,
    ) -> pd.DataFrame:
        params = dict(symbol=symbol.upper(), interval=interval, limit=limit)
        st = _to_millis(start); et = _to_millis(end)
        if st:
            params["startTime"] = st
        if et:
            params["endTime"] = et

        if self._using_connector:
            raw = _retry(lambda: self.um.premium_index_klines(**params))
        else:
            raw = _retry(lambda: self._http_futures_get("fapi/v1/premiumIndexKlines", params))
        return _df_klines(raw)

    def fut_open_interest_hist(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 500,
        start: Optional[TsLike] = None,
        end: Optional[TsLike] = None,
    ) -> pd.DataFrame:
        params = dict(symbol=symbol.upper(), period=period, limit=limit)
        st = _to_millis(start); et = _to_millis(end)
        if st:
            params["startTime"] = st
        if et:
            params["endTime"] = et

        if self._using_connector:
            raw = _retry(lambda: self.um.open_interest_hist(**params))
        else:
            target_start = params.pop("startTime", None)
            target_end = params.pop("endTime", None)
            target_limit = limit or 500

            batch_size = 500  # maximise coverage; slice later to honour limit
            aggregated: List[Dict[str, Any]] = []
            next_end = target_end
            last_earliest: Optional[int] = None

            while True:
                def _call(ne=next_end):
                    batch_params = {
                        "symbol": params["symbol"],
                        "period": params["period"],
                        "limit": batch_size,
                    }
                    if ne is not None:
                        batch_params["endTime"] = ne
                    return self._http_futures_get("futures/data/openInterestHist", batch_params)

                batch = _retry(_call)
                if not batch:
                    break

                aggregated = batch + aggregated if aggregated else batch
                earliest = aggregated[0]["timestamp"]

                if target_start is not None and earliest <= target_start:
                    break
                if len(batch) < batch_size:
                    break

                next_candidate = batch[0]["timestamp"] - 1
                if next_end is not None and next_candidate >= next_end:
                    break
                if last_earliest is not None and next_candidate >= last_earliest:
                    break

                last_earliest = batch[0]["timestamp"]
                next_end = next_candidate

                if target_start is None and len(aggregated) >= target_limit:
                    break
                if target_start is not None and len(aggregated) >= target_limit * 2:
                    break

            raw = aggregated
            raw.sort(key=lambda x: x["timestamp"])
            if target_start is not None:
                raw = [row for row in raw if row["timestamp"] >= target_start]
            if target_end is not None:
                raw = [row for row in raw if row["timestamp"] <= target_end]
            if target_limit and len(raw) > target_limit:
                if target_start is not None:
                    raw = raw[:target_limit]
                else:
                    raw = raw[-target_limit:]
        df = pd.DataFrame(raw)
        if not df.empty:
            df["sumOpenInterest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
            df["sumOpenInterestValue"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df


def top_spot_symbols(
    client: BinanceData,
    *,
    quote_asset: str = "USDT",
    limit: int = 10,
) -> List[str]:
    quote_asset = quote_asset.upper()
    if client._using_connector:
        tickers = _retry(lambda: client.spot_public.ticker_24hr())
    else:
        tickers = _retry(lambda: client._http_spot_get("v3/ticker/24hr", {}))

    df = pd.DataFrame(tickers)
    if df.empty or "symbol" not in df.columns or "quoteVolume" not in df.columns:
        return []

    df = df[df["symbol"].str.endswith(quote_asset)]
    if df.empty:
        return []

    df["quoteVolume"] = pd.to_numeric(df["quoteVolume"], errors="coerce")
    df = df[df["quoteVolume"].notna()]
    if df.empty:
        return []

    df["base"] = df["symbol"].str[:-len(quote_asset)]
    df = df[df["base"].str.upper().str.len() > 0]
    df = df[~df["base"].str.upper().str.endswith(_SYMBOL_SUFFIX_BLACKLIST)]
    if df.empty:
        return []

    df = df.sort_values("quoteVolume", ascending=False)
    return df["symbol"].head(limit).tolist()


def top_market_cap_spot_symbols(
    *,
    quote_asset: str = "USDT",
    limit: int = 20,
) -> List[str]:
    """
    Determine the top spot symbols by market capitalisation using Binance product metadata.

    The function pulls the full product catalogue from binance.com, filters to live spot
    markets for the chosen quote asset, and computes market cap as price * circulating supply.
    """
    try:
        import requests as _requests
    except ImportError as exc:
        raise RuntimeError("requests package is required to resolve market-cap rankings") from exc

    quote_asset = quote_asset.upper()
    url = "https://www.binance.com/bapi/asset/v2/public/asset-service/product/get-products"

    def _fetch():
        response = _requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    payload = _retry(_fetch)
    items = payload.get("data") if isinstance(payload, dict) else None
    if not items:
        return []

    df = pd.DataFrame(items)
    if df.empty:
        return []

    df["symbol"] = df["s"].astype(str).str.upper()
    df["quote"] = df["q"].astype(str).str.upper()
    df = df[df["quote"] == quote_asset]
    if df.empty:
        return []

    df = df[df["st"] == "TRADING"]
    if "etf" in df.columns:
        df = df[df["etf"].isin([False, "false", 0])]

    df["price"] = pd.to_numeric(df["c"], errors="coerce")
    df["circulating"] = pd.to_numeric(df["cs"], errors="coerce")
    df = df[df["price"].notna() & df["circulating"].notna()]
    df = df[df["circulating"] > 0]
    if df.empty:
        return []

    df["market_cap"] = df["price"] * df["circulating"]
    df = df[df["market_cap"].notna() & (df["market_cap"] > 0)]
    if df.empty:
        return []

    df = df[df["symbol"].str.endswith(quote_asset)]
    df["base"] = df["symbol"].str[:-len(quote_asset)]
    df = df[df["base"].str.upper().str.len() > 0]
    df = df[~df["base"].str.upper().str.endswith(_SYMBOL_SUFFIX_BLACKLIST)]
    if df.empty:
        return []

    df = df.sort_values("market_cap", ascending=False)
    df = df.drop_duplicates(subset=["base"], keep="first")

    if limit and limit > 0:
        df = df.head(limit)

    return df["symbol"].tolist()


def spot_symbol_onboard_time(client: BinanceData, symbol: str) -> Optional[int]:
    symbol = symbol.upper()
    if client._using_connector:
        info = _retry(lambda: client.spot_public.exchange_info(symbol=symbol))
    else:
        info = _retry(lambda: client._http_spot_get("v3/exchangeInfo", {"symbol": symbol}))
    symbols = info.get("symbols") if isinstance(info, dict) else None
    if not symbols:
        return None
    entry = next((item for item in symbols if item.get("symbol") == symbol), None)
    if not entry:
        return None
    onboard = entry.get("onboardDate")
    if onboard is None:
        return None
    try:
        return int(onboard)
    except (TypeError, ValueError):
        return None


def fetch_full_hourly_klines(
    client: BinanceData,
    symbol: str,
    *,
    interval: str = "1h",
    batch_limit: int = 1000,
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    next_end: Optional[int] = None
    earliest = spot_symbol_onboard_time(client, symbol)
    prev_first_open: Optional[int] = None

    while True:
        df = client.spot_klines(symbol, interval=interval, end=next_end, limit=batch_limit)
        if df.empty:
            break

        frames.append(df)
        first_open = int(df["open_time"].iloc[0].value // 1_000_000)
        if prev_first_open is not None and first_open >= prev_first_open:
            break
        prev_first_open = first_open

        if len(df) < batch_limit:
            break

        if earliest is not None and first_open <= earliest:
            break

        next_end_candidate = first_open - 1
        if earliest is not None and next_end_candidate <= earliest:
            next_end = earliest
        elif next_end_candidate <= 0:
            next_end = 0
        else:
            next_end = next_end_candidate
        time.sleep(sleep_seconds)

    if not frames:
        return pd.DataFrame()

    frames.reverse()
    full = pd.concat(frames, ignore_index=True)
    full.insert(0, "symbol", symbol.upper())
    return full


def collect_spot_market_data(
    client: BinanceData,
    symbol: str,
    *,
    output_root: Path,
    depth_limit: int = 500,
    trades_limit: int = 1000,
    agg_trades_limit: int = 1000,
    kline_intervals: Optional[List[str]] = None,
    kline_limit: int = 1000,
    open_interest_period: str = "1h",
    open_interest_limit: int = 500,
) -> None:
    """
    Fetch a broad snapshot of spot market data for a symbol and persist it to disk.

    The snapshot covers depth, recent trades, aggregated trades, multiple kline intervals,
    ticker statistics, and (when available) USD-M futures open interest metrics.
    """
    symbol = symbol.upper()
    output_root.mkdir(parents=True, exist_ok=True)
    symbol_dir = output_root / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    def _write_json(filename: str, payload: Any) -> None:
        if payload is None:
            return
        path = symbol_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _write_df_csv(filename: str, frame: Optional[pd.DataFrame]) -> None:
        if frame is None or frame.empty:
            return
        path = symbol_dir / filename
        frame.to_csv(path, index=False)

    fetch_time = datetime.now(timezone.utc).isoformat()

    summary: Dict[str, Any] = {"symbol": symbol, "fetched_at": fetch_time}

    try:
        summary["avg_price"] = client.spot_avg_price(symbol)
    except Exception as exc:
        summary["avg_price_error"] = str(exc)

    try:
        summary["book_ticker"] = client.spot_book_ticker(symbol)
    except Exception as exc:
        summary["book_ticker_error"] = str(exc)

    try:
        summary["ticker_price"] = client.spot_ticker_price(symbol)
    except Exception as exc:
        summary["ticker_price_error"] = str(exc)

    try:
        summary["ticker_24hr"] = client.spot_ticker_24hr(symbol)
    except Exception as exc:
        summary["ticker_24hr_error"] = str(exc)

    _write_json("summary.json", summary)

    try:
        depth = client.spot_depth(symbol, limit=depth_limit)
    except Exception as exc:
        depth = {"error": str(exc)}
    _write_json("depth.json", depth)

    try:
        trades_df = client.spot_trades(symbol, limit=trades_limit)
    except Exception as exc:
        trades_df = pd.DataFrame([{"error": str(exc)}])
    _write_df_csv("trades.csv", trades_df)

    try:
        agg_trades_df = client.spot_agg_trades(symbol, limit=agg_trades_limit)
    except Exception as exc:
        agg_trades_df = pd.DataFrame([{"error": str(exc)}])
    _write_df_csv("agg_trades.csv", agg_trades_df)

    intervals = kline_intervals or ["1m", "1h", "1d"]
    for interval in intervals:
        try:
            klines = client.spot_klines(symbol, interval=interval, limit=kline_limit)
        except Exception as exc:
            klines = pd.DataFrame([{"error": str(exc)}])
        _write_df_csv(f"klines_{interval}.csv", klines)

    try:
        ex_info = client.spot_exchange_info(symbol)
    except Exception as exc:
        ex_info = {"error": str(exc)}
    _write_json("exchange_info.json", ex_info)

    try:
        oi = client.fut_open_interest(symbol)
    except Exception as exc:
        oi = {"error": str(exc)}
    _write_json("futures_open_interest.json", oi)

    try:
        oi_hist = client.fut_open_interest_hist(
            symbol,
            period=open_interest_period,
            limit=open_interest_limit,
        )
    except Exception as exc:
        oi_hist = pd.DataFrame([{"error": str(exc)}])
    _write_df_csv("futures_open_interest_hist.csv", oi_hist)


# ------------------ Example usage ------------------
if __name__ == "__main__":
    client = BinanceData()

    top_symbols = top_market_cap_spot_symbols(quote_asset="USDT", limit=20)
    if not top_symbols:
        raise SystemExit("Unable to determine top market-cap symbols.")

    output_dir = Path("data") / "spot_top20"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Top {len(top_symbols)} USDT pairs by market cap: {', '.join(top_symbols)}")
    for symbol in top_symbols:
        print(f"[{symbol}] Collecting spot market snapshot...")
        collect_spot_market_data(
            client,
            symbol,
            output_root=output_dir,
            depth_limit=500,
            trades_limit=1000,
            agg_trades_limit=1000,
            kline_intervals=["1m", "1h", "1d"],
            kline_limit=1000,
            open_interest_period="1h",
            open_interest_limit=500,
        )

        print(f"[{symbol}] Fetching full hourly kline history...")
        history_df = fetch_full_hourly_klines(client, symbol, interval="1h")
        symbol_dir = output_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        if history_df.empty:
            print(f"[{symbol}] No hourly kline history retrieved.")
        else:
            history_path = symbol_dir / "klines_1h_full_history.csv"
            history_df.to_csv(history_path, index=False)
            print(f"[{symbol}] Saved {len(history_df)} rows to {history_path}")

    print("Collection complete. Files written under:", output_dir.resolve())
