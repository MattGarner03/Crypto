from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
from pytrends.request import TrendReq

# Default configuration -----------------------------------------------------

DEFAULT_PRICE_PATTERNS: tuple[str, ...] = (
    "data/hourly_top10/*_1h.csv",
    "data/spot_top20/*/klines_1h_full_history.csv",
)
DEFAULT_OUTPUT_DIR = Path("data/google_trends")
DEFAULT_GEO = ""
DEFAULT_RATE_DELAY = 1.0  # seconds between API calls
DEFAULT_FMP_CSV = Path("FMP") / "fmp_cryptocurrencies_by_market_cap.csv"

FETCH_CONFIG = {
    "hourly": {
        "span": timedelta(days=7),
        "step": timedelta(hours=1),
        "overlap": timedelta(hours=1),
        "format": "%Y-%m-%dT%H",
    },
    "daily": {
        "span": timedelta(days=269),
        "step": timedelta(days=1),
        "overlap": timedelta(days=7),
        "format": "%Y-%m-%d",
    },
}

QUOTE_SUFFIXES: tuple[str, ...] = (
    "USDT",
    "USDC",
    "USD",
    "BUSD",
    "FDUSD",
    "TUSD",
    "EUR",
    "BTC",
    "ETH",
    "TRY",
    "GBP",
    "AUD",
    "CAD",
    "JPY",
    "BRL",
    "BIDR",
    "DAI",
)

MANUAL_NAME_MAP: Dict[str, str] = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "BNB": "BNB",
    "SOL": "Solana",
    "XRP": "XRP",
    "DOGE": "Dogecoin",
    "ADA": "Cardano",
    "SUI": "Sui",
    "EDEN": "Eden",
    "ASTER": "Astar",
    "ASTR": "Astar",
    "XPL": "XPLUS",
    "XPLUS": "XPLUS",
}


@dataclass
class PriceHistory:
    """Container for token price coverage derived from CSV price history."""

    symbol: str
    start: datetime
    end: datetime
    source: Path

    @property
    def base_symbol(self) -> str:
        return extract_base_symbol(self.symbol)


@dataclass
class TokenTarget:
    """Pair price history with the resolved Google Trends keyword."""

    history: PriceHistory
    keyword: str

    @property
    def symbol(self) -> str:
        return self.history.symbol

    @property
    def start(self) -> datetime:
        return self.history.start

    @property
    def end(self) -> datetime:
        return self.history.end

    @property
    def source(self) -> Path:
        return self.history.source


# ---------------------------------------------------------------------------
# Helper functions


def ensure_utc(value: datetime) -> datetime:
    """Return the datetime as an aware UTC timestamp."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def infer_symbol_from_path(path: Path) -> Optional[str]:
    """Attempt to infer the trading symbol from a price CSV path."""
    stem = path.stem
    first_token = stem.split("_", 1)[0]
    if first_token and first_token.lower() != "klines":
        return first_token.upper()

    parent = path.parent.name
    if parent and parent.lower() not in {"data", "hourly_top10", "spot_top20", "futures_top20"}:
        return parent.upper()

    return None


def extract_base_symbol(symbol: str) -> str:
    """Strip common quote asset suffixes to get the base symbol."""
    upper = symbol.upper()
    for suffix in QUOTE_SUFFIXES:
        if upper.endswith(suffix):
            base = upper[: -len(suffix)]
            return base or upper
    return upper


def parse_price_file_time_bounds(path: Path) -> Optional[Tuple[datetime, datetime]]:
    """Return the earliest open and latest close timestamps found in the CSV."""
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception as exc:
        print(f"[warn] Failed to read header from {path}: {exc}")
        return None

    start_col = None
    for candidate in ("open_time", "timestamp", "openTime", "time"):
        if candidate in header.columns:
            start_col = candidate
            break
    if start_col is None:
        return None

    end_col = "close_time" if "close_time" in header.columns else start_col
    usecols = sorted({start_col, end_col})

    start_val: Optional[pd.Timestamp] = None
    end_val: Optional[pd.Timestamp] = None

    try:
        for chunk in pd.read_csv(
            path,
            usecols=usecols,
            parse_dates=usecols,
            chunksize=200_000,
        ):
            chunk_start = chunk[start_col].dropna()
            if not chunk_start.empty:
                min_val = chunk_start.min()
                start_val = min_val if start_val is None or min_val < start_val else start_val

            chunk_end = chunk[end_col].dropna()
            if not chunk_end.empty:
                max_val = chunk_end.max()
                end_val = max_val if end_val is None or max_val > end_val else end_val
    except pd.errors.EmptyDataError:
        return None
    except Exception as exc:
        print(f"[warn] Failed to scan {path}: {exc}")
        return None

    if start_val is None or end_val is None:
        return None

    start_dt = ensure_utc(pd.Timestamp(start_val).to_pydatetime())
    end_dt = ensure_utc(pd.Timestamp(end_val).to_pydatetime())

    if end_dt <= start_dt:
        return None

    return start_dt, end_dt


def discover_price_histories(patterns: Iterable[str]) -> Dict[str, PriceHistory]:
    """Find price CSVs matching patterns and derive coverage windows."""
    histories: Dict[str, PriceHistory] = {}
    for pattern in patterns:
        for path in Path().glob(pattern):
            if not path.is_file():
                continue

            symbol = infer_symbol_from_path(path)
            if not symbol:
                continue

            bounds = parse_price_file_time_bounds(path)
            if not bounds:
                continue

            start_dt, end_dt = bounds
            current = histories.get(symbol)
            if current is None or start_dt < current.start or end_dt > current.end:
                histories[symbol] = PriceHistory(symbol=symbol, start=start_dt, end=end_dt, source=path)

    return histories


def load_fmp_name_map(csv_path: Path) -> Dict[str, str]:
    """Load the FMP symbol->name mapping if the CSV exists."""
    if not csv_path.exists():
        return {}

    try:
        df = pd.read_csv(csv_path, usecols=["symbol", "name"])
    except Exception as exc:
        print(f"[warn] Unable to load {csv_path}: {exc}")
        return {}

    mapping: Dict[str, str] = {}
    for row in df.itertuples(index=False):
        symbol = str(row.symbol).upper().strip()
        name = str(row.name).strip()
        if symbol and name:
            mapping[symbol] = name

    return mapping


def clean_token_name(raw_name: str) -> str:
    """Trim common suffixes (e.g., ' USD') from token names."""
    name = raw_name.strip()
    upper = name.upper()
    for suffix in (" USD", " USDT", " TOKEN", " (USDT)", " PERPETUAL"):
        if upper.endswith(suffix):
            name = name[: -len(suffix)].strip()
            upper = name.upper()
    return name


def resolve_search_term(symbol: str, fmp_map: Dict[str, str]) -> str:
    """Determine the Google Trends keyword to use for the symbol."""
    base = extract_base_symbol(symbol)
    if base in MANUAL_NAME_MAP:
        return MANUAL_NAME_MAP[base]

    for suffix in ("USDT", "USD", "USDC", "BUSD", "EUR"):
        key = f"{base}{suffix}"
        if key in fmp_map:
            return clean_token_name(fmp_map[key])

    for key, value in fmp_map.items():
        if key.startswith(base):
            return clean_token_name(value)

    return base


def format_timeframe(start: datetime, end: datetime, fmt: str) -> str:
    """Format a timeframe string compatible with pytrends."""
    start_str = ensure_utc(start).strftime(fmt)
    end_str = ensure_utc(end).strftime(fmt)
    return f"{start_str} {end_str}"


def request_trends(
    pytrends: TrendReq,
    keywords: Iterable[str] | str,
    timeframe: str,
    geo: str,
    *,
    retries: int = 3,
) -> Optional[pd.DataFrame]:
    """Call Google Trends with retry/backoff."""
    payload = [keywords] if isinstance(keywords, str) else list(keywords)
    for attempt in range(retries):
        try:
            pytrends.build_payload(payload, timeframe=timeframe, geo=geo)
            return pytrends.interest_over_time()
        except Exception as exc:  # noqa: BLE001 - surface API errors
            wait = 2 ** attempt
            joined = ", ".join(payload)
            print(
                f"[warn] Failed to fetch [{joined}] @ {timeframe} ({exc}); retrying in {wait}s...",
            )
            time.sleep(wait)
    print(f"[error] Giving up on keywords {payload} @ {timeframe}")
    return None


def iter_time_windows(
    start: datetime,
    end: datetime,
    cfg: Dict[str, timedelta],
) -> Iterable[Tuple[datetime, datetime, str]]:
    """Yield chunked [start, end) windows and formatted timeframe strings."""
    start_utc = ensure_utc(start)
    end_utc = ensure_utc(end)
    window_start = start_utc

    while window_start < end_utc:
        window_end = min(end_utc, window_start + cfg["span"])
        timeframe_end = min(end_utc + cfg["step"], window_end + cfg["step"])
        yield window_start, window_end, format_timeframe(window_start, timeframe_end, cfg["format"])
        if window_end >= end_utc:
            break
        window_start = max(start_utc, window_end - cfg["overlap"])


def collect_trends_by_frequency(
    targets: Iterable[TokenTarget],
    frequency: str,
    geo: str,
    rate_delay: float,
    batch_size: int,
) -> Dict[str, list[pd.DataFrame]]:
    """Batch Google Trends calls across tokens for a single frequency."""
    if frequency not in FETCH_CONFIG:
        raise ValueError(f"Unsupported frequency '{frequency}'")

    cfg = FETCH_CONFIG[frequency]
    sorted_targets = sorted(targets, key=lambda target: target.start)
    results: Dict[str, list[pd.DataFrame]] = {target.symbol: [] for target in sorted_targets}
    if not sorted_targets:
        return results

    pytrends = TrendReq(hl="en-US", tz=0)

    for i in range(0, len(sorted_targets), batch_size):
        batch = sorted_targets[i : i + batch_size]
        keywords = [token.keyword for token in batch]
        batch_start = min(token.start for token in batch)
        batch_end = max(token.end for token in batch)

        for _, _, timeframe in iter_time_windows(batch_start, batch_end, cfg):
            data = request_trends(pytrends, keywords, timeframe, geo)
            if data is not None and not data.empty:
                if "isPartial" in data.columns:
                    data = data[~data["isPartial"]]
                    data = data.drop(columns=["isPartial"])

                for token in batch:
                    column = token.keyword
                    if column not in data.columns:
                        continue

                    series = data[column]
                    if series.empty:
                        continue

                    frame = series.reset_index().rename(
                        columns={"date": "timestamp", column: "interest"},
                    )
                    start_naive = token.start.replace(tzinfo=None)
                    end_naive = token.end.replace(tzinfo=None)
                    frame = frame[
                        (frame["timestamp"] >= start_naive)
                        & (frame["timestamp"] <= end_naive)
                    ]

                    if frame.empty:
                        continue

                    frame["symbol"] = token.symbol
                    frame["search_term"] = token.keyword
                    frame["frequency"] = frequency
                    results[token.symbol].append(frame)

            if rate_delay > 0:
                time.sleep(rate_delay)

    return results


def collect_trends(
    targets: Iterable[TokenTarget],
    geo: str,
    output_dir: Path,
    rate_delay: float,
    *,
    batch_size: int = 5,
) -> None:
    """Fetch hourly and daily Google Trends data for all supplied targets."""
    target_list = list(targets)
    if not target_list:
        print("No tokens to process.")
        return

    for token in target_list:
        print(
            f"[{token.symbol}] Price window {token.start.date()} → {token.end.date()} "
            f"(source: {token.source})",
        )
        print(f"[{token.symbol}] Using search term '{token.keyword}'")

    aggregated: Dict[str, Dict[str, list[pd.DataFrame]]] = {
        token.symbol: {"hourly": [], "daily": []} for token in target_list
    }

    for frequency in ("hourly", "daily"):
        frequency_results = collect_trends_by_frequency(
            target_list,
            frequency,
            geo,
            rate_delay,
            batch_size,
        )
        for symbol, frames in frequency_results.items():
            aggregated[symbol][frequency].extend(frames)

    output_dir.mkdir(parents=True, exist_ok=True)

    for token in target_list:
        symbol = token.symbol
        for frequency in ("hourly", "daily"):
            frames = aggregated[symbol][frequency]
            if not frames:
                print(f"[{symbol}] No {frequency} Google Trends data retrieved.")
                continue

            combined = pd.concat(frames, ignore_index=True)
            combined = combined.drop_duplicates(subset="timestamp", keep="last")
            combined = combined.sort_values("timestamp").reset_index(drop=True)

            symbol_dir = output_dir / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            output_path = symbol_dir / f"google_trends_{frequency}.csv"
            combined.to_csv(output_path, index=False)
            print(f"[{symbol}] Saved {len(combined)} {frequency} rows to {output_path}")



# ---------------------------------------------------------------------------
# CLI


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch hourly and daily Google Trends series for each token with available price history."
        ),
    )
    parser.add_argument(
        "--price-glob",
        action="append",
        dest="price_globs",
        help="Glob pattern(s) to locate price history CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write output CSV files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--geo",
        default=DEFAULT_GEO,
        help="Google Trends geo filter (default: global).",
    )
    parser.add_argument(
        "--rate-delay",
        type=float,
        default=DEFAULT_RATE_DELAY,
        help="Seconds to sleep between Google Trends requests.",
    )
    parser.add_argument(
        "--fmp-csv",
        type=Path,
        default=DEFAULT_FMP_CSV,
        help="Optional CSV providing symbol→name mapping (default: FMP export).",
    )
    parser.add_argument(
        "--tokens",
        nargs="*",
        help="Restrict processing to the given trading symbols (e.g. BTCUSDT ETHUSDT).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of tokens per Google Trends request (Google caps at 5).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    price_patterns = tuple(args.price_globs) if args.price_globs else DEFAULT_PRICE_PATTERNS

    histories = discover_price_histories(price_patterns)
    if not histories:
        print("No price history files matched the supplied patterns.")
        return 1

    if args.tokens:
        requested = {token.upper() for token in args.tokens}
        histories = {
            symbol: history
            for symbol, history in histories.items()
            if symbol.upper() in requested or history.base_symbol in requested
        }
        if not histories:
            print("No price histories remain after applying --tokens filter.")
            return 1

    fmp_map = load_fmp_name_map(args.fmp_csv)
    output_dir = args.output_dir

    capped_batch = max(1, min(args.batch_size, 5))
    if args.batch_size != capped_batch:
        print(f"Batch size adjusted to {capped_batch} to satisfy Google Trends limits.")

    sorted_histories = sorted(histories.items(), key=lambda item: item[0])
    targets = [
        TokenTarget(history=history, keyword=resolve_search_term(symbol, fmp_map))
        for symbol, history in sorted_histories
    ]

    collect_trends(
        targets,
        args.geo,
        output_dir,
        args.rate_delay,
        batch_size=capped_batch,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
