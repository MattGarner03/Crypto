from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from data_pull import BinanceData, top_market_cap_spot_symbols

Number = float
BookSide = Sequence[Tuple[Number, Number]]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _seconds_until_next_interval(ts: datetime, interval_minutes: float, align_to_hour: bool) -> float:
    if align_to_hour:
        next_hour = (ts + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        return max(0.0, (next_hour - ts).total_seconds())
    interval_seconds = max(interval_minutes, 0.1) * 60.0
    return interval_seconds


@dataclass
class OrderBookCompressor:
    client: BinanceData
    levels: int = 10
    price_step_multiple: int = 10
    depth_limit: int = 500
    _tick_cache: Dict[str, float] = field(default_factory=dict)

    def build_row(self, symbol: str, timestamp: datetime) -> Dict[str, Number]:
        symbol = symbol.upper()
        depth = self.client.spot_depth(symbol, limit=self.depth_limit)
        tick_size = self._get_tick_size(symbol)
        price_step = tick_size * max(self.price_step_multiple, 1)
        if price_step <= 0:
            price_step = tick_size if tick_size > 0 else 1e-8

        bids = self._convert_levels(depth.get("bids", []))
        asks = self._convert_levels(depth.get("asks", []))

        return self._build_feature_row(
            symbol=symbol,
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            tick_size=tick_size,
            price_step=price_step,
        )

    def _convert_levels(self, raw_levels: Iterable[Sequence[str]]) -> List[Tuple[float, float]]:
        converted: List[Tuple[float, float]] = []
        for entry in raw_levels:
            if len(entry) < 2:
                continue
            try:
                price = float(entry[0])
                qty = float(entry[1])
            except (TypeError, ValueError):
                continue
            if math.isnan(price) or math.isnan(qty) or qty <= 0:
                continue
            converted.append((price, qty))
        return converted

    def _build_feature_row(
        self,
        *,
        symbol: str,
        timestamp: datetime,
        bids: BookSide,
        asks: BookSide,
        tick_size: float,
        price_step: float,
    ) -> Dict[str, Number]:
        row: Dict[str, Number] = {}
        row["timestamp"] = timestamp.isoformat()
        row["timestamp_ms"] = int(timestamp.timestamp() * 1000)
        row["symbol"] = symbol
        row["tick_size"] = tick_size
        row["price_step"] = price_step

        best_bid = bids[0][0] if bids else math.nan
        best_ask = asks[0][0] if asks else math.nan
        row["best_bid"] = best_bid
        row["best_ask"] = best_ask

        if not math.isnan(best_bid) and not math.isnan(best_ask):
            row["spread"] = best_ask - best_bid
            row["mid_price"] = (best_bid + best_ask) / 2
        else:
            row["spread"] = math.nan
            row["mid_price"] = math.nan

        agg_bids = self._compress_side(bids, price_step=price_step, side_name="bid")
        agg_asks = self._compress_side(asks, price_step=price_step, side_name="ask")

        row["bid_book_qty"] = agg_bids[-1]["cum_qty"] if agg_bids else 0.0
        row["ask_book_qty"] = agg_asks[-1]["cum_qty"] if agg_asks else 0.0
        row["bid_book_notional"] = agg_bids[-1]["cum_notional"] if agg_bids else 0.0
        row["ask_book_notional"] = agg_asks[-1]["cum_notional"] if agg_asks else 0.0

        total_qty = row["bid_book_qty"] + row["ask_book_qty"]
        row["orderbook_imbalance"] = (
            row["bid_book_qty"] / total_qty if total_qty > 0 else math.nan
        )

        self._flatten_side(row, "bid", agg_bids)
        self._flatten_side(row, "ask", agg_asks)
        return row

    def _flatten_side(self, row: Dict[str, Number], prefix: str, levels: List[Dict[str, Number]]) -> None:
        for idx in range(self.levels):
            key = f"{prefix}_level_{idx + 1}"
            level = levels[idx] if idx < len(levels) else None
            row[f"{key}_price_vwap"] = level["vwap"] if level else math.nan
            row[f"{key}_qty"] = level["qty"] if level else 0.0
            row[f"{key}_cum_qty"] = level["cum_qty"] if level else 0.0
            row[f"{key}_notional"] = level["notional"] if level else 0.0
            row[f"{key}_price_min"] = level["price_min"] if level else math.nan
            row[f"{key}_price_max"] = level["price_max"] if level else math.nan

    def _compress_side(
        self,
        side: BookSide,
        *,
        price_step: float,
        side_name: str,
    ) -> List[Dict[str, Number]]:
        if not side:
            return []

        best_price = side[0][0]
        buckets: List[Dict[str, Number]] = []
        for price, qty in side:
            distance = (best_price - price) if side_name == "bid" else (price - best_price)
            if distance < 0:
                distance = 0.0
            bucket_index = int(distance // price_step) if price_step > 0 else 0
            if bucket_index >= self.levels:
                continue
            while len(buckets) <= bucket_index:
                buckets.append(
                    {
                        "qty": 0.0,
                        "notional": 0.0,
                        "price_min": math.nan,
                        "price_max": math.nan,
                    }
                )
            bucket = buckets[bucket_index]
            bucket["qty"] += qty
            bucket["notional"] += price * qty
            bucket["price_min"] = (
                price if math.isnan(bucket["price_min"]) else min(bucket["price_min"], price)
            )
            bucket["price_max"] = (
                price if math.isnan(bucket["price_max"]) else max(bucket["price_max"], price)
            )

        cumulative_qty = 0.0
        cumulative_notional = 0.0
        compressed: List[Dict[str, Number]] = []
        for bucket in buckets[: self.levels]:
            qty = bucket["qty"]
            notional = bucket["notional"]
            cumulative_qty += qty
            cumulative_notional += notional
            compressed.append(
                {
                    "qty": qty,
                    "notional": notional,
                    "vwap": notional / qty if qty > 0 else math.nan,
                    "price_min": bucket["price_min"],
                    "price_max": bucket["price_max"],
                    "cum_qty": cumulative_qty,
                    "cum_notional": cumulative_notional,
                }
            )
        return compressed

    def _get_tick_size(self, symbol: str) -> float:
        cached = self._tick_cache.get(symbol)
        if cached:
            return cached
        info = self.client.spot_exchange_info(symbol)
        symbols = info.get("symbols") if isinstance(info, dict) else None
        if not symbols:
            raise RuntimeError(f"Unable to locate exchange info for {symbol}")
        entry = symbols[0]
        filters = entry.get("filters", [])
        tick_size = None
        for fil in filters:
            if fil.get("filterType") == "PRICE_FILTER":
                tick_size = float(fil.get("tickSize", 0))
                break
        if not tick_size or tick_size <= 0:
            tick_size = float(entry.get("tickSize", 0) or 0)
        if tick_size <= 0:
            tick_size = 1e-8
        self._tick_cache[symbol] = tick_size
        return tick_size


def _write_symbol_row(row: Dict[str, Number], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _resolve_symbols(args: argparse.Namespace) -> List[str]:
    if args.symbols:
        return [symbol.upper() for symbol in args.symbols]
    symbols = top_market_cap_spot_symbols(quote_asset=args.quote_asset, limit=args.top)
    return [symbol.upper() for symbol in symbols]


def run_sampler(args: argparse.Namespace) -> None:
    client = BinanceData()
    symbols = _resolve_symbols(args)
    if not symbols:
        raise SystemExit("No symbols resolved for sampling.")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    compressor = OrderBookCompressor(
        client=client,
        levels=args.levels,
        price_step_multiple=args.price_step_multiple,
        depth_limit=args.depth_limit,
    )

    while True:
        snapshot_ts = _now_utc()
        rows: List[Dict[str, Number]] = []
        for symbol in symbols:
            row = compressor.build_row(symbol, snapshot_ts)
            rows.append(row)

            symbol_dir = output_root / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            symbol_csv = symbol_dir / "orderbook_levels.csv"
            _write_symbol_row(row, symbol_csv)

        if rows:
            combined_path = output_root / "orderbook_levels.csv"
            df = pd.DataFrame(rows)
            df.to_csv(combined_path, mode="a", header=not combined_path.exists(), index=False)

        if not args.loop:
            break

        sleep_seconds = _seconds_until_next_interval(snapshot_ts, args.interval_minutes, args.align_to_hour)
        time.sleep(sleep_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect compressed spot order-book snapshots for Binance symbols "
            "and persist them as hourly feature rows."
        )
    )
    parser.add_argument("--symbols", nargs="+", help="Explicit symbols to sample (e.g. BTCUSDT ETHUSDT).")
    parser.add_argument(
        "--quote-asset",
        default="USDT",
        help="Quote asset used when resolving the top market-cap symbols (default: USDT).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top market-cap symbols to sample when --symbols is not provided (default: 10).",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=10,
        help="Number of aggregated price levels to retain for bids and asks.",
    )
    parser.add_argument(
        "--price-step-multiple",
        type=int,
        default=10,
        help="Multiple of the Binance tick size used when grouping price levels.",
    )
    parser.add_argument(
        "--depth-limit",
        type=int,
        default=500,
        help="Number of raw price levels to request from the depth endpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/orderbook_levels",
        help="Root directory for output CSV snapshots.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Continuously collect snapshots at the requested interval.",
    )
    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=60.0,
        help="Interval between snapshots when looping (default: 60 minutes).",
    )
    parser.add_argument(
        "--align-to-hour",
        action="store_true",
        help="When looping, align the next snapshot to the top-of-the-hour boundary.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_sampler(args)


if __name__ == "__main__":
    main()
