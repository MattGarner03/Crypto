"""Assemble a unified feature table from the Binance spot/futures datasets.

The builder assumes the data collection utilities in this repository have populated
``data/spot_top20`` and ``data/futures_top20`` with CSV snapshots, and optionally
Google Trends series under ``data/google_trends``.  It aligns everything to a common
time grid (hourly by default), engineers short-term and long-term signals, and writes
the result to disk for downstream alpha modelling.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class BuilderConfig:
    data_root: Path = Path("data")
    frequency: str = "1h"
    symbols: Optional[list[str]] = None
    min_history: int = 48
    include_trends: bool = True
    output: Optional[Path] = None
    output_format: str = "parquet"
    spot_dir: str = "spot_top20"
    futures_dir: str = "futures_top20"
    hourly_dir: str = "hourly_top10"
    trends_dir: str = "google_trends"
    min_column_coverage: Optional[float] = 0.95
    csv_compression: Optional[str] = "gzip"


class AlphaDatasetBuilder:
    def __init__(self, config: BuilderConfig) -> None:
        self.config = config
        self.freq = config.frequency
        self.freq_delta = pd.to_timedelta(config.frequency)
        if self.freq_delta <= pd.Timedelta(0):
            raise ValueError(f"Invalid frequency '{config.frequency}'")
        self.periods_per_year = max(
            1,
            int(round(pd.Timedelta(days=365) / self.freq_delta)),
        )
        self.symbol_frames: Dict[str, pd.DataFrame] = {}
        self.coverage_series: Optional[pd.Series] = None
        self.dropped_columns: Optional[pd.Series] = None
        self.always_keep_prefixes: tuple[str, ...] = ("target_",)

    # ------------------------------------------------------------------ #
    # Public API

    def build(self) -> pd.DataFrame:
        symbols = self.config.symbols or self._discover_symbols()
        if not symbols:
            raise RuntimeError("No symbols discovered in the data directories.")

        self.symbol_frames = {}
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            symbol = symbol.upper()
            frame = self._build_symbol_frame(symbol)
            if frame is None or frame.empty:
                continue
            frame = frame.copy()
            frame["symbol"] = symbol
            self.symbol_frames[symbol] = frame.copy()
            frames.append(frame)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, axis=0, ignore_index=False)
        combined.index.name = "timestamp"
        combined = combined.set_index("symbol", append=True)
        combined = combined.reorder_levels(["timestamp", "symbol"])
        combined = combined.sort_index()
        combined = self._apply_coverage_filter(combined)
        return combined

    def _should_force_keep(self, column: str) -> bool:
        return any(column.startswith(prefix) for prefix in self.always_keep_prefixes)

    def _apply_coverage_filter(self, dataset: pd.DataFrame) -> pd.DataFrame:
        coverage = dataset.notna().mean()
        coverage.name = "coverage"
        self.coverage_series = coverage

        threshold = self.config.min_column_coverage
        if threshold is None:
            self.dropped_columns = pd.Series(dtype=float)
            return dataset

        keep_mask = coverage >= threshold
        for column in coverage.index:
            if self._should_force_keep(column):
                keep_mask.loc[column] = True

        kept_columns = coverage.index[keep_mask]
        dropped_columns = coverage.index[~keep_mask]
        self.dropped_columns = coverage.loc[dropped_columns].sort_values()

        if len(kept_columns) == 0:
            # Avoid returning an empty dataset; fall back to original columns.
            self.dropped_columns = pd.Series(dtype=float)
            return dataset

        filtered = dataset.loc[:, kept_columns].copy()
        keep_set = set(kept_columns)
        for symbol, frame in list(self.symbol_frames.items()):
            cols_to_keep = [col for col in frame.columns if col == "symbol" or col in keep_set]
            if cols_to_keep:
                self.symbol_frames[symbol] = frame[cols_to_keep]
            else:
                self.symbol_frames[symbol] = frame[["symbol"]] if "symbol" in frame.columns else frame.iloc[:, 0:0]
        return filtered

    def save(self, dataset: pd.DataFrame, path: Path, fmt: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "parquet":
            dataset.to_parquet(path)
        elif fmt == "csv":
            dataset.reset_index().to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported output format '{fmt}'")

    # ------------------------------------------------------------------ #
    # Symbol discovery and file helpers

    def _discover_symbols(self) -> list[str]:
        root = self.config.data_root
        candidates: set[str] = set()

        for subdir in (self.config.spot_dir, self.config.futures_dir):
            path = root / subdir
            if not path.exists():
                continue
            for child in path.iterdir():
                if child.is_dir():
                    candidates.add(child.name.upper())

        hourly_path = root / self.config.hourly_dir
        if hourly_path.exists():
            suffix = f"_{self.freq}.csv"
            for csv_path in hourly_path.glob(f"*{suffix}"):
                candidates.add(csv_path.stem.replace(suffix, "").upper())

        return sorted(candidates)

    @staticmethod
    def _read_csv(path: Path, *, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            frame = pd.read_csv(path, parse_dates=parse_dates)
        except Exception:
            return pd.DataFrame()
        if frame.empty:
            return frame
        if len(frame.columns) == 1 and frame.columns[0] == "error":
            return pd.DataFrame()
        if "error" in frame.columns and frame.shape[1] == 1:
            return pd.DataFrame()
        return frame

    def _prepare_kline(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "open_time" not in frame.columns:
            return pd.DataFrame()
        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["open_time"], utc=True)
        frame = frame.sort_values("timestamp")
        frame = frame.drop_duplicates(subset="timestamp", keep="last")
        frame = frame.set_index("timestamp")
        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "number_of_trades",
            "taker_base_volume",
            "taker_quote_volume",
        ]
        for column in numeric_cols:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    def _resample_ohlcv(self, frame: pd.DataFrame, target_freq: str) -> pd.DataFrame:
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "quote_asset_volume": "sum",
            "number_of_trades": "sum",
            "taker_base_volume": "sum",
            "taker_quote_volume": "sum",
        }
        resampled = frame.resample(target_freq).agg(agg)
        resampled = resampled.dropna(subset=["close"])
        return resampled

    def _load_spot_klines(self, symbol: str) -> pd.DataFrame:
        base = self.config.data_root / self.config.spot_dir / symbol
        if not base.exists():
            frame = self._load_hourly_history(symbol)
            return frame

        candidates: list[Path] = []
        if self.freq == "1h":
            candidates.append(base / "klines_1h_full_history.csv")
        candidates.append(base / f"klines_{self.freq}.csv")

        for path in candidates:
            frame = self._read_csv(path, parse_dates=["open_time", "close_time"])
            prepared = self._prepare_kline(frame)
            if not prepared.empty:
                return prepared

        # fallback: resample from any available kline file
        available = sorted(base.glob("klines_*.csv"))
        for path in available:
            raw = self._read_csv(path, parse_dates=["open_time", "close_time"])
            prepared = self._prepare_kline(raw)
            if prepared.empty:
                continue
            resampled = self._resample_ohlcv(prepared, self.freq)
            if not resampled.empty:
                return resampled

        return self._load_hourly_history(symbol)

    def _load_hourly_history(self, symbol: str) -> pd.DataFrame:
        path = self.config.data_root / self.config.hourly_dir / f"{symbol}_{self.freq}.csv"
        frame = self._read_csv(path, parse_dates=["open_time", "close_time"])
        return self._prepare_kline(frame)

    def _load_futures_klines(self, symbol: str) -> pd.DataFrame:
        base = self.config.data_root / self.config.futures_dir / symbol
        if not base.exists():
            return pd.DataFrame()

        candidates: Iterable[Path] = [
            base / f"futures_klines_{self.freq}.csv",
        ]
        for path in candidates:
            frame = self._read_csv(path, parse_dates=["open_time", "close_time"])
            prepared = self._prepare_kline(frame)
            if not prepared.empty:
                return prepared

        available = sorted(base.glob("futures_klines_*.csv"))
        for path in available:
            raw = self._read_csv(path, parse_dates=["open_time", "close_time"])
            prepared = self._prepare_kline(raw)
            if prepared.empty:
                continue
            resampled = self._resample_ohlcv(prepared, self.freq)
            if not resampled.empty:
                return resampled

        return pd.DataFrame()

    def _load_simple_time_series(
        self,
        path: Path,
        *,
        time_col: str = "timestamp",
        parse_dates: Optional[list[str]] = None,
        aggregation: Optional[str] = None,
    ) -> pd.DataFrame:
        frame = self._read_csv(path, parse_dates=parse_dates or [time_col])
        if frame.empty or time_col not in frame.columns:
            return pd.DataFrame()
        frame = frame.copy()
        frame[time_col] = pd.to_datetime(frame[time_col], utc=True)
        if aggregation:
            frame[time_col] = frame[time_col].dt.floor(aggregation)
        frame = frame.sort_values(time_col)
        frame = frame.drop_duplicates(subset=time_col, keep="last")
        frame = frame.set_index(time_col)
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="ignore")
        return frame

    def _load_taker_ratios(self, symbol: str) -> pd.DataFrame:
        base = self.config.data_root / self.config.futures_dir / symbol
        path = base / f"taker_long_short_ratio_{self.freq}.csv"
        if not path.exists():
            # prefer 5m or 15m as fallbacks for resampling
            for candidate in ("5m", "15m", "1h", "4h"):
                test_path = base / f"taker_long_short_ratio_{candidate}.csv"
                if test_path.exists():
                    path = test_path
                    break
        frame = self._read_csv(path, parse_dates=["timestamp"])
        if frame.empty:
            return pd.DataFrame()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        frame = frame.set_index("timestamp")
        numeric_cols = ["buySellRatio", "sellVol", "buyVol"]
        for column in numeric_cols:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if "symbol" in frame.columns:
            frame = frame.drop(columns=["symbol"])
        if path.name.endswith(f"{self.freq}.csv"):
            return frame
        return frame.resample(self.freq).last()

    def _load_open_interest(self, symbol: str) -> pd.DataFrame:
        base = self.config.data_root / self.config.futures_dir / symbol
        path = base / f"open_interest_hist_{self.freq}.csv"
        if not path.exists():
            for candidate in ("1h", "5m", "1d"):
                test_path = base / f"open_interest_hist_{candidate}.csv"
                if test_path.exists():
                    path = test_path
                    break
        frame = self._read_csv(path, parse_dates=["timestamp"])
        if frame.empty:
            return pd.DataFrame()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        frame = frame.set_index("timestamp")
        numeric_cols = [
            "sumOpenInterest",
            "sumOpenInterestValue",
            "sumOpenInterestQtd",
            "sumOpenInterestUsd",
        ]
        for column in numeric_cols:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for redundant in ("symbol", "pair", "contractType"):
            if redundant in frame.columns:
                frame = frame.drop(columns=[redundant])
        if path.name.endswith(f"{self.freq}.csv"):
            return frame
        return frame.resample(self.freq).last()

    def _load_premium_series(self, symbol: str, kind: str) -> pd.DataFrame:
        base = self.config.data_root / self.config.futures_dir / symbol
        path = base / f"{kind}_{self.freq}.csv"
        frame = self._read_csv(path, parse_dates=["open_time", "close_time"])
        prepared = self._prepare_kline(frame)
        if prepared.empty:
            return prepared
        return prepared[["close"]].rename(columns={"close": f"{kind}_close"})

    def _load_google_trends(self, symbol: str) -> pd.DataFrame:
        if not self.config.include_trends:
            return pd.DataFrame()
        base = self.config.data_root / self.config.trends_dir / symbol
        if not base.exists():
            return pd.DataFrame()
        hourly = base / "google_trends_hourly.csv"
        daily = base / "google_trends_daily.csv"
        if self.freq_delta <= pd.Timedelta(hours=1) and hourly.exists():
            frame = self._load_simple_time_series(
                hourly,
                time_col="timestamp",
                parse_dates=["timestamp"],
            )
        elif daily.exists():
            frame = self._load_simple_time_series(
                daily,
                time_col="timestamp",
                parse_dates=["timestamp"],
            )
        else:
            return pd.DataFrame()

        if frame.empty:
            return frame
        frame = frame.rename(columns={"interest": "google_trends_interest"})
        if "google_trends_interest" in frame.columns:
            frame["google_trends_interest"] = pd.to_numeric(
                frame["google_trends_interest"],
                errors="coerce",
            )
        return frame[["google_trends_interest"]]

    # ------------------------------------------------------------------ #
    # Feature engineering

    def _build_symbol_frame(self, symbol: str) -> Optional[pd.DataFrame]:
        spot_raw = self._load_spot_klines(symbol)
        if spot_raw.empty:
            return None

        spot_features = self._compute_spot_features(spot_raw)
        if spot_features.empty or len(spot_features) < self.config.min_history:
            return None

        base_index = spot_features.index

        futures_features = self._compute_futures_features(symbol, base_index)
        if not futures_features.empty:
            spot_features = spot_features.join(futures_features, how="left")

        trends = self._load_google_trends(symbol)
        if not trends.empty:
            trends = trends.reindex(base_index, method="ffill")
            spot_features = spot_features.join(trends, how="left")

        spot_features = spot_features.sort_index()
        spot_features = spot_features.loc[~spot_features.index.duplicated(keep="last")]
        spot_features = spot_features[spot_features["spot_close"].notna()]

        self._append_targets(spot_features)
        return spot_features

    def _compute_spot_features(self, spot: pd.DataFrame) -> pd.DataFrame:
        df = spot.copy()
        df = df.sort_index()
        close = pd.to_numeric(df["close"], errors="coerce")
        volume = pd.to_numeric(df.get("volume"), errors="coerce")
        quote_volume = pd.to_numeric(df.get("quote_asset_volume"), errors="coerce")
        taker_base = pd.to_numeric(df.get("taker_base_volume"), errors="coerce")
        taker_quote = pd.to_numeric(df.get("taker_quote_volume"), errors="coerce")

        features = pd.DataFrame(index=df.index)
        features["spot_close"] = close
        features["spot_open"] = pd.to_numeric(df.get("open"), errors="coerce")
        features["spot_high"] = pd.to_numeric(df.get("high"), errors="coerce")
        features["spot_low"] = pd.to_numeric(df.get("low"), errors="coerce")
        features["spot_volume"] = volume
        features["spot_quote_volume"] = quote_volume
        features["spot_taker_buy_ratio"] = self._safe_divide(taker_base, volume)
        features["spot_taker_quote_ratio"] = self._safe_divide(taker_quote, quote_volume)

        features["spot_log_return_1"] = np.log(close / close.shift(1))
        features["spot_log_return_4"] = np.log(close / close.shift(self._steps("4h")))
        features["spot_log_return_24"] = np.log(close / close.shift(self._steps("24h")))

        features["spot_abs_return"] = features["spot_log_return_1"].abs()
        features["spot_return_zscore_24h"] = self._rolling_zscore(
            features["spot_log_return_1"],
            self._steps("24h"),
        )
        features["spot_volume_zscore_24h"] = self._rolling_zscore(
            volume,
            self._steps("24h"),
        )
        features["spot_price_zscore_7d"] = self._rolling_zscore(
            close,
            self._steps("7d"),
        )

        vol_window_24h = self._steps("24h")
        vol_window_7d = self._steps("7d")
        features["spot_volatility_24h"] = (
            features["spot_log_return_1"].rolling(vol_window_24h).std()
            * np.sqrt(self.periods_per_year)
        )
        features["spot_volatility_7d"] = (
            features["spot_log_return_1"].rolling(vol_window_7d).std()
            * np.sqrt(self.periods_per_year)
        )

        features["spot_momentum_7d"] = close.pct_change(self._steps("7d"))
        features["spot_momentum_30d"] = close.pct_change(self._steps("30d"))

        fast_span = max(3, self._steps("12h"))
        slow_span = max(fast_span + 1, self._steps("72h"))
        features["spot_ema_fast"] = close.ewm(span=fast_span, adjust=False).mean()
        features["spot_ema_slow"] = close.ewm(span=slow_span, adjust=False).mean()
        features["spot_trend_regime"] = np.where(
            features["spot_ema_fast"] > features["spot_ema_slow"],
            1,
            -1,
        )
        features["spot_trend_strength"] = self._safe_divide(
            features["spot_ema_fast"],
            features["spot_ema_slow"],
        ) - 1.0

        price_range = features["spot_high"] - features["spot_low"]
        features["spot_range_ratio"] = self._safe_divide(price_range, features["spot_close"])

        return features

    def _compute_futures_features(
        self,
        symbol: str,
        base_index: pd.Index,
    ) -> pd.DataFrame:
        components: list[pd.DataFrame] = []

        fut_klines = self._load_futures_klines(symbol)
        if not fut_klines.empty:
            fut = pd.DataFrame(index=fut_klines.index)
            close = pd.to_numeric(fut_klines["close"], errors="coerce")
            volume = pd.to_numeric(fut_klines.get("volume"), errors="coerce")
            taker_base = pd.to_numeric(fut_klines.get("taker_base_volume"), errors="coerce")
            fut["fut_close"] = close
            fut["fut_volume"] = volume
            fut["fut_taker_buy_ratio"] = self._safe_divide(taker_base, volume)
            fut["fut_log_return_1"] = np.log(close / close.shift(1))
            components.append(fut)

        premium = self._load_premium_series(symbol, "premium_index_klines")
        if not premium.empty:
            components.append(premium)

        index_price = self._load_premium_series(symbol, "index_price_klines")
        if not index_price.empty:
            components.append(index_price.rename(columns={"index_price_klines_close": "index_price_close"}))

        taker = self._load_taker_ratios(symbol)
        if not taker.empty:
            taker = taker.rename(
                columns={
                    "buySellRatio": "fut_taker_buy_sell_ratio",
                    "buyVol": "fut_taker_buy_volume",
                    "sellVol": "fut_taker_sell_volume",
                },
            )
            components.append(taker)

        open_interest = self._load_open_interest(symbol)
        if not open_interest.empty:
            oi = open_interest.rename(
                columns={
                    "sumOpenInterest": "open_interest",
                    "sumOpenInterestValue": "open_interest_value",
                    "sumOpenInterestUsd": "open_interest_usd",
                },
            )
            components.append(oi)

        funding_path = (
            self.config.data_root
            / self.config.futures_dir
            / symbol
            / "funding_rate_history.csv"
        )
        funding = self._read_csv(funding_path, parse_dates=["fundingTime"])
        if not funding.empty and "fundingTime" in funding.columns:
            funding = funding.copy()
            funding["timestamp"] = pd.to_datetime(
                funding["fundingTime"],
                utc=True,
                errors="coerce",
            ).dt.floor(self.freq)
            funding = funding.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            funding = funding.set_index("timestamp")
            funding["fundingRate"] = pd.to_numeric(funding.get("fundingRate"), errors="coerce")
            funding["markPrice"] = pd.to_numeric(funding.get("markPrice"), errors="coerce")
            funding = funding.rename(
                columns={
                    "fundingRate": "funding_rate",
                    "markPrice": "funding_mark_price",
                },
            )
            funding = funding[["funding_rate", "funding_mark_price"]]
            components.append(funding)

        if not components:
            return pd.DataFrame(index=base_index)

        combined = pd.concat(components, axis=1)
        combined = combined.sort_index()
        combined = combined.loc[~combined.index.duplicated(keep="last")]
        combined = combined.reindex(base_index)
        if "symbol" in combined.columns:
            combined = combined.drop(columns=["symbol"])

        if "open_interest" in combined.columns:
            combined["open_interest"] = combined["open_interest"].ffill()
            if "open_interest_value" in combined.columns:
                combined["open_interest_value"] = combined["open_interest_value"].ffill()
            if "open_interest_usd" in combined.columns:
                combined["open_interest_usd"] = combined["open_interest_usd"].ffill()

            combined["open_interest_change"] = combined["open_interest"].pct_change()
            combined["open_interest_zscore_7d"] = self._rolling_zscore(
                combined["open_interest"],
                self._steps("7d"),
            )

        if "funding_rate" in combined.columns:
            combined["funding_rate"] = combined["funding_rate"].ffill()
            combined["funding_rate_change"] = combined["funding_rate"].diff()

        if "fut_taker_buy_volume" in combined.columns and "fut_taker_sell_volume" in combined.columns:
            buy = combined["fut_taker_buy_volume"]
            sell = combined["fut_taker_sell_volume"]
            combined["fut_taker_volume_imbalance"] = self._safe_divide(buy - sell, buy + sell)

        if "premium_index_klines_close" in combined.columns:
            combined = combined.rename(
                columns={"premium_index_klines_close": "fut_premium_index"},
            )

        if "index_price_close" not in combined.columns and "index_price_klines_close" in combined.columns:
            combined = combined.rename(
                columns={"index_price_klines_close": "index_price_close"},
            )

        return combined

    def _append_targets(self, frame: pd.DataFrame) -> None:
        horizon_short = 1
        horizon_long = self._steps("24h")
        close = frame["spot_close"]
        frame["target_return_1_step"] = np.log(close.shift(-horizon_short) / close)
        frame["target_return_24h"] = np.log(close.shift(-horizon_long) / close)
        frame["target_direction_1_step"] = np.sign(frame["target_return_1_step"])
        frame["target_direction_24h"] = np.sign(frame["target_return_24h"])

        if "fut_close" in frame.columns:
            frame["fut_spot_basis_pct"] = self._safe_divide(frame["fut_close"], close) - 1.0

        if "fut_close" in frame.columns and "index_price_close" in frame.columns:
            frame["fut_index_basis_pct"] = self._safe_divide(
                frame["fut_close"],
                frame["index_price_close"],
            ) - 1.0

    # ------------------------------------------------------------------ #
    # Math helpers

    def _steps(self, duration: str) -> int:
        delta = pd.to_timedelta(duration)
        if delta <= pd.Timedelta(0):
            return 1
        steps = int(round(delta / self.freq_delta))
        return max(1, steps)

    @staticmethod
    def _safe_divide(
        numerator: Optional[pd.Series],
        denominator: Optional[pd.Series],
    ) -> pd.Series:
        if numerator is None and denominator is None:
            return pd.Series(dtype=float)
        if numerator is None:
            idx = denominator.index if isinstance(denominator, pd.Series) else None
            return pd.Series(np.nan, index=idx, dtype=float)
        if denominator is None:
            idx = numerator.index if isinstance(numerator, pd.Series) else None
            return pd.Series(np.nan, index=idx, dtype=float)
        numerator = numerator.astype(float)
        denominator = denominator.astype(float)
        denominator = denominator.replace({0.0: np.nan})
        return numerator / denominator

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        if window <= 1:
            return pd.Series(index=series.index, dtype=float)
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        z = (series - rolling_mean) / rolling_std
        return z.replace({np.inf: np.nan, -np.inf: np.nan})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct a feature table from the collected Binance datasets.",
    )
    parser.add_argument(
        "--frequency",
        default="1h",
        help="Sampling frequency for the feature grid (default: 1h).",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Optional list of symbols to include (defaults to everything discovered).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory containing spot/futures/trends data (default: data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination file for the assembled feature table.",
    )
    parser.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Output format (default: parquet).",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=48,
        help="Minimum number of rows required per symbol (default: 48).",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.95,
        help=(
            "Minimum non-null ratio required to retain a column (default: 0.95). "
            "Set to a negative value to skip filtering."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of symbols to export individually (default: 10).",
    )
    parser.add_argument(
        "--alpha-dir",
        type=Path,
        help="Directory for per-symbol feature exports (default: data/alpha_features).",
    )
    parser.add_argument(
        "--csv-compression",
        choices=("none", "gzip", "bz2", "zip", "xz"),
        default="gzip",
        help="Compression to apply to per-symbol CSV exports (default: gzip).",
    )
    parser.add_argument(
        "--skip-trends",
        action="store_true",
        help="Skip Google Trends series even if available.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = [symbol.upper() for symbol in args.symbols] if args.symbols else None
    min_cov = None if args.min_coverage is None or args.min_coverage < 0 else float(args.min_coverage)
    if min_cov is not None:
        min_cov = max(0.0, min(1.0, min_cov))
    csv_comp = args.csv_compression.lower() if args.csv_compression else "gzip"
    if csv_comp == "none":
        csv_comp = None
    config = BuilderConfig(
        data_root=args.data_root,
        frequency=args.frequency,
        symbols=symbols,
        min_history=args.min_history,
        include_trends=not args.skip_trends,
        output=args.output,
        output_format=args.format,
        min_column_coverage=min_cov,
        csv_compression=csv_comp,
    )
    builder = AlphaDatasetBuilder(config)
    dataset = builder.build()
    if dataset.empty:
        print("No features generated.")
        return 1

    symbol_frames = getattr(builder, "symbol_frames", {}) or {}
    if symbol_frames:
        sorted_symbols = sorted(
            symbol_frames.items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )
        requested_top = args.top_n if args.top_n is not None else 10
        if requested_top is None or requested_top <= 0:
            top_n = len(sorted_symbols)
        else:
            top_n = min(requested_top, len(sorted_symbols))
        top_symbols = sorted_symbols[:top_n]

        alpha_dir = args.alpha_dir or (config.data_root / "alpha_features")
        alpha_dir.mkdir(parents=True, exist_ok=True)

        coverage_records: list[dict[str, object]] = []
        coverage_path: Optional[Path] = None
        missing_path: Optional[Path] = None
        drop_info_path: Optional[Path] = None
        columns_union = sorted(
            {
                column
                for _, frame in top_symbols
                for column in frame.columns
                if column != "symbol"
            },
        )

        for symbol, frame in top_symbols:
            frame_to_save = frame.copy()
            frame_to_save = frame_to_save.sort_index()
            frame_to_save = frame_to_save.loc[~frame_to_save.index.duplicated(keep="last")]
            total_rows = len(frame_to_save)

            export_frame = (
                frame_to_save.reset_index()
                .rename(columns={"index": "timestamp"})
                .sort_values("timestamp")
            )
            columns_order = ["timestamp"]
            if "symbol" in export_frame.columns:
                columns_order.append("symbol")
            columns_order.extend(
                column for column in export_frame.columns if column not in columns_order
            )
            export_frame = export_frame[columns_order]

            base_name = f"{symbol}_{config.frequency}"
            parquet_path = alpha_dir / f"{base_name}.parquet"

            compression = config.csv_compression
            csv_extension = ".csv"
            csv_compression_arg: Optional[object] = None
            if compression:
                comp = compression.lower()
                extension_map = {
                    "gzip": ".csv.gz",
                    "bz2": ".csv.bz2",
                    "xz": ".csv.xz",
                    "zip": ".zip",
                }
                csv_extension = extension_map.get(comp, ".csv")
                if comp == "zip":
                    csv_compression_arg = {
                        "method": "zip",
                        "archive_name": f"{base_name}.csv",
                    }
                else:
                    csv_compression_arg = comp
            csv_path = alpha_dir / f"{base_name}{csv_extension}"

            export_frame.to_parquet(parquet_path, index=False)
            csv_kwargs = {"index": False}
            if csv_compression_arg is not None:
                csv_kwargs["compression"] = csv_compression_arg
            export_frame.to_csv(csv_path, **csv_kwargs)
            if compression and csv_extension != ".csv":
                legacy_csv = alpha_dir / f"{base_name}.csv"
                if legacy_csv.exists():
                    try:
                        legacy_csv.unlink()
                    except OSError:
                        pass

            for column in columns_union:
                if column in frame_to_save.columns:
                    series = frame_to_save[column]
                    non_null = int(series.notna().sum())
                else:
                    non_null = 0
                coverage_pct = float(non_null / total_rows) if total_rows else float("nan")
                coverage_records.append(
                    {
                        "symbol": symbol,
                        "column": column,
                        "present": column in frame_to_save.columns,
                        "non_null_count": non_null,
                        "total_rows": total_rows,
                        "coverage_pct": coverage_pct,
                    },
                )

        if coverage_records:
            coverage_df = pd.DataFrame(coverage_records)
            coverage_df["coverage_pct"] = coverage_df["coverage_pct"].round(6)
            coverage_df["has_data"] = coverage_df["non_null_count"] > 0
            coverage_path = alpha_dir / f"column_coverage_top{len(top_symbols)}.csv"
            coverage_df.to_csv(coverage_path, index=False)

            missing_df = coverage_df[~coverage_df["has_data"]].copy()
            if not missing_df.empty:
                missing_path = alpha_dir / f"column_gaps_top{len(top_symbols)}.csv"
                missing_df.to_csv(missing_path, index=False)

        dropped_columns = getattr(builder, "dropped_columns", None)
        if dropped_columns is not None and not dropped_columns.empty:
            drop_df = dropped_columns.reset_index()
            drop_df.columns = ["column", "coverage_pct"]
            drop_df["coverage_pct"] = drop_df["coverage_pct"].round(6)
            drop_info_path = alpha_dir / "dropped_columns.csv"
            drop_df.to_csv(drop_info_path, index=False)

        exported_symbols = ", ".join(symbol for symbol, _ in top_symbols)
        print(
            f"Exported per-symbol features for {len(top_symbols)} symbols "
            f"({exported_symbols}) to {alpha_dir.resolve()} "
            f"(CSV compression: {config.csv_compression or 'none'})",
        )
        if coverage_path is not None:
            print(f"Column coverage summary: {coverage_path}")
        if missing_path is not None:
            print(f"Columns with no data: {missing_path}")
        if drop_info_path is not None:
            print(f"Dropped columns (coverage below threshold): {drop_info_path}")

    target_ext = config.output_format
    output_path = config.output or (
        config.data_root / f"alpha_features_{config.frequency}.{target_ext}"
    )
    builder.save(dataset, output_path, config.output_format)
    print(f"Wrote {len(dataset)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
