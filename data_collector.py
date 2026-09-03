"""
Historical Data Harvester & Preprocessor for Live Price Action Scanner.

Fetches large volumes of Binance historical Kline/candlestick data across multiple symbols
and timeframes using pagination, fallback endpoints, rate-limit backoff, and robust data cleaning.
Saves validated datasets in CSV and Parquet formats in `data/historical/`.
"""

import os
import sys
import time
import math
import random
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import requests
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from config import (
    BASE_DIR,
    BINANCE_REST_BASE,
    BINANCE_REST_FALLBACKS,
    BINANCE_API_KEY,
    logger,
)

# Optional Parquet support
try:
    import pyarrow
    PARQUET_SUPPORTED = True
except ImportError:
    PARQUET_SUPPORTED = False

console = Console()

# Interval to milliseconds mapping
INTERVAL_MS_MAP: Dict[str, int] = {
    "1m": 60 * 1000,
    "3m": 3 * 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "2h": 2 * 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "6h": 6 * 60 * 60 * 1000,
    "8h": 8 * 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
    "3d": 3 * 24 * 60 * 60 * 1000,
    "1w": 7 * 24 * 60 * 60 * 1000,
    "1M": 30 * 24 * 60 * 60 * 1000,
}

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
DEFAULT_TIMEFRAMES = ["15m", "1h", "4h"]
DEFAULT_TARGET_CANDLES = 5000
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "historical"


class DataCollector:
    """
    Production-grade historical candlestick harvester and caching pipeline for Binance.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
        use_api_key: bool = True,
        endpoints: Optional[List[str]] = None,
        request_delay: float = 0.1,
        max_retries: int = 5,
        session: Optional[requests.Session] = None,
    ):
        self.symbols = [s.upper().strip() for s in (symbols or DEFAULT_SYMBOLS)]
        self.timeframes = [t.strip() for t in (timeframes or DEFAULT_TIMEFRAMES)]
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_api_key = use_api_key
        self.api_key = BINANCE_API_KEY if use_api_key else ""
        self.endpoints = endpoints or ([BINANCE_REST_BASE] + list(BINANCE_REST_FALLBACKS))
        self.current_endpoint_idx = 0
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.session = session or requests.Session()

        # Weight monitoring
        self.last_used_weight_1m = 0

    @property
    def current_endpoint(self) -> str:
        return self.endpoints[self.current_endpoint_idx % len(self.endpoints)]

    def rotate_endpoint(self) -> str:
        self.current_endpoint_idx = (self.current_endpoint_idx + 1) % len(self.endpoints)
        new_ep = self.current_endpoint
        logger.info(f"Rotated to fallback Binance endpoint: {new_ep}")
        return new_ep

    def fetch_klines_chunk(
        self,
        symbol: str,
        interval: str,
        limit: int = 1000,
        end_time: Optional[int] = None,
        start_time: Optional[int] = None,
    ) -> List[List[Any]]:
        """
        Fetches a single chunk of klines (up to 1000 candles) with fallback rotation,
        exponential backoff on 429/418, and rate-limit header parsing.
        """
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if end_time is not None:
            params["endTime"] = int(end_time)
        if start_time is not None:
            params["startTime"] = int(start_time)

        headers: Dict[str, str] = {
            "User-Agent": "PriceActionScanner/1.0",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key

        retries = 0
        backoff_delay = 2.0

        while retries <= self.max_retries:
            base_url = self.current_endpoint
            url = f"{base_url}/api/v3/klines"

            # Check if rate limit weight is nearing safety threshold
            if self.last_used_weight_1m >= 1000:
                logger.warning(f"Binance 1m weight at {self.last_used_weight_1m}/1200. Pausing 5s to cool down.")
                time.sleep(5.0)

            try:
                response = self.session.get(url, params=params, headers=headers, timeout=12)

                # Track rate limit headers
                weight_hdr = response.headers.get("x-mbx-used-weight-1m")
                if weight_hdr:
                    try:
                        self.last_used_weight_1m = int(weight_hdr)
                    except ValueError:
                        pass

                # Success
                if response.status_code == 200:
                    data = response.json()
                    if not isinstance(data, list):
                        raise ValueError(f"Expected JSON list from {url}, got: {type(data)}")
                    # Brief inter-request courtesy sleep
                    if self.request_delay > 0:
                        time.sleep(self.request_delay)
                    return data

                # Rate Limit Hit: HTTP 429 or 418
                elif response.status_code in (429, 418):
                    retry_after = response.headers.get("Retry-After")
                    wait_time = int(retry_after) if retry_after and retry_after.isdigit() else backoff_delay
                    wait_time += random.uniform(0.5, 1.5)
                    logger.warning(
                        f"Rate limit hit ({response.status_code}) on {base_url}. Backing off {wait_time:.1f}s. "
                        f"Used weight: {self.last_used_weight_1m}"
                    )
                    time.sleep(wait_time)
                    backoff_delay = min(backoff_delay * 2.0, 60.0)
                    self.rotate_endpoint()
                    retries += 1

                # Server Errors (5xx)
                elif 500 <= response.status_code < 600:
                    logger.warning(f"Server error {response.status_code} from {base_url}. Retrying with fallback...")
                    self.rotate_endpoint()
                    time.sleep(backoff_delay)
                    backoff_delay = min(backoff_delay * 1.5, 30.0)
                    retries += 1

                # Client Errors (4xx other than 429/418)
                else:
                    error_text = response.text[:200]
                    logger.error(f"Binance API returned error {response.status_code} from {url}: {error_text}")
                    raise requests.HTTPError(f"HTTP {response.status_code}: {error_text}", response=response)

            except (requests.ConnectionError, requests.Timeout) as net_err:
                logger.warning(f"Network error accessing {base_url}: {net_err}. Rotating endpoint...")
                self.rotate_endpoint()
                time.sleep(backoff_delay)
                backoff_delay = min(backoff_delay * 1.5, 30.0)
                retries += 1

            except Exception as e:
                logger.warning(f"Unexpected error requesting {url}: {e}")
                self.rotate_endpoint()
                time.sleep(backoff_delay)
                retries += 1

        raise ConnectionError(
            f"Exceeded max retries ({self.max_retries}) for {symbol} {interval} across all endpoints."
        )

    def clean_and_validate(
        self,
        raw_candles: List[List[Any]],
        interval: str,
        drop_incomplete: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Parses raw Binance kline records into a typed, validated, deduplicated, sorted DataFrame.

        Validations:
          1. Required columns & numeric types
          2. Deduplication on timestamp
          3. Strict ascending timestamp order
          4. High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close, Volume >= 0
          5. Verification of missing intervals / gaps
          6. Exclusion of incomplete (currently forming) candle if requested
        """
        raw_columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ]

        if not raw_candles:
            empty_df = pd.DataFrame(
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades", "taker_buy_base",
                    "taker_buy_quote", "is_closed"
                ]
            )
            return empty_df, {"valid": False, "reason": "No raw candle data provided"}

        df = pd.DataFrame(raw_candles, columns=raw_columns)

        # Drop ignore column
        if "ignore" in df.columns:
            df.drop(columns=["ignore"], inplace=True)

        # Numeric conversions
        float_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype(np.int64)
        df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce").astype(np.int64)
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype(np.int64)

        # Drop any row where essential OHLCV is NaN
        initial_count = len(df)
        df.dropna(subset=["open_time", "open", "high", "low", "close", "volume"], inplace=True)
        dropped_nans = initial_count - len(df)

        # Remove incomplete currently forming candle if close_time > now_ms
        current_time_ms = int(time.time() * 1000)
        incomplete_dropped = 0
        if drop_incomplete and not df.empty:
            last_close = df.iloc[-1]["close_time"]
            if last_close > current_time_ms:
                df = df.iloc[:-1].copy()
                incomplete_dropped = 1

        # Deduplication on open_time
        pre_dedup = len(df)
        df.drop_duplicates(subset=["open_time"], keep="last", inplace=True)
        duplicates_removed = pre_dedup - len(df)

        # Strict ascending sort
        df.sort_values(by="open_time", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Price sanity validation
        invalid_mask = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"]) |
            (df["volume"] < 0)
        )
        invalid_rows_count = int(invalid_mask.sum())
        if invalid_rows_count > 0:
            logger.warning(f"Found {invalid_rows_count} price anomaly rows violating H>=L or V>=0. Filtering out.")
            df = df[~invalid_mask].copy().reset_index(drop=True)

        # Format timestamps
        # Convert ms timestamps to datetime UTC strings (YYYY-MM-DD HH:MM:SS)
        dt_series = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        dt_close_series = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        df["timestamp"] = dt_series.dt.strftime("%Y-%m-%d %H:%M:%S")
        df["close_time"] = dt_close_series.dt.strftime("%Y-%m-%d %H:%M:%S")
        df["is_closed"] = True

        # Reorder columns
        final_columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "is_closed"
        ]
        df_final = df[final_columns].copy()

        # Gap / Continuity Analysis
        gap_info = self._analyze_continuity(df["open_time"], interval)

        integrity_report = {
            "valid": True,
            "candle_count": len(df_final),
            "dropped_nans": dropped_nans,
            "incomplete_dropped": incomplete_dropped,
            "duplicates_removed": duplicates_removed,
            "invalid_price_rows": invalid_rows_count,
            "start_time": df_final.iloc[0]["timestamp"] if not df_final.empty else None,
            "end_time": df_final.iloc[-1]["timestamp"] if not df_final.empty else None,
            "gap_count": gap_info["gap_count"],
            "missing_candles_est": gap_info["missing_candles_est"],
            "continuity_pct": gap_info["continuity_pct"],
        }

        return df_final, integrity_report

    def _analyze_continuity(self, open_times: pd.Series, interval: str) -> Dict[str, Any]:
        """Calculates timestamp deltas and detects gaps in consecutive candles."""
        if len(open_times) < 2:
            return {"gap_count": 0, "missing_candles_est": 0, "continuity_pct": 100.0}

        expected_step = INTERVAL_MS_MAP.get(interval)
        if not expected_step:
            return {"gap_count": 0, "missing_candles_est": 0, "continuity_pct": 100.0}

        deltas = open_times.diff().iloc[1:]
        gap_mask = deltas > (expected_step * 1.5)
        gap_count = int(gap_mask.sum())

        missing_candles_est = 0
        if gap_count > 0:
            excess_ms = (deltas[gap_mask] - expected_step).sum()
            missing_candles_est = int(math.ceil(excess_ms / expected_step))

        total_expected_candles = len(open_times) + missing_candles_est
        continuity_pct = (
            (len(open_times) / total_expected_candles * 100.0)
            if total_expected_candles > 0
            else 100.0
        )

        return {
            "gap_count": gap_count,
            "missing_candles_est": missing_candles_est,
            "continuity_pct": round(continuity_pct, 2),
        }

    def harvest_symbol_timeframe(
        self,
        symbol: str,
        interval: str,
        target_candles: int = DEFAULT_TARGET_CANDLES,
        drop_incomplete: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Path, Optional[Path]]:
        """
        Paginates backward using Binance REST API /api/v3/klines (limit=1000)
        until target_candles is satisfied. Saves CSV & Parquet files.
        """
        logger.info(f"Starting harvest for {symbol} [{interval}] (Target: {target_candles} candles)...")
        collected_raw: List[List[Any]] = []
        end_time: Optional[int] = None

        needed = target_candles + (10 if drop_incomplete else 0)
        chunk_limit = 1000

        while len(collected_raw) < needed:
            current_batch_limit = min(chunk_limit, needed - len(collected_raw) + 100)
            batch = self.fetch_klines_chunk(
                symbol=symbol,
                interval=interval,
                limit=current_batch_limit,
                end_time=end_time,
            )

            if not batch:
                logger.info(f"No further candles returned for {symbol} [{interval}]. Reached market inception.")
                break

            # Insert batch in front to preserve chronological order (since we paginate backward)
            collected_raw = batch + collected_raw

            earliest_open_time = batch[0][0]
            # Set end_time to 1 ms before earliest candle in this batch
            end_time = earliest_open_time - 1

            logger.debug(
                f"{symbol} [{interval}]: Retrieved batch of {len(batch)}. Total raw: {len(collected_raw)}"
            )

            if len(batch) < current_batch_limit:
                logger.info(f"Reached beginning of historical records for {symbol} [{interval}].")
                break

        # Clean, validate, deduplicate
        df_clean, report = self.clean_and_validate(
            collected_raw, interval=interval, drop_incomplete=drop_incomplete
        )

        # Truncate to target_candles if we fetched more
        if len(df_clean) > target_candles:
            df_clean = df_clean.tail(target_candles).reset_index(drop=True)
            report["candle_count"] = len(df_clean)
            report["start_time"] = df_clean.iloc[0]["timestamp"]
            report["end_time"] = df_clean.iloc[-1]["timestamp"]

        # Save datasets
        csv_path, parquet_path = self.save_dataset(df_clean, symbol, interval)

        report["csv_path"] = str(csv_path)
        report["csv_size_kb"] = round(csv_path.stat().st_size / 1024, 2)
        if parquet_path and parquet_path.exists():
            report["parquet_path"] = str(parquet_path)
            report["parquet_size_kb"] = round(parquet_path.stat().st_size / 1024, 2)
        else:
            report["parquet_path"] = None
            report["parquet_size_kb"] = 0.0

        logger.info(
            f"Successfully saved {symbol} [{interval}]: {len(df_clean)} candles -> {csv_path.name} "
            f"({report['start_time']} to {report['end_time']})"
        )

        return df_clean, report, csv_path, parquet_path

    def save_dataset(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
    ) -> Tuple[Path, Optional[Path]]:
        """Saves DataFrame as clean CSV and Parquet files in output_dir."""
        csv_filename = f"{symbol}_{interval}.csv"
        csv_path = self.output_dir / csv_filename
        df.to_csv(csv_path, index=False)

        parquet_path: Optional[Path] = None
        if PARQUET_SUPPORTED:
            parquet_filename = f"{symbol}_{interval}.parquet"
            parquet_path = self.output_dir / parquet_filename
            df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")

        return csv_path, parquet_path

    def harvest_all(
        self,
        target_candles: int = DEFAULT_TARGET_CANDLES,
        drop_incomplete: bool = True,
    ) -> Dict[str, Any]:
        """Harvests all configured symbols and timeframes sequentially."""
        total_tasks = len(self.symbols) * len(self.timeframes)
        results: Dict[str, Dict[str, Any]] = {}
        grand_total_candles = 0

        console.print(
            f"\n[bold cyan]Binance Historical Data Harvester[/bold cyan] — "
            f"Harvesting [green]{len(self.symbols)} symbols[/green] × "
            f"[green]{len(self.timeframes)} timeframes[/green] (Target: [yellow]{target_candles:,}[/yellow] candles/pair)\n"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            overall_task = progress.add_task("[yellow]Overall Progress", total=total_tasks)

            for sym in self.symbols:
                results[sym] = {}
                for tf in self.timeframes:
                    pair_label = f"{sym} [{tf}]"
                    progress.update(overall_task, description=f"[cyan]Harvesting {pair_label}...")

                    try:
                        _, report, csv_p, pq_p = self.harvest_symbol_timeframe(
                            symbol=sym,
                            interval=tf,
                            target_candles=target_candles,
                            drop_incomplete=drop_incomplete,
                        )
                        results[sym][tf] = report
                        grand_total_candles += report["candle_count"]
                    except Exception as e:
                        logger.error(f"Failed harvesting {pair_label}: {e}", exc_info=True)
                        results[sym][tf] = {"valid": False, "error": str(e), "candle_count": 0}

                    progress.advance(overall_task)

        return {
            "results": results,
            "grand_total_candles": grand_total_candles,
            "symbols": self.symbols,
            "timeframes": self.timeframes,
            "target_candles": target_candles,
            "output_dir": str(self.output_dir),
        }

    def print_summary_table(self, harvest_data: Dict[str, Any]) -> None:
        """Renders an attractive summary table of the harvested data."""
        table = Table(
            title="Binance Historical Data Harvesting Summary",
            header_style="bold magenta",
            show_lines=True,
        )
        table.add_column("Symbol", style="bold yellow", justify="center")
        table.add_column("Interval", style="cyan", justify="center")
        table.add_column("Candles", style="green", justify="right")
        table.add_column("Start Date (UTC)", style="white", justify="center")
        table.add_column("End Date (UTC)", style="white", justify="center")
        table.add_column("Continuity", style="bold green", justify="center")
        table.add_column("CSV Size", style="dim", justify="right")
        table.add_column("Parquet Size", style="dim", justify="right")
        table.add_column("Integrity", style="bold green", justify="center")

        results = harvest_data.get("results", {})
        total_candles = 0

        for sym, tfs in results.items():
            for tf, rep in tfs.items():
                if rep.get("valid", False):
                    count = rep.get("candle_count", 0)
                    total_candles += count
                    cont_pct = rep.get("continuity_pct", 100.0)
                    cont_str = f"{cont_pct:.1f}%" if cont_pct == 100.0 else f"[yellow]{cont_pct:.1f}%[/yellow]"
                    csv_size = f"{rep.get('csv_size_kb', 0):.1f} KB"
                    pq_size = f"{rep.get('parquet_size_kb', 0):.1f} KB" if rep.get("parquet_path") else "N/A"
                    integrity = "[green]PASSED[/green]"

                    table.add_row(
                        sym,
                        tf,
                        f"{count:,}",
                        str(rep.get("start_time")),
                        str(rep.get("end_time")),
                        cont_str,
                        csv_size,
                        pq_size,
                        integrity,
                    )
                else:
                    table.add_row(
                        sym,
                        tf,
                        "[red]0[/red]",
                        "ERROR",
                        "ERROR",
                        "0.0%",
                        "N/A",
                        "N/A",
                        f"[red]FAILED ({rep.get('error', 'Unknown')})[/red]",
                    )

        console.print(table)
        console.print(
            f"\n[bold green]Harvest Complete![/bold green] "
            f"Successfully archived [bold cyan]{total_candles:,}[/bold cyan] candles into "
            f"[underline]{harvest_data.get('output_dir')}[/underline]\n"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Binance Historical Candlestick Harvester & Caching Pipeline"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help=f"Trading pairs to harvest (default: {' '.join(DEFAULT_SYMBOLS)})",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=DEFAULT_TIMEFRAMES,
        help=f"Candlestick timeframes (default: {' '.join(DEFAULT_TIMEFRAMES)})",
    )
    parser.add_argument(
        "--candles",
        type=int,
        default=DEFAULT_TARGET_CANDLES,
        help=f"Target candles per pair/timeframe (default: {DEFAULT_TARGET_CANDLES})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--keep-incomplete",
        action="store_true",
        help="Retain the currently forming candle if present (default: drops incomplete)",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.1,
        help="Polite inter-request delay in seconds (default: 0.1s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    collector = DataCollector(
        symbols=args.symbols,
        timeframes=args.timeframes,
        output_dir=Path(args.output_dir),
        request_delay=args.request_delay,
    )
    harvest_summary = collector.harvest_all(
        target_candles=args.candles,
        drop_incomplete=not args.keep_incomplete,
    )
    collector.print_summary_table(harvest_summary)


if __name__ == "__main__":
    main()
