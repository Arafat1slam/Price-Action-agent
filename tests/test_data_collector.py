import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from data_collector import DataCollector, INTERVAL_MS_MAP


@pytest.fixture
def sample_raw_candles():
    """Generates 10 valid raw Binance kline records for 1h interval."""
    base_time = 1700000000000  # ms
    step = 3600 * 1000
    candles = []
    for i in range(10):
        t = base_time + i * step
        c_time = t + step - 1
        o = 50000.0 + i * 10
        h = o + 50.0
        l = o - 40.0
        c = o + 20.0
        v = 12.5 + i
        candles.append([
            t, str(o), str(h), str(l), str(c), str(v),
            c_time, str(v * c), 150 + i, str(v * 0.6), str(v * 0.6 * c), "0"
        ])
    return candles


def test_clean_and_validate_basic(sample_raw_candles, tmp_path):
    collector = DataCollector(output_dir=tmp_path)
    df, report = collector.clean_and_validate(sample_raw_candles, interval="1h", drop_incomplete=False)

    assert report["valid"] is True
    assert report["candle_count"] == 10
    assert report["dropped_nans"] == 0
    assert report["duplicates_removed"] == 0
    assert report["invalid_price_rows"] == 0
    assert report["continuity_pct"] == 100.0
    assert "timestamp" in df.columns
    assert "open" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns
    assert "is_closed" in df.columns
    assert df["is_closed"].all()


def test_clean_and_validate_dedup_and_sort(sample_raw_candles, tmp_path):
    collector = DataCollector(output_dir=tmp_path)
    # Duplicate first candle and shuffle order
    shuffled = [sample_raw_candles[3], sample_raw_candles[0], sample_raw_candles[0], sample_raw_candles[1]]
    df, report = collector.clean_and_validate(shuffled, interval="1h", drop_incomplete=False)

    assert report["duplicates_removed"] == 1
    assert len(df) == 3
    # Verify strict ascending sort
    ts_list = pd.to_datetime(df["timestamp"]).tolist()
    assert ts_list == sorted(ts_list)


def test_clean_and_validate_price_sanity(sample_raw_candles, tmp_path):
    collector = DataCollector(output_dir=tmp_path)
    # Inject an invalid candle where High < Low
    corrupt_candle = list(sample_raw_candles[0])
    corrupt_candle[0] = corrupt_candle[0] + 999999999
    corrupt_candle[1] = "50000.0"
    corrupt_candle[2] = "40000.0"  # High < Low
    corrupt_candle[3] = "45000.0"  # Low
    corrupt_candle[4] = "42000.0"

    data = sample_raw_candles + [corrupt_candle]
    df, report = collector.clean_and_validate(data, interval="1h", drop_incomplete=False)

    assert report["invalid_price_rows"] == 1
    assert len(df) == 10


def test_gap_detection(sample_raw_candles, tmp_path):
    collector = DataCollector(output_dir=tmp_path)
    # Create a gap by removing candle at index 5 and 6
    gapped = sample_raw_candles[:5] + sample_raw_candles[7:]
    df, report = collector.clean_and_validate(gapped, interval="1h", drop_incomplete=False)

    assert report["gap_count"] == 1
    assert report["missing_candles_est"] == 2
    assert report["continuity_pct"] < 100.0


def test_save_dataset_csv_and_parquet(sample_raw_candles, tmp_path):
    collector = DataCollector(output_dir=tmp_path)
    df, _ = collector.clean_and_validate(sample_raw_candles, interval="1h", drop_incomplete=False)

    csv_path, parquet_path = collector.save_dataset(df, "BTCUSDT", "1h")
    assert csv_path.exists()
    assert csv_path.stat().st_size > 0

    # Read back CSV
    df_read = pd.read_csv(csv_path)
    assert len(df_read) == 10
    assert list(df_read.columns) == list(df.columns)

    if parquet_path:
        assert parquet_path.exists()
        assert parquet_path.stat().st_size > 0
        df_pq = pd.read_parquet(parquet_path)
        assert len(df_pq) == 10


def test_endpoint_rotation(tmp_path):
    collector = DataCollector(
        endpoints=["https://api.binance.com", "https://api1.binance.com", "https://api2.binance.com"],
        output_dir=tmp_path,
    )
    assert collector.current_endpoint == "https://api.binance.com"
    collector.rotate_endpoint()
    assert collector.current_endpoint == "https://api1.binance.com"
    collector.rotate_endpoint()
    assert collector.current_endpoint == "https://api2.binance.com"
    collector.rotate_endpoint()
    assert collector.current_endpoint == "https://api.binance.com"
