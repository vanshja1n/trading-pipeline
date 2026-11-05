"""Tests for Phase 3: Data validation, cleaning, and SQLite storage."""

import tempfile
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import pytest

from autotrade.data.validation import validate_ohlcv
from autotrade.data.cleaning import clean_ohlcv
from autotrade.data.storage import Database


def test_validate_ohlcv_valid_series():
    dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="Asia/Kolkata")
    df = pd.DataFrame(
        {
            "open": [100.0, 102.0, 101.0, 105.0, 107.0],
            "high": [105.0, 106.0, 104.0, 108.0, 110.0],
            "low": [98.0, 100.0, 99.0, 103.0, 105.0],
            "close": [103.0, 101.0, 104.0, 107.0, 108.0],
            "volume": [1000.0, 1500.0, 1200.0, 2000.0, 1800.0],
        },
        index=dates,
    )
    report = validate_ohlcv(df, symbol="RELIANCE")
    assert report.is_valid is True
    assert len(report.errors) == 0
    assert report.total_bars == 5


def test_validate_ohlcv_detects_flaws():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "open": [100.0, -10.0, 101.0],  # negative price
            "high": [90.0, 106.0, 104.0],   # high < low in bar 0
            "low": [95.0, 100.0, 99.0],
            "close": [92.0, 101.0, 104.0],
            "volume": [1000.0, -50.0, 1200.0],  # negative volume
        },
        index=dates,
    )
    report = validate_ohlcv(df, symbol="TEST")
    assert report.is_valid is False
    assert len(report.errors) > 0


def test_clean_ohlcv_repairs_and_deduplicates():
    # Duplicate index and inconsistent High
    ts = pd.Timestamp("2024-01-01 10:00:00", tz="Asia/Kolkata")
    dates = [ts, ts, pd.Timestamp("2024-01-02 10:00:00", tz="Asia/Kolkata")]
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 105.0],
            "high": [90.0, 95.0, 102.0],  # High < Open
            "low": [95.0, 92.0, 100.0],
            "close": [98.0, 97.0, 103.0],
            "volume": [100.0, 200.0, 300.0],
        },
        index=dates,
    )
    cleaned = clean_ohlcv(df, repair_ohlc=True)
    assert len(cleaned) == 2  # Deduplicated
    assert (cleaned["high"] >= cleaned["open"]).all()
    assert (cleaned["high"] >= cleaned["close"]).all()


def test_database_crud():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = Database(db_path)

        dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="Asia/Kolkata")
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [99.0, 100.0, 101.0],
                "close": [103.0, 104.0, 105.0],
                "volume": [1000.0, 1200.0, 1500.0],
            },
            index=dates,
        )

        # 1. Save & Load Market Data
        count = db.save_market_data("RELIANCE", df)
        assert count == 3

        loaded = db.load_market_data("RELIANCE")
        assert len(loaded) == 3
        assert db.get_available_symbols() == ["RELIANCE"]

        # 2. Log Signal
        db.log_signal("2024-01-01T15:30:00", "RELIANCE", "SMA_Cross", 1.0, {"fast": 20, "slow": 50})

        # 3. Log Order
        db.log_order(
            order_id="ORD-001",
            timestamp="2024-01-02T09:15:00",
            symbol="RELIANCE",
            side="BUY",
            qty=10,
            price=2500.0,
            costs=35.5,
        )
        orders = db.load_orders("RELIANCE")
        assert len(orders) == 1
        assert orders.iloc[0]["order_id"] == "ORD-001"
