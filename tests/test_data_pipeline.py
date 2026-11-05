"""Tests for data ingestion: CSV loading, alias resolution, timezone normalization."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from autotrade.data.ingestion import load_csv, load_csv_directory
from autotrade.market.constants import IST


def test_load_csv_canonical_format():
    # Synthetic CSV with canonical columns
    content = """Date,Open,High,Low,Close,Volume
2024-01-01,100.0,105.0,99.0,103.0,10000
2024-01-02,103.0,107.0,102.0,106.0,15000
2024-01-03,106.0,106.5,101.0,102.0,12000
"""
    with tempfile.NamedTemporaryFile("w", suffix="_RELIANCE.csv", delete=False) as f:
        f.write(content)
        path = f.name

    try:
        df = load_csv(path)
        assert len(df) == 3
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.tz == IST
        assert df.attrs["symbol"] == Path(path).stem.upper()
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_csv_alias_handling():
    # Synthetic CSV with aliases (e.g., TotalTrdQty, Adj Close)
    content = """timestamp,Open,High,Low,Adj Close,TotalTrdQty
2024-01-01,2500.0,2550.0,2480.0,2520.0,500000
2024-01-02,2525.0,2560.0,2510.0,2555.0,600000
"""
    with tempfile.NamedTemporaryFile("w", suffix="_TCS.csv", delete=False) as f:
        f.write(content)
        path = f.name

    try:
        df = load_csv(path, symbol="TCS")
        assert len(df) == 2
        assert "close" in df.columns
        assert "volume" in df.columns
        assert df.iloc[0]["close"] == 2520.0
        assert df.iloc[0]["volume"] == 500000.0
        assert df.attrs["symbol"] == "TCS"
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_csv_missing_columns_raises():
    content = """Date,Open,High,Low
2024-01-01,100.0,105.0,99.0
"""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(content)
        path = f.name

    try:
        with pytest.raises(ValueError, match="missing required columns"):
            load_csv(path)
    finally:
        Path(path).unlink(missing_ok=True)


def test_fetch_market_snapshot_mocked():
    from unittest.mock import MagicMock, patch
    from autotrade.data.ingestion import fetch_market_snapshot

    mock_ticker = MagicMock()
    mock_ticker.fast_info.last_price = 1350.0
    mock_ticker.fast_info.previous_close = 1300.0
    mock_ticker.fast_info.day_high = 1360.0
    mock_ticker.fast_info.day_low = 1340.0
    mock_ticker.fast_info.last_volume = 2000000

    with patch("yfinance.Ticker", return_value=mock_ticker):
        rows = fetch_market_snapshot(["RELIANCE", "TCS"])
        assert len(rows) == 2
        for r in rows:
            assert "Symbol" in r
            assert "LTP (₹)" in r
            assert "₹1,350.00" in r["LTP (₹)"]
            assert "Change" in r
            assert "+3.85%" in r["Change"]
            assert "Day High (₹)" in r
            assert "₹1,360.00" in r["Day High (₹)"]
            assert "Day Low (₹)" in r
            assert "₹1,340.00" in r["Day Low (₹)"]
            assert "Volume" in r
            assert "2,000,000" in r["Volume"]

