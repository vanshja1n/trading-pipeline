"""Tests for Stock Correlation Analysis Engine."""

import numpy as np
import pandas as pd
import pytest

from autotrade.portfolio.correlation import (
    CorrelationPair,
    CorrelationResult,
    compute_correlation_analysis,
    compute_correlation_matrix,
)
from autotrade.market.constants import IST


def _create_synthetic_ohlcv(symbol: str, n_bars: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="B", tz=IST)
    rets = rng.normal(0.0005, 0.015, size=n_bars)
    prices = 500.0 * np.cumprod(1 + rets)
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": 50000.0,
        },
        index=dates,
    )


def test_correlation_insufficient_data():
    data_cache = {
        "RELIANCE": _create_synthetic_ohlcv("RELIANCE", 20, seed=1),
    }
    res = compute_correlation_analysis(data_cache, ["RELIANCE"], min_history=60)
    assert not res.data_sufficient
    assert len(res.highest_pairs) == 0
    assert "Need at least 2 symbols" in res.warning


def test_correlation_matrix_computation():
    data_cache = {
        "RELIANCE": _create_synthetic_ohlcv("RELIANCE", 100, seed=1),
        "TCS": _create_synthetic_ohlcv("TCS", 100, seed=2),
        "INFY": _create_synthetic_ohlcv("INFY", 100, seed=3),
    }
    symbols = ["RELIANCE", "TCS", "INFY"]
    res = compute_correlation_analysis(data_cache, symbols, min_history=60)
    
    assert res.data_sufficient
    mat = res.correlation_matrix
    assert mat.shape == (3, 3)
    # Diagonal should be 1.0
    for sym in symbols:
        assert pytest.approx(mat.loc[sym, sym], abs=1e-5) == 1.0
    
    # Check pairs
    assert len(res.highest_pairs) > 0
    assert len(res.lowest_pairs) > 0
    assert res.interpretation != ""
    assert res.data_points == 99


def test_correlation_alias():
    data_cache = {
        "RELIANCE": _create_synthetic_ohlcv("RELIANCE", 100, seed=1),
        "TCS": _create_synthetic_ohlcv("TCS", 100, seed=2),
    }
    res = compute_correlation_matrix(data_cache, ["RELIANCE", "TCS"], min_history=60)
    assert res.data_sufficient
