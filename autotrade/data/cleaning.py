"""Data cleaning and sanitization pipeline for financial time series."""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


def clean_ohlcv(
    df: pd.DataFrame,
    fill_missing: bool = True,
    repair_ohlc: bool = True,
    drop_zeros: bool = True,
) -> pd.DataFrame:
    """Clean and sanitize an OHLCV DataFrame.

    Operations:
    1. Sort index chronologically and remove duplicate timestamps (keep last).
    2. Drop bars where price is <= 0.
    3. Optionally repair OHLC inconsistencies (e.g., clamp High to max(O,H,L,C)).
    4. Fill missing values (forward fill price, zero-fill volume).
    5. Cast columns to float64.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV dataframe.
    fill_missing : bool
        If True, forward-fill missing price data.
    repair_ohlc : bool
        If True, ensure High is >= all prices and Low is <= all prices.
    drop_zeros : bool
        If True, drop rows where Open, High, Low, or Close is <= 0.

    Returns
    -------
    pd.DataFrame
        Sanitized OHLCV dataframe.
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    # 1. Deduplicate index & sort
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
    if out.index.duplicated().any():
        dup_count = out.index.duplicated().sum()
        log.warning("Dropping %d duplicate timestamps", dup_count)
        out = out[~out.index.duplicated(keep="last")]

    # 2. Drop negative or zero prices
    if drop_zeros:
        price_cols = [c for c in ["open", "high", "low", "close"] if c in out.columns]
        valid_prices = (out[price_cols] > 0).all(axis=1)
        if not valid_prices.all():
            dropped = int((~valid_prices).sum())
            log.warning("Dropping %d rows with non-positive prices", dropped)
            out = out[valid_prices]

    # 3. Fill missing data
    if fill_missing:
        # Forward fill prices, then backward fill if starting with NaN
        price_cols = [c for c in ["open", "high", "low", "close"] if c in out.columns]
        out[price_cols] = out[price_cols].ffill().bfill()
        if "volume" in out.columns:
            out["volume"] = out["volume"].fillna(0.0)

    # 4. Repair OHLC bounds if requested
    if repair_ohlc and all(c in out.columns for c in ["open", "high", "low", "close"]):
        # True high is at least max(open, high, close)
        calc_high = out[["open", "high", "close"]].max(axis=1)
        # True low is at most min(open, low, close)
        calc_low = out[["open", "low", "close"]].min(axis=1)

        out["high"] = np.maximum(out["high"], calc_high)
        out["low"] = np.minimum(out["low"], calc_low)

    # Keep attributes (e.g. symbol)
    out.attrs = df.attrs.copy()

    return out.astype("float64")
