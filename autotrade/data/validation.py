"""Data validation module for equity price series.

Provides robust checks for:
- Missing values and NaN rates
- Timestamp monotonicity and duplicate bars
- OHLC consistency (High >= Low, High >= Open/Close, Low <= Open/Close)
- Non-positive prices and negative volumes
- Data quality scoring and structured validation reports
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass
class ValidationReport:
    """Structured report detailing the quality of an OHLCV dataset."""

    symbol: str
    is_valid: bool
    total_bars: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"Data Validation [{self.symbol}]: {status}",
            f"Total Bars: {self.total_bars}",
        ]
        if self.errors:
            lines.append("Errors:")
            for err in self.errors:
                lines.append(f"  - {err}")
        if self.warnings:
            lines.append("Warnings:")
            for warn in self.warnings:
                lines.append(f"  - {warn}")
        return "\n".join(lines)


def validate_ohlcv(df: pd.DataFrame, symbol: str = "UNKNOWN") -> ValidationReport:
    """Run comprehensive validation on an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with canonical lowercase columns.
    symbol : str
        Ticker symbol for reporting.

    Returns
    -------
    ValidationReport
        Detailed report of data validity, errors, and warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}

    total_bars = len(df)
    metrics["total_bars"] = total_bars

    if total_bars == 0:
        return ValidationReport(
            symbol=symbol,
            is_valid=False,
            total_bars=0,
            errors=["DataFrame is empty"],
        )

    required_cols = {"open", "high", "low", "close", "volume"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(missing_cols)}")
        return ValidationReport(
            symbol=symbol,
            is_valid=False,
            total_bars=total_bars,
            errors=errors,
            metrics=metrics,
        )

    # 1. Index and timestamp ordering
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("Index is not a DatetimeIndex")
    else:
        if not df.index.is_monotonic_increasing:
            errors.append("Timestamps are not strictly monotonically increasing")
        duplicate_ts = df.index.duplicated().sum()
        metrics["duplicate_timestamps"] = int(duplicate_ts)
        if duplicate_ts > 0:
            errors.append(f"Found {duplicate_ts} duplicate timestamps")

    # 2. Null / NaN counts
    null_counts = df[list(required_cols)].isnull().sum()
    total_nulls = int(null_counts.sum())
    metrics["null_counts"] = null_counts.to_dict()
    if total_nulls > 0:
        errors.append(f"Found {total_nulls} missing values across OHLCV columns")

    # Drop nulls temporarily for price sanity checks if any exist
    clean = df.dropna(subset=["open", "high", "low", "close", "volume"])

    if not clean.empty:
        # 3. Non-positive prices
        non_positive = (clean[["open", "high", "low", "close"]] <= 0).sum().sum()
        metrics["non_positive_prices"] = int(non_positive)
        if non_positive > 0:
            errors.append(f"Found {non_positive} non-positive price values (<= 0)")

        # 4. Negative volume
        neg_volume = int((clean["volume"] < 0).sum())
        metrics["negative_volume"] = neg_volume
        if neg_volume > 0:
            errors.append(f"Found {neg_volume} negative volume records")

        # 5. OHLC logical relationships
        # High must be >= Low, Open, Close
        high_low_violations = int((clean["high"] < clean["low"]).sum())
        high_open_violations = int((clean["high"] < clean["open"]).sum())
        high_close_violations = int((clean["high"] < clean["close"]).sum())

        low_open_violations = int((clean["low"] > clean["open"]).sum())
        low_close_violations = int((clean["low"] > clean["close"]).sum())

        ohlc_violations = (
            high_low_violations
            + high_open_violations
            + high_close_violations
            + low_open_violations
            + low_close_violations
        )
        metrics["ohlc_violations"] = ohlc_violations

        if high_low_violations > 0:
            errors.append(f"Found {high_low_violations} bars where High < Low")
        if high_open_violations > 0 or high_close_violations > 0:
            errors.append(
                f"Found {high_open_violations + high_close_violations} bars where High < Open or Close"
            )
        if low_open_violations > 0 or low_close_violations > 0:
            errors.append(
                f"Found {low_open_violations + low_close_violations} bars where Low > Open or Close"
            )

        # 6. Warnings: Zero volume days (normal for illiquid stocks, but worth warning)
        zero_vol = int((clean["volume"] == 0).sum())
        metrics["zero_volume_bars"] = zero_vol
        if zero_vol > 0:
            pct = (zero_vol / total_bars) * 100
            warnings.append(f"{zero_vol} bars ({pct:.1f}%) have zero volume")

    is_valid = len(errors) == 0

    return ValidationReport(
        symbol=symbol,
        is_valid=is_valid,
        total_bars=total_bars,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )
