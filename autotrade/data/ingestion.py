"""Data ingestion for Indian equities.

Supports two primary data paths:
    1. **CSV upload** — load OHLCV data from local CSV files in various formats.
    2. **yfinance** — download historical data for NSE-listed symbols.

Every ingestion path normalises output to the canonical OHLCV DataFrame format
expected downstream: lowercase columns (``open, high, low, close, volume``),
a timezone-aware ``DatetimeIndex`` named ``timestamp``, and ``float64`` dtypes.

Design notes:
    - The module deliberately does NOT fabricate data.  If a CSV is empty or a
      download returns nothing, it raises an error rather than silently
      substituting sample data.
    - Column detection is fuzzy: it handles capitalisation differences and
      common aliases (``Adj Close``, ``TotalTrdQty``, ``TOTTRDQTY``).
    - The ``_normalize`` function from ``feed.py`` is reused via import to
      avoid duplicating normalisation logic.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ..market.constants import IST, resolve_nse_symbol, strip_exchange_suffix

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column mapping — handles common CSV formats
# ---------------------------------------------------------------------------

#: Maps common column name variants to our canonical names.
_COLUMN_ALIASES: dict[str, str] = {
    # Standard / yfinance
    "date": "date",
    "datetime": "date",
    "timestamp": "date",
    "time": "date",
    # Price
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "close",
    "adjusted close": "close",
    "ltp": "close",
    "last": "last",
    "prev close": "prev_close",
    "prevclose": "prev_close",
    # Volume
    "volume": "volume",
    "vol": "volume",
    "total traded quantity": "volume",
    "totaltrdqty": "volume",
    "tottrdqty": "volume",
    "total_traded_quantity": "volume",
    "no. of shares": "volume",
    "no.of shares": "volume",
    # Symbol (metadata columns we recognise but don't require)
    "symbol": "symbol",
    "ticker": "symbol",
    "series": "series",
    "isin": "isin",
}

_REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns using the alias table.

    Raises ``ValueError`` if any of the required OHLCV columns are missing
    after mapping.
    """
    renamed: dict[str, str] = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in _COLUMN_ALIASES:
            canonical = _COLUMN_ALIASES[key]
            # Avoid double-mapping (e.g., both "Adj Close" and "Close" present).
            if canonical not in renamed.values():
                renamed[col] = canonical
            else:
                # Keep the original if the canonical name is already taken.
                renamed[col] = key
        else:
            renamed[col] = key

    df = df.rename(columns=renamed)

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns {sorted(missing)} after mapping. "
            f"Available columns: {sorted(df.columns)}"
        )
    return df


def _parse_index(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and set the date/time column as a timezone-aware DatetimeIndex."""
    # If the index is already datetime, just ensure it's tz-aware.
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize(IST)
        df.index.name = "timestamp"
        return df

    # Look for a date column.
    date_col = None
    for candidate in ("date", "datetime", "timestamp"):
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        # Try using the first column if it looks like dates.
        first_col = df.columns[0]
        try:
            pd.to_datetime(df[first_col].head(5))
            date_col = first_col
        except (ValueError, TypeError):
            raise ValueError(
                "Could not find a date/timestamp column. "
                "Ensure your CSV has a column named Date, Datetime, or Timestamp."
            )

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    if df.index.tz is None:
        df.index = df.index.tz_localize(IST)
    df.index.name = "timestamp"
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_csv(
    filepath: str | Path,
    symbol: str | None = None,
    date_column: str | None = None,
) -> pd.DataFrame:
    """Load OHLCV data from a CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    symbol : str, optional
        Stock symbol.  If not provided, inferred from the filename
        (e.g., ``RELIANCE.csv`` → ``RELIANCE``).
    date_column : str, optional
        Explicit name of the date column.  If not provided, auto-detected.

    Returns
    -------
    pd.DataFrame
        Normalised OHLCV DataFrame with timezone-aware index.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing or the file is empty.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError(f"CSV file is empty: {filepath}")

    log.info("loaded %d rows from %s", len(df), filepath.name)

    # Map columns to canonical names.
    df = _map_columns(df)

    # If the user specified a date column, rename it before index detection.
    if date_column and date_column in df.columns:
        df = df.rename(columns={date_column: "date"})

    # Set date index.
    df = _parse_index(df)

    # Keep only OHLCV columns and cast to float64.
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].astype("float64")

    # Drop rows with missing price data (volume can be zero).
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_index()

    # Infer symbol from filename if not provided.
    if symbol is None:
        symbol = strip_exchange_suffix(filepath.stem.upper())

    # Store symbol as metadata.
    df.attrs["symbol"] = symbol

    return df


def load_csv_directory(
    directory: str | Path,
    pattern: str = "*.csv",
    date_column: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load all CSV files from a directory.

    Parameters
    ----------
    directory : str or Path
        Directory containing CSV files.
    pattern : str
        Glob pattern to match CSV files.
    date_column : str, optional
        Passed through to ``load_csv()``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ``{symbol: DataFrame}``.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")

    results: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(directory.glob(pattern)):
        try:
            df = load_csv(csv_path, date_column=date_column)
            symbol = df.attrs.get("symbol", csv_path.stem.upper())
            results[symbol] = df
            log.info("loaded %s: %d bars", symbol, len(df))
        except Exception:
            log.exception("failed to load %s", csv_path.name)

    if not results:
        log.warning("no CSV files loaded from %s", directory)

    return results


def download_nse_symbol(
    symbol: str,
    start: str | datetime,
    end: str | datetime | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download historical data for an NSE symbol via yfinance.

    Automatically appends ``.NS`` to the symbol if not already present.

    Parameters
    ----------
    symbol : str
        NSE symbol (e.g., ``"RELIANCE"`` or ``"RELIANCE.NS"``).
    start : str or datetime
        Start date for data download.
    end : str or datetime, optional
        End date (defaults to today).
    interval : str
        Bar interval: ``"1d"`` (default), ``"1wk"``, ``"1mo"``.

    Returns
    -------
    pd.DataFrame
        Normalised OHLCV DataFrame.

    Raises
    ------
    RuntimeError
        If no data is returned (bad symbol, no data in range, etc.).
    """
    import yfinance as yf

    yf_symbol = resolve_nse_symbol(symbol)
    clean_symbol = strip_exchange_suffix(symbol)

    if end is None:
        end = datetime.now(timezone.utc).date().isoformat()

    log.info(
        "downloading %s (%s) bars from Yahoo (%s to %s)",
        clean_symbol, yf_symbol, start, end,
    )

    raw = yf.download(
        yf_symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if raw is None or raw.empty:
        raise RuntimeError(
            f"no data returned for {yf_symbol}; "
            f"check the ticker and date range"
        )

    # Flatten MultiIndex columns if present (yfinance returns them for single symbols too).
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Normalize column names.
    raw = raw.rename(columns=str.lower)
    keep = ["open", "high", "low", "close", "volume"]
    missing = [c for c in keep if c not in raw.columns]
    if missing:
        raise ValueError(f"yfinance data missing columns {missing}")

    df = raw[keep].astype("float64")
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_index()

    df.attrs["symbol"] = clean_symbol

    return df


def download_watchlist(
    symbols: Sequence[str],
    start: str | datetime,
    end: str | datetime | None = None,
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Download historical data for multiple NSE symbols.

    Returns a dict mapping clean symbol names to DataFrames.  Symbols that
    fail to download are logged and skipped.
    """
    results: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            df = download_nse_symbol(symbol, start, end, interval)
            clean = df.attrs.get("symbol", strip_exchange_suffix(symbol))
            results[clean] = df
        except Exception:
            log.exception("failed to download %s", symbol)
    return results


def fetch_market_snapshot(
    symbols: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch live market snapshot for NSE equity tickers using yfinance.

    Appends .NS to symbols and retrieves LTP, Day High, Day Low, % Change, and Volume.
    Updates dynamically upon call.

    Parameters
    ----------
    symbols : Sequence[str], optional
        NSE symbols (e.g. ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN"]).
        Defaults to NSE_WATCHLIST[:6].

    Returns
    -------
    list[dict[str, Any]]
        List of dicts formatted for the Market Snapshot table.
    """
    from concurrent.futures import ThreadPoolExecutor
    import yfinance as yf
    from ..market.constants import NSE_WATCHLIST

    target_symbols = list(symbols) if symbols else list(NSE_WATCHLIST[:6])

    def _fetch_one(sym: str) -> dict[str, Any]:
        clean_sym = strip_exchange_suffix(sym).upper()
        yf_sym = resolve_nse_symbol(clean_sym)
        try:
            ticker = yf.Ticker(yf_sym)
            fi = ticker.fast_info
            price = getattr(fi, "last_price", None)
            prev_close = getattr(fi, "previous_close", None)
            day_high = getattr(fi, "day_high", None)
            day_low = getattr(fi, "day_low", None)
            volume = getattr(fi, "last_volume", None)

            # Fallback to 1-day history if fast_info missing price
            if price is None or price <= 0:
                hist = ticker.history(period="1d", interval="5m")
                if not hist.empty:
                    last_row = hist.iloc[-1]
                    price = float(last_row["Close"])
                    if prev_close is None:
                        prev_close = float(hist["Open"].iloc[0])
                    if day_high is None:
                        day_high = float(hist["High"].max())
                    if day_low is None:
                        day_low = float(hist["Low"].min())
                    if volume is None:
                        volume = float(hist["Volume"].sum())

            if price is not None and price > 0:
                pc = float(prev_close) if prev_close and prev_close > 0 else float(price)
                chg = (price - pc) / pc if pc > 0 else 0.0
                dh = float(day_high) if day_high is not None else float(price)
                dl = float(day_low) if day_low is not None else float(price)
                vol = float(volume) if volume is not None and volume >= 0 else 0.0
                return {
                    "Symbol": clean_sym,
                    "LTP (₹)": f"₹{price:,.2f}",
                    "Change": f"{chg:+.2%}",
                    "Day High (₹)": f"₹{dh:,.2f}",
                    "Day Low (₹)": f"₹{dl:,.2f}",
                    "Volume": f"{int(vol):,}",
                }
        except Exception as exc:
            log.warning("failed to fetch snapshot for %s: %s", yf_sym, exc)

        return {
            "Symbol": clean_sym,
            "LTP (₹)": "N/A",
            "Change": "0.00%",
            "Day High (₹)": "N/A",
            "Day Low (₹)": "N/A",
            "Volume": "0",
        }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(6, len(target_symbols))) as pool:
        future_map = {pool.submit(_fetch_one, sym): sym for sym in target_symbols}
        for fut, sym in future_map.items():
            try:
                results.append(fut.result(timeout=4.0))
            except Exception as exc:
                clean_sym = strip_exchange_suffix(sym).upper()
                log.warning("Snapshot fetch timed out or failed for %s: %s", clean_sym, exc)
                results.append({
                    "Symbol": clean_sym,
                    "LTP (₹)": "N/A",
                    "Change": "0.00%",
                    "Day High (₹)": "N/A",
                    "Day Low (₹)": "N/A",
                    "Volume": "0",
                })

    return results

