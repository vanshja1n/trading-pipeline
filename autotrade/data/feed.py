"""Market data access.

Two sources, one shape. Everything downstream expects a DataFrame indexed by
timestamp with lowercase columns: open, high, low, close, volume.

- YahooFeed:  free, no API key, good for daily backtests.
- AlpacaFeed: needs keys, gives intraday bars and matches what you'll actually
              trade against in paper/live.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"

_ALPACA_TIMEFRAMES = {
    "1Min": (1, "Min"),
    "5Min": (5, "Min"),
    "15Min": (15, "Min"),
    "1Hour": (1, "Hour"),
    "1Day": (1, "Day"),
}

_YAHOO_INTERVALS = {
    "1Min": "1m",
    "5Min": "5m",
    "15Min": "15m",
    "1Hour": "1h",
    "1Day": "1d",
}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten/lowercase columns and drop incomplete rows."""
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns (field, ticker); keep the field level.
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    keep = ["open", "high", "low", "close", "volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"data source returned no {missing} column")
    df = df[keep].astype("float64")
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    return df.dropna(subset=["open", "high", "low", "close"]).sort_index()


class YahooFeed:
    """Historical bars from Yahoo Finance. No API key required.

    Parameters
    ----------
    use_cache : bool
        If True, cache downloaded data as CSV files for faster reruns.
    market : str
        ``"US"`` (default) for US equities, ``"IN"`` for NSE-listed Indian
        equities.  When ``"IN"``, the ``.NS`` suffix is automatically appended
        to symbols before querying yfinance.
    """

    def __init__(self, use_cache: bool = True, market: str = "US"):
        self.use_cache = use_cache
        self.market = market.upper()

    def _resolve_symbol(self, symbol: str) -> str:
        """Apply market-specific symbol resolution."""
        if self.market == "IN":
            from ..market.constants import resolve_nse_symbol
            return resolve_nse_symbol(symbol)
        return symbol

    def history(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime | None = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        import yfinance as yf

        interval = _YAHOO_INTERVALS.get(timeframe)
        if interval is None:
            raise ValueError(f"unsupported timeframe {timeframe!r}")

        yf_symbol = self._resolve_symbol(symbol)
        end = end or datetime.now(timezone.utc).date().isoformat()
        cache_path = CACHE_DIR / f"{symbol}_{interval}_{start}_{end}.csv"
        if self.use_cache and cache_path.exists():
            log.debug("cache hit %s", cache_path.name)
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            cached.index = pd.to_datetime(cached.index, utc=True)
            return cached

        log.info("downloading %s (%s) %s bars from Yahoo (%s to %s)",
                 symbol, yf_symbol, interval, start, end)
        raw = yf.download(
            yf_symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,  # splits/dividends folded in -- essential for backtests
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            raise RuntimeError(f"no data returned for {yf_symbol}; check the ticker and dates")

        df = _normalize(raw)
        if self.use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path)
        return df


class AlpacaFeed:
    """Bars from Alpaca's market data API. Works with paper-account keys."""

    def __init__(self, api_key: str, api_secret: str):
        from alpaca.data.historical import StockHistoricalDataClient

        self._client = StockHistoricalDataClient(api_key, api_secret)

    def history(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime | None = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        if timeframe not in _ALPACA_TIMEFRAMES:
            raise ValueError(f"unsupported timeframe {timeframe!r}")
        amount, unit = _ALPACA_TIMEFRAMES[timeframe]

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(amount, TimeFrameUnit(unit)),
            start=pd.Timestamp(start).to_pydatetime(),
            end=pd.Timestamp(end).to_pydatetime() if end else None,
        )
        bars = self._client.get_stock_bars(request)
        df = bars.df
        if df.empty:
            raise RuntimeError(f"Alpaca returned no bars for {symbol}")
        if isinstance(df.index, pd.MultiIndex):  # (symbol, timestamp)
            df = df.xs(symbol, level="symbol")
        return _normalize(df)

    def latest_bars(self, symbol: str, lookback: int, timeframe: str = "1Day") -> pd.DataFrame:
        """Recent bars for a live decision loop, sized to cover indicator warmup."""
        span = {
            "1Min": timedelta(minutes=lookback * 3),
            "5Min": timedelta(minutes=lookback * 12),
            "15Min": timedelta(minutes=lookback * 35),
            "1Hour": timedelta(hours=lookback * 3),
            "1Day": timedelta(days=lookback * 2),
        }[timeframe]
        # Alpaca's free tier can't query the most recent 15 minutes of SIP data.
        end = datetime.now(timezone.utc) - timedelta(minutes=16)
        df = self.history(symbol, start=end - span, end=end, timeframe=timeframe)
        return df.tail(lookback)


def get_feed(api_key: str | None = None, api_secret: str | None = None):
    """Alpaca when credentials exist, Yahoo otherwise."""
    if api_key and api_secret:
        return AlpacaFeed(api_key, api_secret)
    return YahooFeed()
