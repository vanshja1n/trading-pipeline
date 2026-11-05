"""Live market data integration layer for Indian equities (NSE).

Provides real-time, delayed, and latest-available quotes, market session tracking,
intraday bar series, caching, and paper-trading synchronization.

Design Principles:
- Single responsibility: live market ingestion, validation, and session states.
- Reuses existing constants, calendar (NSECalendar), and symbol helpers.
- Strictly NO mock or fabricated market prices: missing or unavailable data is
  explicitly flagged as such.
- Transparency: clearly indicates data freshness (e.g. Delayed vs Latest Close)
  in accordance with provider capabilities.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..market.calendar import NSECalendar
from ..market.constants import (
    CURRENCY_SYMBOL,
    IST,
    MARKET_CLOSE,
    MARKET_OPEN,
    NSE_WATCHLIST,
    PRE_OPEN_START,
    resolve_nse_symbol,
    strip_exchange_suffix,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class LiveQuote:
    """Represents a validated market quote for a single equity."""

    symbol: str
    price: float
    previous_close: float
    change: float
    change_pct: float
    volume: float
    status: str
    timestamp: datetime
    open: float | None = None
    high: float | None = None
    low: float | None = None
    vwap: float | None = None
    data_source: str = "yfinance"
    freshness: str = "LATEST_AVAILABLE"  # "REALTIME", "DELAYED", "LATEST_AVAILABLE", "UNAVAILABLE"
    is_valid: bool = True
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "Symbol": self.symbol,
            "Price": f"{CURRENCY_SYMBOL}{self.price:,.2f}" if self.is_valid else "N/A",
            "Change": f"{self.change:+,.2f}" if self.is_valid else "N/A",
            "Change %": f"{self.change_pct:+.2f}%" if self.is_valid else "N/A",
            "Volume": f"{int(self.volume):,}" if self.is_valid and self.volume > 0 else "N/A",
            "Status": self.status,
            "Raw_Price": self.price,
            "Raw_Change": self.change,
            "Raw_Change_Pct": self.change_pct,
            "Raw_Volume": self.volume,
            "Timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S") if self.timestamp else "",
            "Freshness": self.freshness,
        }


@dataclass
class MarketSessionStatus:
    """Current NSE market session status derived from NSECalendar."""

    status: str  # "OPEN", "CLOSED", "PRE-OPEN", "WEEKEND", "HOLIDAY"
    is_open: bool
    exchange: str = "NSE"
    server_time: datetime = field(default_factory=lambda: datetime.now(IST))
    details: str = ""
    next_session: str | None = None


@dataclass
class LiveMarketSnapshot:
    """Full snapshot of the universe quotes and market state."""

    session: MarketSessionStatus
    quotes: dict[str, LiveQuote]
    timestamp: datetime
    is_stale: bool = False
    error: str | None = None
    warning: str | None = None


# ---------------------------------------------------------------------------
# Market Session Evaluation
# ---------------------------------------------------------------------------

def evaluate_market_session(
    calendar: NSECalendar | None = None,
    dt: datetime | None = None,
) -> MarketSessionStatus:
    """Determine the accurate NSE session status at time *dt*.

    Accurately distinguishes:
    - Weekend
    - NSE exchange holiday
    - Pre-open session (09:00 - 09:15 IST)
    - Continuous trading regular session (09:15 - 15:30 IST)
    - Post-market session (after 15:30 IST)
    - Pre-market early morning (before 09:00 IST)
    """
    cal = calendar or NSECalendar()
    if dt is None:
        dt = datetime.now(IST)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    else:
        dt = dt.astimezone(IST)

    today = dt.date()
    t = dt.time()

    # 1. Weekend
    if cal.is_weekend(today):
        next_td = cal.next_trading_day(today)
        return MarketSessionStatus(
            status="WEEKEND",
            is_open=False,
            exchange="NSE",
            server_time=dt,
            details=f"Market Closed: Weekend. Next trading session on {next_td.strftime('%A, %d %b %Y')}.",
            next_session=str(next_td),
        )

    # 2. Exchange holiday
    if cal.is_holiday(today):
        next_td = cal.next_trading_day(today)
        return MarketSessionStatus(
            status="HOLIDAY",
            is_open=False,
            exchange="NSE",
            server_time=dt,
            details=f"Market Closed: Official NSE Holiday. Next trading session on {next_td.strftime('%A, %d %b %Y')}.",
            next_session=str(next_td),
        )

    # 3. Regular trading weekday checks
    if t < PRE_OPEN_START:
        return MarketSessionStatus(
            status="CLOSED",
            is_open=False,
            exchange="NSE",
            server_time=dt,
            details="Pre-market: Orders open at 09:00 IST; Continuous trading starts at 09:15 IST.",
            next_session="09:00 IST Today",
        )
    elif PRE_OPEN_START <= t < MARKET_OPEN:
        return MarketSessionStatus(
            status="PRE-OPEN",
            is_open=False,
            exchange="NSE",
            server_time=dt,
            details="Pre-Open Order Accumulation & Equilibrium Price Discovery (09:00 - 09:15 IST).",
            next_session="09:15 IST Today",
        )
    elif MARKET_OPEN <= t < MARKET_CLOSE:
        return MarketSessionStatus(
            status="OPEN",
            is_open=True,
            exchange="NSE",
            server_time=dt,
            details="Continuous Regular Trading Session (09:15 - 15:30 IST).",
            next_session=None,
        )
    else:
        next_td = cal.next_trading_day(today)
        return MarketSessionStatus(
            status="CLOSED",
            is_open=False,
            exchange="NSE",
            server_time=dt,
            details=f"Regular session ended at 15:30 IST. Next trading session on {next_td.strftime('%A, %d %b %Y')}.",
            next_session=str(next_td),
        )


# ---------------------------------------------------------------------------
# Live Market Feed
# ---------------------------------------------------------------------------

class LiveMarketFeed:
    """Manages live market quote retrieval, multi-threaded fetching, and TTL caching.

    Parameters
    ----------
    cache_ttl_seconds : float
        Duration in seconds before cached quotes are considered expired. Default 30.0s.
    calendar : NSECalendar, optional
        Calendar instance for session evaluation.
    max_workers : int
        Maximum concurrent threads for fetching stock quotes. Default 8.
    """

    def __init__(
        self,
        cache_ttl_seconds: float = 30.0,
        calendar: NSECalendar | None = None,
        max_workers: int = 8,
    ):
        self.cache_ttl_seconds = float(cache_ttl_seconds)
        self.calendar = calendar or NSECalendar()
        self.max_workers = max_workers

        self._cached_quotes: dict[str, LiveQuote] = {}
        self._last_update: datetime | None = None
        self._last_session: MarketSessionStatus | None = None
        self._last_error: str | None = None

    @property
    def last_update_time(self) -> datetime | None:
        """Timestamp of the most recent successful quote update."""
        return self._last_update

    def is_cache_valid(self, max_age_seconds: float | None = None) -> bool:
        """Check if cached quotes are still fresh."""
        if not self._cached_quotes or self._last_update is None:
            return False
        ttl = max_age_seconds if max_age_seconds is not None else self.cache_ttl_seconds
        elapsed = (datetime.now(IST) - self._last_update).total_seconds()
        return elapsed < ttl

    def get_market_session(self, dt: datetime | None = None) -> MarketSessionStatus:
        """Evaluate session status."""
        return evaluate_market_session(self.calendar, dt)


    def parse_quote_data(
        self,
        symbol: str,
        price: float | None,
        previous_close: float | None,
        volume: float | None = None,
        open_price: float | None = None,
        high_price: float | None = None,
        low_price: float | None = None,
        raw_timestamp: datetime | None = None,
        is_market_open: bool = False,
    ) -> LiveQuote:
        """Clean and validate raw quote fields into a canonical LiveQuote."""
        clean_sym = strip_exchange_suffix(symbol).upper()

        if price is None or price <= 0:
            return LiveQuote(
                symbol=clean_sym,
                price=0.0,
                previous_close=previous_close or 0.0,
                change=0.0,
                change_pct=0.0,
                volume=volume or 0.0,
                status="UNAVAILABLE",
                timestamp=raw_timestamp or datetime.now(IST),
                freshness="UNAVAILABLE",
                is_valid=False,
                error_message="Invalid or missing price data from provider",
            )

        prev_c = float(previous_close) if previous_close and previous_close > 0 else float(price)
        p = float(price)
        change = p - prev_c
        change_pct = (change / prev_c * 100.0) if prev_c > 0 else 0.0
        vol = float(volume) if volume and volume >= 0 else 0.0

        ts = raw_timestamp or datetime.now(IST)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=IST)
        else:
            ts = ts.astimezone(IST)

        freshness = "DELAYED" if is_market_open else "LATEST_AVAILABLE"
        trading_status = "TRADING" if is_market_open else "CLOSED"

        return LiveQuote(
            symbol=clean_sym,
            price=round(p, 2),
            previous_close=round(prev_c, 2),
            change=round(change, 2),
            change_pct=round(change_pct, 2),
            volume=vol,
            status=trading_status,
            timestamp=ts,
            open=round(float(open_price), 2) if open_price is not None else None,
            high=round(float(high_price), 2) if high_price is not None else None,
            low=round(float(low_price), 2) if low_price is not None else None,
            data_source="yfinance (NSE)",
            freshness=freshness,
            is_valid=True,
        )

    def fetch_single_quote(self, symbol: str) -> LiveQuote:
        """Fetch quote for a single symbol using yfinance fast_info."""
        import yfinance as yf

        clean_sym = strip_exchange_suffix(symbol).upper()
        yf_sym = resolve_nse_symbol(clean_sym)
        session = evaluate_market_session(self.calendar)

        try:
            ticker = yf.Ticker(yf_sym)
            fi = ticker.fast_info

            price = getattr(fi, "last_price", None)
            prev_close = getattr(fi, "previous_close", None)
            open_p = getattr(fi, "open", None)
            high_p = getattr(fi, "day_high", None)
            low_p = getattr(fi, "day_low", None)
            vol = getattr(fi, "last_volume", None)

            # If fast_info returns None for price, attempt a 1-day fallback
            if price is None or price <= 0:
                hist = ticker.history(period="1d", interval="5m")
                if not hist.empty:
                    last_row = hist.iloc[-1]
                    price = float(last_row["Close"])
                    if prev_close is None:
                        prev_close = float(hist["Open"].iloc[0])
                    if vol is None:
                        vol = float(hist["Volume"].sum())
                    if high_p is None:
                        high_p = float(hist["High"].max())
                    if low_p is None:
                        low_p = float(hist["Low"].min())
                    if open_p is None:
                        open_p = float(hist["Open"].iloc[0])

            quote = self.parse_quote_data(
                symbol=clean_sym,
                price=price,
                previous_close=prev_close,
                volume=vol,
                open_price=open_p,
                high_price=high_p,
                low_price=low_p,
                raw_timestamp=datetime.now(IST),
                is_market_open=session.is_open,
            )
            return quote

        except Exception as exc:
            log.warning("failed to fetch live quote for %s: %s", yf_sym, exc)
            return LiveQuote(
                symbol=clean_sym,
                price=0.0,
                previous_close=0.0,
                change=0.0,
                change_pct=0.0,
                volume=0.0,
                status="ERROR",
                timestamp=datetime.now(IST),
                freshness="UNAVAILABLE",
                is_valid=False,
                error_message=str(exc),
            )

    def fetch_quotes(
        self,
        symbols: Sequence[str] | None = None,
        force_refresh: bool = False,
    ) -> dict[str, LiveQuote]:
        """Fetch quotes for multiple symbols with parallel execution and caching.

        If network or provider error occurs, existing cached quotes are retained
        and marked stale rather than crashing or substituting fake values.
        """
        sym_list = list(symbols) if symbols else list(NSE_WATCHLIST)

        # Check cache validity
        if not force_refresh and self.is_cache_valid():
            return {s: self._cached_quotes[s] for s in sym_list if s in self._cached_quotes}

        quotes: dict[str, LiveQuote] = {}
        new_valid_count = 0
        try:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(sym_list))) as pool:
                future_map = {pool.submit(self.fetch_single_quote, s): s for s in sym_list}
                for future in as_completed(future_map):
                    sym = future_map[future]
                    try:
                        quote = future.result()
                        quotes[quote.symbol] = quote
                        if quote.is_valid:
                            new_valid_count += 1
                    except Exception as err:
                        log.error("thread error fetching %s: %s", sym, err)
                        # Retain previously cached quote if available
                        if sym in self._cached_quotes:
                            stale_q = self._cached_quotes[sym]
                            quotes[sym] = stale_q
                        else:
                            quotes[sym] = LiveQuote(
                                symbol=sym,
                                price=0.0,
                                previous_close=0.0,
                                change=0.0,
                                change_pct=0.0,
                                volume=0.0,
                                status="ERROR",
                                timestamp=datetime.now(IST),
                                freshness="UNAVAILABLE",
                                is_valid=False,
                                error_message=str(err),
                            )

            # Update cache if at least one new valid quote was received
            if new_valid_count > 0:
                self._cached_quotes.update(quotes)
                self._last_update = datetime.now(IST)
                self._last_error = None
            elif not any(q.is_valid for q in quotes.values()):
                self._last_error = "Provider returned no valid market data"

        except Exception as exc:
            log.exception("batch quote fetch encountered error: %s", exc)
            self._last_error = f"Live market provider error: {exc}"
            # Preserve existing cache
            if self._cached_quotes:
                return {s: self._cached_quotes[s] for s in sym_list if s in self._cached_quotes}

        return quotes

    def get_snapshot(
        self,
        symbols: Sequence[str] | None = None,
        force_refresh: bool = False,
    ) -> LiveMarketSnapshot:
        """Return a complete LiveMarketSnapshot including session and quote data."""
        session = evaluate_market_session(self.calendar)
        self._last_session = session

        quotes = self.fetch_quotes(symbols=symbols, force_refresh=force_refresh)

        # Evaluate freshness & staleness
        is_stale = False
        warning = None
        if self._last_update is not None:
            age = (datetime.now(IST) - self._last_update).total_seconds()
            if age > (self.cache_ttl_seconds * 3):
                is_stale = True
                warning = f"Data is stale (last updated {int(age)} seconds ago at {self._last_update.strftime('%H:%M:%S IST')})."
        elif not quotes or all(not q.is_valid for q in quotes.values()):
            is_stale = True
            warning = "Live market data currently unavailable from provider."

        return LiveMarketSnapshot(
            session=session,
            quotes=quotes,
            timestamp=self._last_update or datetime.now(IST),
            is_stale=is_stale,
            error=self._last_error,
            warning=warning,
        )

    def get_intraday_bars(
        self,
        symbol: str,
        period: str = "1d",
        interval: str = "5m",
    ) -> pd.DataFrame:
        """Download normalized intraday bars with VWAP calculation.

        Returns DataFrame indexed by DatetimeIndex in IST with lowercase
        columns: open, high, low, close, volume, vwap.
        """
        import yfinance as yf

        clean_sym = strip_exchange_suffix(symbol).upper()
        yf_sym = resolve_nse_symbol(clean_sym)

        try:
            ticker = yf.Ticker(yf_sym)
            raw = ticker.history(period=period, interval=interval)
            if raw is None or raw.empty:
                return pd.DataFrame()

            # Flatten MultiIndex if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            df = raw.rename(columns=str.lower)
            keep = ["open", "high", "low", "close", "volume"]
            missing = [c for c in keep if c not in df.columns]
            if missing:
                log.warning("missing intraday columns: %s", missing)
                return pd.DataFrame()

            df = df[keep].astype("float64")

            # Convert index to IST
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert(IST)
            else:
                df.index = df.index.tz_convert(IST)
            df.index.name = "timestamp"

            # Compute VWAP: sum(Price * Volume) / sum(Volume)
            typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
            pv = typical_price * df["volume"]
            cum_vol = df["volume"].cumsum()
            df["vwap"] = (pv.cumsum() / cum_vol.replace(0, np.nan)).ffill()

            df.attrs["symbol"] = clean_sym
            return df.dropna(subset=["open", "high", "low", "close"]).sort_index()

        except Exception as exc:
            log.warning("failed to fetch intraday bars for %s: %s", yf_sym, exc)
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# Paper Trading Synchronization
# ---------------------------------------------------------------------------

def sync_paper_portfolio_with_live_prices(
    quotes: dict[str, LiveQuote],
    tracker: Any,
    broker: Any | None = None,
) -> dict[str, Any]:
    """Feed latest validated market prices into paper trading portfolio and broker.

    Updates:
    - Tracker current prices
    - Unrealized P&L
    - Total mark-to-market equity
    - Broker asset valuations

    Strictly simulated: does NOT send real orders.
    """
    updated_count = 0
    total_val = 0.0

    for sym, quote in quotes.items():
        if quote.is_valid and quote.price > 0:
            tracker.update_price(sym, quote.price)
            if broker is not None and hasattr(broker, "update_price"):
                broker.update_price(sym, quote.price)
            updated_count += 1

    snap = tracker.get_snapshot()

    return {
        "updated_count": updated_count,
        "cash": snap.cash,
        "invested_equity": snap.invested_equity,
        "total_equity": snap.total_equity,
        "unrealized_pnl": snap.unrealized_pnl,
        "realized_pnl": snap.realized_pnl,
        "allocations": snap.allocations,
        "timestamp": snap.timestamp,
    }
