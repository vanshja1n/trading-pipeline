"""Tests for Live Market Integration layer.

Verifies:
- Live quote parsing and canonicalization
- Invalid and missing price handling (No Fake Data policy)
- Market session states (Open, Closed, Pre-open, Weekend, Holiday)
- Cache and TTL refresh behavior
- Stale data detection
- Provider / API failure resilience
- Paper portfolio mark-to-market valuation sync
- Intraday bars and VWAP calculation
"""

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from autotrade.data.live import (
    LiveMarketFeed,
    LiveMarketSnapshot,
    LiveQuote,
    MarketSessionStatus,
    evaluate_market_session,
    sync_paper_portfolio_with_live_prices,
)
from autotrade.execution.simulated import SimulatedBroker
from autotrade.market.calendar import NSECalendar
from autotrade.market.constants import IST, NSE_WATCHLIST
from autotrade.portfolio.tracker import PortfolioTracker


# ---------------------------------------------------------------------------
# 1. Live Quote Parsing & Validation
# ---------------------------------------------------------------------------

def test_live_quote_parsing_valid():
    feed = LiveMarketFeed()
    ts = datetime(2026, 11, 4, 10, 30, tzinfo=IST)

    quote = feed.parse_quote_data(
        symbol="RELIANCE.NS",
        price=1350.50,
        previous_close=1300.00,
        volume=5_000_000,
        open_price=1310.00,
        high_price=1360.00,
        low_price=1305.00,
        raw_timestamp=ts,
        is_market_open=True,
    )

    assert quote.is_valid is True
    assert quote.symbol == "RELIANCE"
    assert quote.price == 1350.50
    assert quote.previous_close == 1300.00
    assert quote.change == 50.50
    assert quote.change_pct == pytest.approx(3.88, abs=0.01)
    assert quote.volume == 5_000_000
    assert quote.status == "TRADING"
    assert quote.freshness == "DELAYED"

    d = quote.to_dict()
    assert d["Symbol"] == "RELIANCE"
    assert "₹1,350.50" in d["Price"]
    assert "+50.50" in d["Change"]
    assert "+3.88%" in d["Change %"]


def test_live_quote_invalid_and_missing_prices():
    """Verify that non-positive or missing prices are rejected with NO fake data generation."""
    feed = LiveMarketFeed()

    # None price
    q_none = feed.parse_quote_data("TCS", price=None, previous_close=3500.0)
    assert not q_none.is_valid
    assert q_none.price == 0.0
    assert q_none.status == "UNAVAILABLE"
    assert q_none.freshness == "UNAVAILABLE"

    # Zero price
    q_zero = feed.parse_quote_data("INFY", price=0.0, previous_close=1500.0)
    assert not q_zero.is_valid
    assert q_zero.status == "UNAVAILABLE"

    # Negative price
    q_neg = feed.parse_quote_data("SBIN", price=-100.0, previous_close=700.0)
    assert not q_neg.is_valid
    assert q_neg.status == "UNAVAILABLE"


# ---------------------------------------------------------------------------
# 2. Market Session States & Calendar Awareness
# ---------------------------------------------------------------------------

def test_market_session_states():
    cal = NSECalendar()

    # Weekend (Saturday, Nov 7, 2026)
    sat_dt = datetime(2026, 11, 7, 12, 0, tzinfo=IST)
    session = evaluate_market_session(cal, sat_dt)
    assert session.status == "WEEKEND"
    assert not session.is_open

    # Holiday (Republic Day: Jan 26, 2024 - Friday)
    holiday_dt = datetime(2024, 1, 26, 11, 0, tzinfo=IST)
    session_hol = evaluate_market_session(cal, holiday_dt)
    assert session_hol.status == "HOLIDAY"
    assert not session_hol.is_open

    # Regular Trading Day: Pre-Open (Wednesday, Jan 24, 2024 at 09:05 AM IST)
    pre_open_dt = datetime(2024, 1, 24, 9, 5, tzinfo=IST)
    session_pre = evaluate_market_session(cal, pre_open_dt)
    assert session_pre.status == "PRE-OPEN"
    assert not session_pre.is_open

    # Regular Trading Day: Market Open (Wednesday, Jan 24, 2024 at 10:15 AM IST)
    open_dt = datetime(2024, 1, 24, 10, 15, tzinfo=IST)
    session_open = evaluate_market_session(cal, open_dt)
    assert session_open.status == "OPEN"
    assert session_open.is_open

    # Regular Trading Day: After Market Close (Wednesday, Jan 24, 2024 at 16:00 IST)
    close_dt = datetime(2024, 1, 24, 16, 0, tzinfo=IST)
    session_close = evaluate_market_session(cal, close_dt)
    assert session_close.status == "CLOSED"
    assert not session_close.is_open


# ---------------------------------------------------------------------------
# 3. Caching and Refresh Behavior
# ---------------------------------------------------------------------------

def test_caching_and_force_refresh():
    feed = LiveMarketFeed(cache_ttl_seconds=5.0)

    # Seed cache manually
    now = datetime.now(IST)
    q1 = LiveQuote(
        symbol="RELIANCE",
        price=1300.0,
        previous_close=1290.0,
        change=10.0,
        change_pct=0.78,
        volume=1000,
        status="CLOSED",
        timestamp=now,
        is_valid=True,
    )
    feed._cached_quotes = {"RELIANCE": q1}
    feed._last_update = now

    # Within TTL: returns cached quote
    assert feed.is_cache_valid()
    quotes = feed.fetch_quotes(symbols=["RELIANCE"], force_refresh=False)
    assert quotes["RELIANCE"].price == 1300.0

    # Expire TTL
    feed._last_update = now - timedelta(seconds=10.0)
    assert not feed.is_cache_valid()


def test_stale_data_detection():
    feed = LiveMarketFeed(cache_ttl_seconds=10.0)
    old_time = datetime.now(IST) - timedelta(seconds=50.0)  # > 3 * TTL

    feed._cached_quotes = {
        "TCS": LiveQuote(
            symbol="TCS",
            price=2500.0,
            previous_close=2500.0,
            change=0.0,
            change_pct=0.0,
            volume=500,
            status="CLOSED",
            timestamp=old_time,
            is_valid=True,
        )
    }
    feed._last_update = old_time

    # Simulate provider failure so stale cache is retained
    with patch.object(feed, "fetch_single_quote", side_effect=RuntimeError("Network timeout")):
        snapshot = feed.get_snapshot(symbols=["TCS"], force_refresh=True)
        assert snapshot.is_stale is True
        assert snapshot.warning is not None
        assert "stale" in snapshot.warning.lower()



# ---------------------------------------------------------------------------
# 4. Provider / API Failure Resilience
# ---------------------------------------------------------------------------

def test_provider_failure_graceful_handling():
    feed = LiveMarketFeed(cache_ttl_seconds=30.0)

    # Mock yfinance Ticker throwing an exception (e.g. Network error)
    with patch("yfinance.Ticker", side_effect=RuntimeError("Connection refused by remote host")):
        q = feed.fetch_single_quote("RELIANCE")
        assert not q.is_valid
        assert q.status == "ERROR"
        assert q.price == 0.0
        assert "Connection refused" in q.error_message

        # Batch fetch should not crash
        quotes = feed.fetch_quotes(symbols=["RELIANCE", "TCS"], force_refresh=True)
        assert "RELIANCE" in quotes
        assert not quotes["RELIANCE"].is_valid
        assert not quotes["TCS"].is_valid

        # Snapshot should not crash and should record error
        snap = feed.get_snapshot(symbols=["RELIANCE"], force_refresh=True)
        assert snap.is_stale is True
        assert snap.error is not None


# ---------------------------------------------------------------------------
# 5. Paper Trading Portfolio Valuation Sync
# ---------------------------------------------------------------------------

def test_paper_portfolio_valuation_sync():
    tracker = PortfolioTracker(starting_cash=500_000.0)
    broker = SimulatedBroker(starting_cash=500_000.0)

    # Buy 100 shares of RELIANCE at ₹1,200 (Total cost: ₹120,000)
    tracker.record_fill(symbol="RELIANCE", side="buy", qty=100, fill_price=1200.0)
    broker._positions["RELIANCE"] = 100
    broker._avg_prices["RELIANCE"] = 1200.0
    broker.cash = 380_000.0

    # Initial check
    snap_init = tracker.get_snapshot()
    assert snap_init.cash == 380_000.0
    assert snap_init.invested_equity == 120_000.0
    assert snap_init.total_equity == 500_000.0
    assert snap_init.unrealized_pnl == 0.0

    # Now live prices arrive: RELIANCE rises to ₹1,350 (+₹150/share)
    live_quotes = {
        "RELIANCE": LiveQuote(
            symbol="RELIANCE",
            price=1350.0,
            previous_close=1200.0,
            change=150.0,
            change_pct=12.5,
            volume=2_000_000,
            status="TRADING",
            timestamp=datetime.now(IST),
            is_valid=True,
        )
    }

    result = sync_paper_portfolio_with_live_prices(live_quotes, tracker, broker)

    # Valuation check
    assert result["updated_count"] == 1
    assert result["invested_equity"] == 100 * 1350.0  # ₹135,000
    assert result["total_equity"] == 380_000.0 + 135_000.0  # ₹515,000
    assert result["unrealized_pnl"] == 100 * (1350.0 - 1200.0)  # +₹15,000

    # Broker mark-to-market check
    acc = broker.get_account()
    assert acc.equity == 515_000.0


# ---------------------------------------------------------------------------
# 6. Intraday Bars and VWAP Calculation
# ---------------------------------------------------------------------------

def test_intraday_bars_and_vwap_calculation():
    feed = LiveMarketFeed()

    # Synthetic intraday bars returned by provider
    timestamps = pd.date_range("2026-11-04 09:15", periods=5, freq="5min", tz=IST)
    mock_df = pd.DataFrame(
        {
            "Open": [100.0, 102.0, 101.0, 103.0, 105.0],
            "High": [103.0, 104.0, 103.0, 106.0, 107.0],
            "Low": [99.0, 101.0, 100.0, 102.0, 104.0],
            "Close": [102.0, 101.0, 103.0, 105.0, 106.0],
            "Volume": [1000.0, 2000.0, 1500.0, 3000.0, 2500.0],
        },
        index=timestamps,
    )

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_df

    with patch("yfinance.Ticker", return_value=mock_ticker):
        bars = feed.get_intraday_bars("RELIANCE", period="1d", interval="5m")

        assert not bars.empty
        assert "vwap" in bars.columns
        assert list(bars.columns) == ["open", "high", "low", "close", "volume", "vwap"]
        assert bars.index.tz == IST
        assert len(bars) == 5

        # Check VWAP for bar 0: typical price = (103 + 99 + 102) / 3 = 101.333
        bar0_typical = (103.0 + 99.0 + 102.0) / 3.0
        assert bars["vwap"].iloc[0] == pytest.approx(bar0_typical, abs=0.01)
