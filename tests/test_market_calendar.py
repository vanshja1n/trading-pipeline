"""Tests for NSE calendar, trading hours, holidays, and symbol resolution."""

from datetime import date, datetime, time
import pytest

from autotrade.market.calendar import NSECalendar
from autotrade.market.constants import (
    IST,
    NSE_WATCHLIST,
    resolve_nse_symbol,
    strip_exchange_suffix,
)


def test_symbol_helpers():
    assert resolve_nse_symbol("RELIANCE") == "RELIANCE.NS"
    assert resolve_nse_symbol("reliance") == "RELIANCE.NS"
    assert resolve_nse_symbol("TCS.NS") == "TCS.NS"
    assert resolve_nse_symbol("TCS.BO") == "TCS.BO"

    assert strip_exchange_suffix("RELIANCE.NS") == "RELIANCE"
    assert strip_exchange_suffix("RELIANCE.BO") == "RELIANCE"
    assert strip_exchange_suffix("TCS") == "TCS"
    assert strip_exchange_suffix("tcs.ns") == "TCS"


def test_nse_calendar_holidays():
    cal = NSECalendar()

    # 2024 Republic Day: Friday Jan 26, 2024
    assert not cal.is_trading_day(date(2024, 1, 26))
    assert cal.is_holiday(date(2024, 1, 26))

    # Regular weekday in Jan 2024: Wednesday Jan 24, 2024
    assert cal.is_trading_day(date(2024, 1, 24))
    assert not cal.is_holiday(date(2024, 1, 24))

    # Weekend: Saturday Jan 27, 2024
    assert not cal.is_trading_day(date(2024, 1, 27))
    assert cal.is_weekend(date(2024, 1, 27))


def test_nse_calendar_next_previous_trading_day():
    cal = NSECalendar()

    # Jan 25, 2024 was Thursday (trading day). Next day Jan 26 was Republic Day holiday (Friday).
    # Then Jan 27-28 weekend. Next trading day should be Jan 29, 2024 (Monday).
    next_day = cal.next_trading_day(date(2024, 1, 25))
    assert next_day == date(2024, 1, 29)

    # Previous trading day from Monday Jan 29, 2024 should be Thursday Jan 25, 2024
    prev_day = cal.previous_trading_day(date(2024, 1, 29))
    assert prev_day == date(2024, 1, 25)


def test_nse_calendar_market_hours():
    cal = NSECalendar()

    # Wednesday Jan 24, 2024 at 10:30 AM IST (market open)
    open_dt = datetime(2024, 1, 24, 10, 30, tzinfo=IST)
    assert cal.is_market_open(open_dt)

    # Wednesday Jan 24, 2024 at 8:45 AM IST (before pre-open)
    early_dt = datetime(2024, 1, 24, 8, 45, tzinfo=IST)
    assert not cal.is_market_open(early_dt)

    # Wednesday Jan 24, 2024 at 4:00 PM IST (after market close)
    late_dt = datetime(2024, 1, 24, 16, 0, tzinfo=IST)
    assert not cal.is_market_open(late_dt)

    # Republic Day Jan 26, 2024 at 11:00 AM IST (holiday)
    holiday_dt = datetime(2024, 1, 26, 11, 0, tzinfo=IST)
    assert not cal.is_market_open(holiday_dt)


def test_watchlist_contains_major_stocks():
    assert "RELIANCE" in NSE_WATCHLIST
    assert "TCS" in NSE_WATCHLIST
    assert "INFY" in NSE_WATCHLIST
    assert "HDFCBANK" in NSE_WATCHLIST
