"""NSE trading calendar.

Holidays are sourced from NSE circulars. The list must be updated annually
when NSE publishes the next year's holiday schedule (usually in December).

Update procedure:
    1. Visit https://www.nseindia.com/regulations/holiday-list
    2. Add the new year's holidays to ``_NSE_HOLIDAYS`` below.
    3. Run ``pytest tests/test_market_calendar.py`` to verify.

Design notes:
    - Holidays are stored as ``(year, month, day)`` tuples for readability and
      to avoid import-time ``datetime.date`` construction overhead.
    - Weekend detection uses ``.weekday()`` (Mon=0 … Sun=6), not holidays,
      because NSE never trades on weekends regardless of special sessions.
    - The calendar is intentionally simple: no half-day (Muhurat) handling yet.
      Muhurat trading on Diwali evening is an edge case that does not affect
      daily-bar backtests.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Iterator

from .constants import IST, MARKET_CLOSE, MARKET_OPEN

# ---------------------------------------------------------------------------
# NSE holidays – sourced from official NSE circulars
# ---------------------------------------------------------------------------
# fmt: off
_NSE_HOLIDAYS: dict[int, list[tuple[int, int]]] = {
    # ── 2024 ──────────────────────────────────────────────────────────────
    # Source: NSE Circular No. 20231222-16 (NSE/CM/2023/12/22)
    2024: [
        ( 1, 22),  # Special Holiday (Ram Mandir Pran Pratishtha)
        ( 1, 26),  # Republic Day
        ( 3,  8),  # Mahashivratri
        ( 3, 25),  # Holi
        ( 3, 29),  # Good Friday
        ( 4, 11),  # Id-Ul-Fitr (Ramadan)
        ( 4, 17),  # Ram Navami
        ( 5,  1),  # Maharashtra Day
        ( 5, 23),  # Buddha Purnima
        ( 6, 17),  # Eid-Ul-Adha (Bakri Id)
        ( 7, 17),  # Muharram
        ( 8, 15),  # Independence Day
        ( 9, 16),  # Milad-un-Nabi (Prophet's Birthday)
        (10,  2),  # Mahatma Gandhi Jayanti
        (11,  1),  # Diwali Laxmi Pujan *
        (11, 15),  # Guru Nanak Jayanti
        (12, 25),  # Christmas
    ],
    # ── 2025 ──────────────────────────────────────────────────────────────
    # Source: NSE Circular No. 20241224-13
    2025: [
        ( 2, 26),  # Mahashivratri
        ( 3, 14),  # Holi
        ( 3, 31),  # Id-Ul-Fitr (Ramadan)
        ( 4, 10),  # Shri Mahavir Jayanti
        ( 4, 14),  # Dr. Baba Saheb Ambedkar Jayanti
        ( 4, 18),  # Good Friday
        ( 5,  1),  # Maharashtra Day
        ( 5, 12),  # Buddha Purnima
        ( 8, 15),  # Independence Day
        ( 8, 27),  # Janmashtami
        (10,  2),  # Mahatma Gandhi Jayanti
        (10, 20),  # Diwali Laxmi Pujan *
        (10, 21),  # Diwali Balipratipada
        (11,  5),  # Prakash Gurpurb Sri Guru Nanak Dev
        (12, 25),  # Christmas
    ],
    # ── 2026 ──────────────────────────────────────────────────────────────
    # Source: NSE Circular (update when published, typically Dec 2025)
    # These are PROVISIONAL based on known fixed holidays and expected
    # dates for lunar-calendar festivals. Verify against the official
    # NSE circular before using for production trading decisions.
    2026: [
        ( 1, 26),  # Republic Day
        ( 2, 17),  # Mahashivratri
        ( 3,  3),  # Holi
        ( 3, 20),  # Id-Ul-Fitr (Ramadan) – tentative
        ( 3, 30),  # Shri Mahavir Jayanti
        ( 4,  3),  # Good Friday
        ( 4, 14),  # Dr. Baba Saheb Ambedkar Jayanti
        ( 5,  1),  # Maharashtra Day / May Day
        ( 5, 29),  # Eid-Ul-Adha (Bakri Id) – tentative
        ( 8, 15),  # Independence Day (Saturday – may not be listed)
        ( 8, 17),  # Janmashtami
        (10,  2),  # Mahatma Gandhi Jayanti
        (10,  8),  # Dussehra
        (10, 25),  # Diwali Laxmi Pujan – tentative *
        (11, 24),  # Guru Nanak Jayanti
        (12, 25),  # Christmas
    ],
}
# fmt: on


def _build_holiday_set() -> set[date]:
    """Convert the nested dict of tuples into a flat set of ``date`` objects."""
    holidays: set[date] = set()
    for year, entries in _NSE_HOLIDAYS.items():
        for month, day in entries:
            d = date(year, month, day)
            # Only include holidays that fall on weekdays (Mon–Fri).
            # Weekend holidays don't affect the trading calendar.
            if d.weekday() < 5:
                holidays.add(d)
    return holidays


class NSECalendar:
    """NSE trading calendar with holiday and market-hours awareness.

    >>> cal = NSECalendar()
    >>> cal.is_trading_day(date(2025, 1, 26))  # Republic Day (Sunday in 2025)
    False
    >>> cal.is_trading_day(date(2025, 3, 14))  # Holi (Friday in 2025)
    False
    """

    def __init__(self, extra_holidays: list[date] | None = None):
        self._holidays = _build_holiday_set()
        if extra_holidays:
            self._holidays.update(extra_holidays)

    # ---------------------------------------------------------------- queries

    @property
    def holidays(self) -> frozenset[date]:
        """All known holidays as an immutable set."""
        return frozenset(self._holidays)

    def is_holiday(self, d: date) -> bool:
        """Is *d* an NSE-declared holiday?"""
        return d in self._holidays

    def is_weekend(self, d: date) -> bool:
        """Is *d* a Saturday or Sunday?"""
        return d.weekday() >= 5

    def is_trading_day(self, d: date) -> bool:
        """Is *d* a regular NSE trading session?

        True when *d* is a weekday that is not an NSE holiday.
        """
        return d.weekday() < 5 and d not in self._holidays

    def next_trading_day(self, d: date) -> date:
        """Return the first trading day strictly after *d*."""
        candidate = d + timedelta(days=1)
        while not self.is_trading_day(candidate):
            candidate += timedelta(days=1)
        return candidate

    def previous_trading_day(self, d: date) -> date:
        """Return the most recent trading day strictly before *d*."""
        candidate = d - timedelta(days=1)
        while not self.is_trading_day(candidate):
            candidate -= timedelta(days=1)
        return candidate

    def trading_days_between(
        self, start: date, end: date, inclusive: bool = True
    ) -> list[date]:
        """Return all trading days in [start, end] (or (start, end) if not inclusive)."""
        days: list[date] = []
        current = start if inclusive else start + timedelta(days=1)
        limit = end if inclusive else end - timedelta(days=1)
        while current <= limit:
            if self.is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)
        return days

    def trading_days_in_year(self, year: int) -> int:
        """Count trading days in a calendar year.

        For years without holiday data, estimates from ``NSE_TRADING_DAYS_PER_YEAR``.
        """
        if year not in _NSE_HOLIDAYS:
            from .constants import NSE_TRADING_DAYS_PER_YEAR
            return NSE_TRADING_DAYS_PER_YEAR
        return len(self.trading_days_between(date(year, 1, 1), date(year, 12, 31)))

    def iter_trading_days(
        self, start: date, end: date
    ) -> Iterator[date]:
        """Yield trading days from *start* through *end* inclusive."""
        current = start
        while current <= end:
            if self.is_trading_day(current):
                yield current
            current += timedelta(days=1)

    # ------------------------------------------------------------ market hours

    def is_market_open(self, dt: datetime | None = None) -> bool:
        """Is the NSE equity segment currently open?

        *dt* defaults to the current time if not provided.
        """
        if dt is None:
            dt = datetime.now(IST)
        elif dt.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")
        else:
            dt = dt.astimezone(IST)

        if not self.is_trading_day(dt.date()):
            return False

        t = dt.time()
        return MARKET_OPEN <= t < MARKET_CLOSE

    def market_open_datetime(self, d: date) -> datetime:
        """Return the market open datetime for date *d* in IST."""
        return datetime.combine(d, MARKET_OPEN, tzinfo=IST)

    def market_close_datetime(self, d: date) -> datetime:
        """Return the market close datetime for date *d* in IST."""
        return datetime.combine(d, MARKET_CLOSE, tzinfo=IST)

    # ------------------------------------------------------------ mutation

    def add_holidays(self, holidays: list[date]) -> None:
        """Add extra holidays (e.g., loaded from a config file at runtime)."""
        for d in holidays:
            if d.weekday() < 5:
                self._holidays.add(d)

    def __repr__(self) -> str:
        years = sorted({d.year for d in self._holidays})
        return f"NSECalendar(years={years}, holidays={len(self._holidays)})"
