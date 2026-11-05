"""Indian stock market configuration and calendar.

Provides NSE-specific trading calendar, holidays, market hours, timezone
handling, and symbol conventions. All market-specific constants are centralised
here so the rest of the codebase stays market-agnostic.
"""

from .calendar import NSECalendar
from .constants import (
    CURRENCY,
    IST,
    MARKET_CLOSE,
    MARKET_OPEN,
    NSE_TRADING_DAYS_PER_YEAR,
    NSE_WATCHLIST,
    resolve_nse_symbol,
)

__all__ = [
    "NSECalendar",
    "CURRENCY",
    "IST",
    "MARKET_OPEN",
    "MARKET_CLOSE",
    "NSE_TRADING_DAYS_PER_YEAR",
    "NSE_WATCHLIST",
    "resolve_nse_symbol",
]
