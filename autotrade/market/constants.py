"""Indian market constants.

All NSE-specific values live here so they are easy to find, easy to update, and
impossible to scatter across dozens of files. If the market changes (different
holidays, different hours), this is the one file you edit.
"""

from __future__ import annotations

from datetime import time, timezone, timedelta


# ---------------------------------------------------------------------------
# Timezone
# ---------------------------------------------------------------------------

#: Indian Standard Time (UTC+05:30), used for all market-hour calculations.
IST = timezone(timedelta(hours=5, minutes=30))

# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------

CURRENCY = "INR"
CURRENCY_SYMBOL = "₹"

# ---------------------------------------------------------------------------
# Market hours
# ---------------------------------------------------------------------------

#: NSE equity segment: pre-open 9:00, continuous trading 9:15–15:30.
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
PRE_OPEN_START = time(9, 0)

# ---------------------------------------------------------------------------
# Trading days
# ---------------------------------------------------------------------------

#: Approximate number of NSE trading days per year.  The exact count varies
#: (2023: 249, 2024: 247, 2025: ~248).  This is used for annualisation of
#: returns and volatility.  Override it in config if your backtest is sensitive.
NSE_TRADING_DAYS_PER_YEAR = 248

# ---------------------------------------------------------------------------
# Default watchlist
# ---------------------------------------------------------------------------

#: Initial NSE watchlist — large-cap, liquid names.
NSE_WATCHLIST: list[str] = [
    "RELIANCE",
    "TCS",
    "INFY",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "ITC",
    "LT",
    "BHARTIARTL",
    "AXISBANK",
]

# ---------------------------------------------------------------------------
# Symbol helpers
# ---------------------------------------------------------------------------

#: Suffix that yfinance requires for NSE-listed equities.
YFINANCE_NSE_SUFFIX = ".NS"
YFINANCE_BSE_SUFFIX = ".BO"


def resolve_nse_symbol(symbol: str) -> str:
    """Ensure a symbol has the ``.NS`` suffix for yfinance queries.

    If the symbol already ends with ``.NS`` or ``.BO``, return it unchanged.
    Otherwise, append ``.NS``.

    >>> resolve_nse_symbol("RELIANCE")
    'RELIANCE.NS'
    >>> resolve_nse_symbol("RELIANCE.NS")
    'RELIANCE.NS'
    >>> resolve_nse_symbol("RELIANCE.BO")
    'RELIANCE.BO'
    """
    upper = symbol.upper().strip()
    if upper.endswith(YFINANCE_NSE_SUFFIX) or upper.endswith(YFINANCE_BSE_SUFFIX):
        return upper
    return f"{upper}{YFINANCE_NSE_SUFFIX}"


def strip_exchange_suffix(symbol: str) -> str:
    """Remove ``.NS`` or ``.BO`` suffix from a symbol.

    >>> strip_exchange_suffix("RELIANCE.NS")
    'RELIANCE'
    >>> strip_exchange_suffix("TCS")
    'TCS'
    """
    upper = symbol.upper().strip()
    for suffix in (YFINANCE_NSE_SUFFIX, YFINANCE_BSE_SUFFIX):
        if upper.endswith(suffix):
            return upper[: -len(suffix)]
    return upper
