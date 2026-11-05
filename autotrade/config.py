"""Configuration objects for backtesting, risk, and live trading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class MarketConfig:
    """Market-specific settings.

    Centralises all market-dependent constants so the rest of the codebase
    remains market-agnostic.  Pass this to ``BacktestConfig`` and ``LiveConfig``
    to switch between US and Indian markets without touching strategy code.
    """

    #: ``"US"`` or ``"IN"`` — controls symbol resolution and calendar.
    market: str = "IN"
    #: Display currency code.
    currency: str = "INR"
    #: Currency symbol for display.
    currency_symbol: str = "₹"
    #: IANA timezone name for the market.
    timezone: str = "Asia/Kolkata"
    #: Approximate trading days per year (used for annualisation).
    trading_days_per_year: int = 248
    #: Default watchlist when none is specified.
    default_symbols: list[str] = field(default_factory=lambda: [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "SBIN", "ITC", "LT", "BHARTIARTL", "AXISBANK",
    ])

    def __post_init__(self) -> None:
        self.market = self.market.upper()
        if self.market not in ("US", "IN"):
            raise ValueError(f"market must be 'US' or 'IN', got {self.market!r}")


# Pre-built market configs for convenience.
MARKET_INDIA = MarketConfig()
MARKET_US = MarketConfig(
    market="US",
    currency="USD",
    currency_symbol="$",
    timezone="US/Eastern",
    trading_days_per_year=252,
    default_symbols=["SPY"],
)


@dataclass
class RiskConfig:
    """Everything that decides how much you are allowed to lose.

    The risk manager is deliberately separate from the strategy: a strategy
    says *which way* to lean, risk decides *how big* and *when to stop*.
    """

    # Fraction of equity allocated to a single full-strength signal (0-1).
    max_position_weight: float = 0.95
    # Hard cap on gross exposure across all symbols.
    max_gross_exposure: float = 1.0
    # Flatten and stop trading once equity falls this far below its peak.
    max_drawdown: float = 0.20
    # Stop trading for the rest of the session after this daily loss (live only).
    max_daily_loss: float = 0.05
    # Exit a position if it moves this many ATRs against the entry. 0 disables.
    stop_loss_atr: float = 0.0
    # Scale positions toward this annualized volatility. 0 disables vol targeting.
    target_volatility: float = 0.0
    # Ceiling on the vol-targeting multiplier, so a quiet market can't lever you up.
    max_leverage: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.max_position_weight <= 1:
            raise ValueError("max_position_weight must be in (0, 1]")
        if not 0 < self.max_drawdown < 1:
            raise ValueError("max_drawdown must be in (0, 1)")
        if self.max_leverage < 1:
            raise ValueError("max_leverage must be >= 1")


@dataclass
class BacktestConfig:
    starting_cash: float = 10_000.0
    # Round-trip cost estimate in basis points, charged on every fill.
    slippage_bps: float = 2.0
    # Per-share commission (Alpaca equities are $0, but model it anyway).
    commission_per_share: float = 0.0
    # Use Indian equity transaction cost structure (STT, GST, Stamp duty, etc.)
    use_indian_costs: bool = False
    # Minimum change in target position before we bother trading (avoids churn).
    rebalance_threshold: float = 0.05
    # Annualization factor: 248 NSE trading days by default, 252 for US.
    periods_per_year: int = 248
    # Risk-free rate used by Sharpe (Indian 10Y G-Sec yield ~7%).
    risk_free_rate: float = 0.07
    # Market config for market-specific defaults.
    market: MarketConfig = field(default_factory=MarketConfig)

    def __post_init__(self) -> None:
        # If periods_per_year was left at the default, sync it with market.
        # This avoids silently using Indian annualisation for US backtests.
        if self.periods_per_year == 248 and self.market.market == "US":
            self.periods_per_year = 252
            self.risk_free_rate = 0.04


@dataclass
class LiveConfig:
    """Live/paper trading settings. Defaults are intentionally paranoid."""

    symbols: list[str] = field(default_factory=lambda: ["RELIANCE"])
    # Bar size the strategy consumes: "1Day", "1Hour", "15Min", "5Min", "1Min".
    timeframe: str = "1Day"
    # How many historical bars to pull so indicators have enough warmup.
    lookback_bars: int = 400
    # Seconds between decision loops.
    poll_seconds: int = 60
    # Minimum drift between current and target weight before rebalancing.
    # Mirrors BacktestConfig.rebalance_threshold: without it, rounding noise in
    # `int(weight * equity / price)` triggers one-share round trips every bar.
    rebalance_threshold: float = 0.05
    # When True, log intended orders but never submit them.
    dry_run: bool = True
    # Must be flipped explicitly to touch a real-money account.
    allow_live_money: bool = False
    # Market configuration.
    market: MarketConfig = field(default_factory=MarketConfig)

    @property
    def api_key(self) -> str | None:
        return os.getenv("ALPACA_API_KEY")

    @property
    def api_secret(self) -> str | None:
        return os.getenv("ALPACA_API_SECRET")

    @property
    def paper(self) -> bool:
        """Paper unless the user has explicitly opted into live money."""
        env_live = os.getenv("ALPACA_LIVE", "").lower() in ("1", "true", "yes")
        return not (env_live and self.allow_live_money)

