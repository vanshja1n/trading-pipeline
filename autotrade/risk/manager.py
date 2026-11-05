"""Risk management: position sizing and the kill switch.

The strategy proposes, the risk manager disposes. Keeping them separate means
you can test a strategy's edge and your survival rules independently -- and it
means one bad signal can't put the whole account on the line.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import RiskConfig

log = logging.getLogger(__name__)


@dataclass
class StopDecision:
    triggered: bool
    price: float | None = None
    reason: str = ""


class RiskManager:
    def __init__(self, config: RiskConfig, starting_equity: float):
        self.config = config
        self.starting_equity = starting_equity
        self.suspension_history: list[dict[str, str]] = []
        self.reset(starting_equity)

    def reset(self, equity: float) -> None:
        self.peak_equity = equity
        self.session_start_equity = equity
        self.halted = False
        self.halt_reason = ""

    # ------------------------------------------------------------------ sizing

    def target_weight(self, signal: float, volatility: float | None = None) -> float:
        """Convert a strategy signal in [-1, 1] into a portfolio weight.

        ``volatility`` is the annualized realized vol of the asset; when
        vol targeting is enabled, a calmer asset gets a bigger position and a
        wild one gets a smaller position, so risk per trade stays comparable.
        """
        if self.halted:
            return 0.0

        weight = float(signal) * self.config.max_position_weight

        if self.config.target_volatility > 0 and volatility and volatility > 0:
            scale = min(self.config.target_volatility / volatility, self.config.max_leverage)
            weight *= scale

        cap = self.config.max_gross_exposure
        return max(-cap, min(cap, weight))

    def shares_for_weight(self, weight: float, equity: float, price: float) -> int:
        """Whole shares implied by a target weight. Fractional shares are a
        broker feature, not a modeling one -- rounding down keeps fills honest."""
        if price <= 0:
            return 0
        return int(abs(weight) * equity / price) * (1 if weight >= 0 else -1)

    def shares_for_risk_budget(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        risk_fraction: float = 0.01,
    ) -> int:
        """Size position such that hitting stop_price loses exactly risk_fraction of equity.

        Parameters
        ----------
        equity : float
            Current total equity.
        entry_price : float
            Expected entry price.
        stop_price : float
            Price at which the stop-loss will trigger.
        risk_fraction : float
            Maximum fraction of equity to lose on this trade (e.g. 0.01 = 1%).
        """
        if equity <= 0 or entry_price <= 0 or stop_price <= 0:
            return 0
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 0
        max_loss = equity * risk_fraction
        target_shares = int(max_loss / risk_per_share)

        # Enforce max position weight cap
        max_allowed_shares = self.shares_for_weight(
            self.config.max_position_weight, equity, entry_price
        )
        return min(target_shares, abs(max_allowed_shares))

    # ------------------------------------------------------------- kill switch

    def update_equity(self, equity: float) -> bool:
        """Feed the latest mark-to-market equity. Returns True if trading is halted."""
        self.peak_equity = max(self.peak_equity, equity)

        drawdown = 1 - equity / self.peak_equity if self.peak_equity > 0 else 0.0
        if drawdown >= self.config.max_drawdown and not self.halted:
            self._halt(f"max drawdown hit: -{drawdown:.1%} from peak {self.peak_equity:,.2f}")

        daily = 1 - equity / self.session_start_equity if self.session_start_equity > 0 else 0.0
        if daily >= self.config.max_daily_loss and not self.halted:
            self._halt(f"daily loss limit hit: -{daily:.1%} on the session")

        return self.halted

    def start_session(self, equity: float) -> None:
        """Call at the start of each trading day to reset the daily loss budget."""
        self.session_start_equity = equity

    def _halt(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason
        self.suspension_history.append({"reason": reason})
        log.warning("TRADING HALTED -- %s", reason)

    def resume(self) -> None:
        """Manual override. Deliberately not automatic: if the kill switch fired,
        a human should look at why before the bot starts again."""
        log.warning("trading resumed manually (was: %s)", self.halt_reason)
        self.halted = False
        self.halt_reason = ""

    # ---------------------------------------------------------------- stop loss

    def check_stop(
        self,
        position: int,
        entry_price: float,
        bar_low: float,
        bar_high: float,
        atr_value: float | None,
    ) -> StopDecision:
        """Was an ATR stop touched during this bar?

        Assumes the worst realistic fill: exactly at the stop price. Real fills
        can gap through it, so treat backtested stop exits as optimistic.
        """
        k = self.config.stop_loss_atr
        if not k or position == 0 or not atr_value or atr_value <= 0:
            return StopDecision(False)

        if position > 0:
            stop_price = entry_price - k * atr_value
            if bar_low <= stop_price:
                return StopDecision(True, stop_price, f"long stop at {stop_price:.2f}")
        else:
            stop_price = entry_price + k * atr_value
            if bar_high >= stop_price:
                return StopDecision(True, stop_price, f"short stop at {stop_price:.2f}")
        return StopDecision(False)
