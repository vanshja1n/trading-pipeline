"""Event-driven backtester for a single symbol.

Design rules that keep the results honest:

1. A signal computed from bar ``t``'s close is executed at bar ``t+1``'s open.
   Never at bar t's close -- you didn't have that information in time.
2. Every fill pays slippage and commission.
3. Positions are whole shares sized off *current* equity, not starting equity.
4. Stops fill at the stop price, which is optimistic. Real gaps are worse.

If a backtest looks amazing, assume you broke rule 1 somewhere before you
assume you found an edge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from ..config import BacktestConfig, RiskConfig
from ..reporting import metrics as metrics_mod
from ..risk.manager import RiskManager
from ..strategy import indicators as ind
from ..strategy.base import Strategy

from ..execution.costs import FlatCostModel, IndianEquityCostModel, TransactionCostModel

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity: pd.Series
    positions: pd.Series
    target_weights: pd.Series
    trades: pd.DataFrame
    metrics: metrics_mod.Metrics
    strategy_name: str
    symbol: str
    halted: bool = False
    halt_reason: str = ""
    _config: BacktestConfig = field(default_factory=BacktestConfig, repr=False)

    def summary(self) -> str:
        header = f"{self.strategy_name} on {self.symbol}"
        window = f"{self.equity.index[0].date()} -> {self.equity.index[-1].date()}"
        lines = [header, window, str(self.metrics)]
        if self.halted:
            lines.append(f"  !! kill switch fired: {self.halt_reason}")
        return "\n".join(lines)


class Backtester:
    def __init__(
        self,
        strategy: Strategy,
        config: BacktestConfig | None = None,
        risk_config: RiskConfig | None = None,
        cost_model: TransactionCostModel | None = None,
    ):
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.risk_config = risk_config or RiskConfig()
        if cost_model is not None:
            self.cost_model = cost_model
        elif self.config.use_indian_costs:
            self.cost_model = IndianEquityCostModel()
        else:
            self.cost_model = FlatCostModel(self.config.commission_per_share)

    def run(self, bars: pd.DataFrame, symbol: str = "?") -> BacktestResult:
        cfg = self.config
        signals = self.strategy(bars)
        risk = RiskManager(self.risk_config, cfg.starting_cash)

        atr = ind.atr(bars["high"], bars["low"], bars["close"], 14)
        vol = ind.realized_volatility(bars["close"], 20, cfg.periods_per_year)

        cash = cfg.starting_cash
        position = 0          # shares, negative = short
        avg_entry = 0.0
        pending_target: float | None = None
        blocked_sign = 0      # suppress re-entry in this direction after a stop

        equity_curve: list[float] = []
        position_track: list[int] = []
        weight_track: list[float] = []
        trades: list[dict] = []

        def fill(target_shares: int, raw_price: float, ts, reason: str) -> None:
            """Move the position to target_shares at raw_price, with costs."""
            nonlocal cash, position, avg_entry
            delta = target_shares - position
            if delta == 0:
                return

            slip = cfg.slippage_bps / 10_000.0
            price = raw_price * (1 + slip) if delta > 0 else raw_price * (1 - slip)
            
            side_str = "BUY" if delta > 0 else "SELL"
            breakdown = self.cost_model.calculate(side_str, abs(delta), price)
            commission = breakdown.total

            realized = None
            # Any part of `delta` that shrinks the existing position realizes P&L.
            if position != 0 and (position > 0) != (delta > 0):
                closed = min(abs(delta), abs(position))
                realized = (price - avg_entry) * closed * (1 if position > 0 else -1)

            cash -= delta * price + commission

            if position == 0 or (position > 0) == (delta > 0):
                # Opening or adding: blend into the average entry price.
                avg_entry = (avg_entry * abs(position) + price * abs(delta)) / abs(target_shares)
            elif target_shares != 0 and (target_shares > 0) != (position > 0):
                avg_entry = price  # flipped sides entirely
            elif target_shares == 0:
                avg_entry = 0.0

            position = target_shares
            trades.append(
                {
                    "timestamp": ts,
                    "side": "buy" if delta > 0 else "sell",
                    "shares": abs(delta),
                    "price": price,
                    "commission": commission,
                    "cost_breakdown": breakdown.to_dict(),
                    "position_after": position,
                    "pnl": realized,
                    "reason": reason,
                }
            )

        session_date = None
        for i, ts in enumerate(bars.index):
            # Reset the daily loss budget when the calendar day rolls over,
            # otherwise max_daily_loss silently becomes a since-inception limit.
            if ts.date() != session_date:
                session_date = ts.date()
                risk.start_session(cash + position * float(bars["open"].iloc[i]))

            open_, high, low, close = (
                float(bars["open"].iloc[i]),
                float(bars["high"].iloc[i]),
                float(bars["low"].iloc[i]),
                float(bars["close"].iloc[i]),
            )

            # 1. Execute the order decided at the previous close, at this open.
            if pending_target is not None:
                prior_vol = float(vol.iloc[i - 1]) if i > 0 and pd.notna(vol.iloc[i - 1]) else None
                weight = risk.target_weight(pending_target, prior_vol)
                equity_at_open = cash + position * open_
                desired = risk.shares_for_weight(weight, equity_at_open, open_)

                current_weight = position * open_ / equity_at_open if equity_at_open > 0 else 0.0
                moving_to_flat = desired == 0 and position != 0
                if moving_to_flat or abs(weight - current_weight) >= cfg.rebalance_threshold:
                    fill(desired, open_, ts, "signal")
                pending_target = None

            # 2. Did an ATR stop get touched inside this bar?
            if position != 0:
                atr_val = float(atr.iloc[i - 1]) if i > 0 and pd.notna(atr.iloc[i - 1]) else None
                stop = risk.check_stop(position, avg_entry, low, high, atr_val)
                if stop.triggered:
                    blocked_sign = 1 if position > 0 else -1
                    fill(0, stop.price, ts, stop.reason)

            # 3. Mark to market and let the kill switch look at it.
            equity = cash + position * close
            halted = risk.update_equity(equity)

            equity_curve.append(equity)
            position_track.append(position)
            weight_track.append(position * close / equity if equity > 0 else 0.0)

            # 4. Decide what we want to hold going into the next bar.
            raw_signal = float(signals.iloc[i])
            if blocked_sign and (raw_signal == 0 or (raw_signal > 0) != (blocked_sign > 0)):
                blocked_sign = 0  # signal reset; re-entry allowed again
            if halted:
                pending_target = 0.0
            elif blocked_sign and raw_signal != 0 and (raw_signal > 0) == (blocked_sign > 0):
                pending_target = 0.0
            else:
                pending_target = raw_signal

        equity_series = pd.Series(equity_curve, index=bars.index, name="equity")
        weights = pd.Series(weight_track, index=bars.index, name="weight")
        trades_df = pd.DataFrame(trades)

        return BacktestResult(
            equity=equity_series,
            positions=pd.Series(position_track, index=bars.index, name="position"),
            target_weights=weights,
            trades=trades_df,
            metrics=metrics_mod.compute(
                equity_series,
                trades_df,
                weights,
                periods_per_year=cfg.periods_per_year,
                risk_free_rate=cfg.risk_free_rate,
            ),
            strategy_name=self.strategy.name,
            symbol=symbol,
            halted=risk.halted,
            halt_reason=risk.halt_reason,
            _config=cfg,
        )
