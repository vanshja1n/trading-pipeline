"""Live/paper trading loop.

Same strategy and risk objects the backtester uses -- only the broker and the
data source change. That symmetry is the whole point: if the live loop needed
its own copy of the logic, the backtest would stop meaning anything.

Behavior worth knowing:
* Acts once per newly *closed* bar, never mid-bar, so it matches the backtest.
* Persists kill-switch state to disk, so restarting the process does not
  quietly re-arm a bot that just blew through its drawdown limit.
* Refuses to place the first order until it has full indicator warmup.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..broker.base import Broker
from ..config import LiveConfig, RiskConfig
from ..data.feed import AlpacaFeed
from ..risk.manager import RiskManager
from ..strategy import indicators as ind
from ..strategy.base import Strategy

log = logging.getLogger(__name__)

STATE_PATH = Path(__file__).resolve().parents[2] / "state" / "live_state.json"
LOCK_PATH = Path(__file__).resolve().parents[2] / "state" / "live.lock"


class AlreadyRunning(RuntimeError):
    """Another trader instance holds the lock."""


class InstanceLock:
    """PID lockfile: exactly one trader per account, ever.

    Two instances sharing an account fight each other -- one opens a position,
    the other sees a position its own signal doesn't want and closes it. They
    also clobber each other's kill-switch state. Cheaper to refuse to start.
    """

    def __init__(self, path: Path = LOCK_PATH):
        self.path = path
        self.acquired = False

    def _holder(self) -> int | None:
        """PID of a live process holding the lock, or None if free/stale."""
        try:
            pid = int(self.path.read_text().strip())
        except (OSError, ValueError):
            return None
        try:
            os.kill(pid, 0)  # signal 0 just tests existence
        except ProcessLookupError:
            return None      # died without cleaning up
        except PermissionError:
            return pid       # alive, owned by another user
        return pid

    def acquire(self) -> None:
        holder = self._holder()
        if holder is not None and holder != os.getpid():
            raise AlreadyRunning(
                f"another trader is already running (PID {holder}).\n"
                f"  Stop it first:  kill {holder}\n"
                f"  Running two bots on one account makes them trade against "
                f"each other.\n  If you're sure that process is gone, delete {self.path}"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(str(os.getpid()))
        self.acquired = True

    def release(self) -> None:
        # Only remove our own lock -- never yank one out from under another process.
        if self.acquired and self._holder() == os.getpid():
            self.path.unlink(missing_ok=True)
        self.acquired = False

    def __enter__(self) -> InstanceLock:
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class LiveTrader:
    def __init__(
        self,
        strategy: Strategy,
        broker: Broker,
        feed: AlpacaFeed,
        config: LiveConfig,
        risk_config: RiskConfig | None = None,
    ):
        self.strategy = strategy
        self.broker = broker
        self.feed = feed
        self.config = config
        account = broker.get_account()
        self.risk = RiskManager(risk_config or RiskConfig(), account.equity)
        self._last_bar: dict[str, pd.Timestamp] = {}
        self._session_date: str | None = None
        self._load_state()

    # ------------------------------------------------------------------ state

    def _load_state(self) -> None:
        if not STATE_PATH.exists():
            return
        try:
            state = json.loads(STATE_PATH.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("could not read state file (%s); starting fresh", exc)
            return
        self.risk.peak_equity = state.get("peak_equity", self.risk.peak_equity)
        self.risk.halted = state.get("halted", False)
        self.risk.halt_reason = state.get("halt_reason", "")
        self._last_bar = {k: pd.Timestamp(v) for k, v in state.get("last_bar", {}).items()}
        if self.risk.halted:
            log.warning("resuming in HALTED state: %s", self.risk.halt_reason)
            log.warning("clear %s to re-arm after you've reviewed what happened", STATE_PATH)

    def _save_state(self) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(
            json.dumps(
                {
                    "peak_equity": self.risk.peak_equity,
                    "halted": self.risk.halted,
                    "halt_reason": self.risk.halt_reason,
                    "last_bar": {k: str(v) for k, v in self._last_bar.items()},
                    "updated": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
        )

    # ------------------------------------------------------------------- loop

    def run(self, max_iterations: int | None = None) -> None:
        cfg = self.config
        mode = "DRY RUN" if cfg.dry_run else ("PAPER" if cfg.paper else "*** LIVE MONEY ***")
        log.info("starting %s | %s | symbols=%s | %s bars", self.strategy.name, mode,
                 cfg.symbols, cfg.timeframe)

        # A backgrounded process ignores SIGINT, so `kill <pid>` (SIGTERM) is how
        # you actually stop one. Route it through KeyboardInterrupt so either
        # signal saves state and releases the lock on the way out.
        def _on_sigterm(signum, frame):
            raise KeyboardInterrupt

        try:
            signal.signal(signal.SIGTERM, _on_sigterm)
        except ValueError:
            pass  # not the main thread; caller handles shutdown

        iterations = 0
        with InstanceLock():
            try:
                while max_iterations is None or iterations < max_iterations:
                    iterations += 1
                    try:
                        self.step()
                    except Exception:
                        # A transient API error shouldn't kill an unattended bot.
                        log.exception("error in trading loop; retrying next cycle")
                    # Ctrl+C usually lands here, during the sleep. The outer try
                    # catches it so you get a clean exit, not a traceback.
                    time.sleep(cfg.poll_seconds)
            except KeyboardInterrupt:
                log.info("interrupted; leaving positions untouched and exiting")
            finally:
                self._save_state()

    def step(self) -> None:
        """One decision cycle across all symbols."""
        if not self.broker.is_market_open():
            log.debug("market closed; idling")
            return

        account = self.broker.get_account()

        today = datetime.now(timezone.utc).date().isoformat()
        if self._session_date != today:
            self.risk.start_session(account.equity)
            self._session_date = today
            log.info("new session %s | equity $%.2f", today, account.equity)

        if self.risk.update_equity(account.equity):
            self._flatten_all("kill switch: " + self.risk.halt_reason)
            self._save_state()
            return

        for symbol in self.config.symbols:
            try:
                self._handle_symbol(symbol, account.equity)
            except Exception:
                log.exception("failed handling %s", symbol)

        self._save_state()

    def _handle_symbol(self, symbol: str, equity: float) -> None:
        bars = self.feed.latest_bars(symbol, self.config.lookback_bars, self.config.timeframe)
        if len(bars) <= self.strategy.warmup:
            log.warning("%s: only %d bars, need > %d for warmup -- skipping",
                        symbol, len(bars), self.strategy.warmup)
            return

        last_ts = bars.index[-1]
        if self._last_bar.get(symbol) == last_ts:
            return  # already acted on this bar
        self._last_bar[symbol] = last_ts

        signal = float(self.strategy(bars).iloc[-1])
        vol_series = ind.realized_volatility(bars["close"], 20)
        vol = float(vol_series.iloc[-1]) if pd.notna(vol_series.iloc[-1]) else None

        weight = self.risk.target_weight(signal, vol)
        price = float(bars["close"].iloc[-1])
        # Split equity evenly across symbols so N symbols don't each take a full position.
        per_symbol_equity = equity / max(len(self.config.symbols), 1)
        desired = self.risk.shares_for_weight(weight, per_symbol_equity, price)
        current = self.broker.get_position(symbol)
        delta = desired - current

        log.info(
            "%s @ %s | px=%.2f signal=%+.2f weight=%+.2f | have %d want %d",
            symbol, last_ts, price, signal, weight, current, desired,
        )

        if delta == 0:
            return

        # Same gate the backtester applies. Equity drifts every bar, so the
        # rounded share target flips by one constantly; without this the bot
        # churns one-share round trips that the backtest never modeled.
        current_weight = current * price / per_symbol_equity if per_symbol_equity > 0 else 0.0
        moving_to_flat = desired == 0 and current != 0
        drift = abs(weight - current_weight)
        if not moving_to_flat and drift < self.config.rebalance_threshold:
            log.info(
                "%s: holding %d shares -- drift %.3f below rebalance threshold %.3f",
                symbol, current, drift, self.config.rebalance_threshold,
            )
            return

        self.broker.submit_order(symbol, abs(delta), "buy" if delta > 0 else "sell")

    def _flatten_all(self, reason: str) -> None:
        log.warning("flattening all positions -- %s", reason)
        self.broker.cancel_all_orders()
        for symbol in self.config.symbols:
            self.broker.close_position(symbol)
