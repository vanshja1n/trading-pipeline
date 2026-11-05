"""Strategy interface.

A strategy answers exactly one question: given bars up to and including now,
what fraction of my allowed exposure do I want in this symbol?

    +1.0 = fully long,  0.0 = flat,  -1.0 = fully short.

It does NOT decide dollar amounts, share counts, or when to stop trading.
That is the risk manager's job (autotrade/risk/manager.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


class Strategy(ABC):
    """Base class for all strategies."""

    #: Bars of history needed before signals are meaningful.
    warmup: int = 0

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        """Return target exposure in [-1, 1] for each bar.

        The value at index ``t`` is the position you want to hold *based on
        information available at the close of bar t*. The engine executes it at
        the open of bar t+1, so never use ``.shift(-n)`` or any forward-looking
        transform in here.
        """

    def validate(self, bars: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in bars.columns]
        if missing:
            raise ValueError(f"{self.name}: bars missing columns {missing}")
        if len(bars) <= self.warmup:
            raise ValueError(
                f"{self.name}: needs more than {self.warmup} bars, got {len(bars)}"
            )

    def __call__(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        signals = self.generate_signals(bars)
        return signals.reindex(bars.index).fillna(0.0).clip(-1.0, 1.0).rename("target")
