"""Indian equity transaction cost calculation engine.

Implements statutory taxes, exchange fees, SEBI charges, GST, and broker commissions
for both delivery and intraday equity transactions according to Indian financial regulations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class CostBreakdown:
    """Detailed breakdown of transaction costs for an execution."""

    turnover: float
    stt: float
    exchange_charges: float
    sebi_charges: float
    stamp_duty: float
    gst: float
    brokerage: float

    @property
    def total(self) -> float:
        """Total transaction costs including taxes and brokerage."""
        return (
            self.stt
            + self.exchange_charges
            + self.sebi_charges
            + self.stamp_duty
            + self.gst
            + self.brokerage
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "turnover": round(self.turnover, 2),
            "stt": round(self.stt, 2),
            "exchange_charges": round(self.exchange_charges, 2),
            "sebi_charges": round(self.sebi_charges, 2),
            "stamp_duty": round(self.stamp_duty, 2),
            "gst": round(self.gst, 2),
            "brokerage": round(self.brokerage, 2),
            "total_costs": round(self.total, 2),
        }


class TransactionCostModel(ABC):
    """Abstract base class for transaction cost models."""

    @abstractmethod
    def calculate(
        self,
        side: str,
        shares: float,
        price: float,
        is_intraday: bool = False,
    ) -> CostBreakdown:
        """Calculate the cost breakdown for a trade fill."""


class IndianEquityCostModel(TransactionCostModel):
    """Indian statutory and exchange cost model for NSE equities.

    Default rates reflect current SEBI & Indian Finance Act mandates:
    - Delivery:
        - STT: 0.1% on Buy & Sell
        - Exchange charges (NSE): 0.00345%
        - SEBI charges: ₹10 per crore (0.0001%)
        - Stamp duty: 0.015% on Buy only
        - GST: 18% on (Brokerage + Exchange fees + SEBI charges)
        - Brokerage: ₹0 for delivery (discount broker model) or flat/percent
    - Intraday:
        - STT: 0.025% on Sell only
        - Stamp duty: 0.003% on Buy only
        - Brokerage: ₹20 or 0.03% whichever is lower
    """

    def __init__(
        self,
        brokerage_per_order_delivery: float = 0.0,
        brokerage_per_order_intraday: float = 20.0,
        exchange_rate: float = 0.0000345,
        sebi_rate: float = 0.0000010,
        gst_rate: float = 0.18,
    ):
        self.brokerage_del = brokerage_per_order_delivery
        self.brokerage_intra = brokerage_per_order_intraday
        self.exchange_rate = exchange_rate
        self.sebi_rate = sebi_rate
        self.gst_rate = gst_rate

    def calculate(
        self,
        side: str,
        shares: float,
        price: float,
        is_intraday: bool = False,
    ) -> CostBreakdown:
        side_upper = side.upper()
        turnover = abs(shares) * price

        if turnover <= 0:
            return CostBreakdown(
                turnover=0.0,
                stt=0.0,
                exchange_charges=0.0,
                sebi_charges=0.0,
                stamp_duty=0.0,
                gst=0.0,
                brokerage=0.0,
            )

        exchange_charges = turnover * self.exchange_rate
        sebi_charges = turnover * self.sebi_rate

        if is_intraday:
            # Intraday
            stt = turnover * 0.00025 if side_upper in ("SELL", "SHORT") else 0.0
            stamp_duty = turnover * 0.00003 if side_upper in ("BUY", "LONG") else 0.0
            # Brokerage min(₹20, 0.03%)
            brokerage = min(self.brokerage_intra, turnover * 0.0003)
        else:
            # Delivery
            stt = turnover * 0.0010  # 0.1% on both buy and sell
            stamp_duty = turnover * 0.00015 if side_upper in ("BUY", "LONG") else 0.0
            brokerage = self.brokerage_del

        # GST applies to: Brokerage + Exchange fees + SEBI fees
        gst = (brokerage + exchange_charges + sebi_charges) * self.gst_rate

        return CostBreakdown(
            turnover=turnover,
            stt=stt,
            exchange_charges=exchange_charges,
            sebi_charges=sebi_charges,
            stamp_duty=stamp_duty,
            gst=gst,
            brokerage=brokerage,
        )


class FlatCostModel(TransactionCostModel):
    """Simple flat commission model (for US equities / baseline testing)."""

    def __init__(self, commission_per_share: float = 0.005):
        self.commission_per_share = commission_per_share

    def calculate(
        self,
        side: str,
        shares: float,
        price: float,
        is_intraday: bool = False,
    ) -> CostBreakdown:
        turnover = abs(shares) * price
        comm = abs(shares) * self.commission_per_share
        return CostBreakdown(
            turnover=turnover,
            stt=0.0,
            exchange_charges=0.0,
            sebi_charges=0.0,
            stamp_duty=0.0,
            gst=0.0,
            brokerage=comm,
        )
