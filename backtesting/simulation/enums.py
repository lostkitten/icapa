"""Enumerations that define daily index-simulation behavior."""

from __future__ import annotations

from enum import StrEnum


class _SimulationEnum(StrEnum):
    """String enum with strict, explicit parsing for configuration files."""

    @classmethod
    def from_str(cls, value: str):
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"invalid {cls.__name__} value: {value!r}") from exc


class DividendTreatment(_SimulationEnum):
    """Select one of the two explicit dividend calculation variants."""

    STANDARD = "standard"
    ALTERNATIVE = "alternative"


class WeightDrift(_SimulationEnum):
    """Select the v1 constituent-weight drift compatibility behavior."""

    PRICE_RETURN = "price_return"
    MARKET_CAP = "market_cap"


class RebalanceTiming(_SimulationEnum):
    """Select how a non-business effective date is applied."""

    NEXT_BUSINESS_DAY = "next_business_day"
    EXACT_DATE = "exact_date"


class RebalancePhase(_SimulationEnum):
    """Select whether a scheduled rebalance occurs before or after daily return."""

    OPEN = "before_return"
    CLOSE = "after_return"
    BEFORE_RETURN = OPEN
    AFTER_RETURN = CLOSE


class WeightSnapshotMode(_SimulationEnum):
    """Select which constituent-weight snapshots are materialized in results."""

    NONE = "none"
    REBALANCE = "rebalance_dates"
    DAILY = "daily"
    REBALANCE_DATES = REBALANCE


__all__ = [
    "DividendTreatment",
    "RebalancePhase",
    "RebalanceTiming",
    "WeightDrift",
    "WeightSnapshotMode",
]
