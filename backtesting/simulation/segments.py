"""Immutable segment planning primitives for cached index simulations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class ImmutableSimulationSegment:
    """One cacheable interval within an effective-date holding period."""

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    effective_date: pd.Timestamp
    next_effective_date: pd.Timestamp | None
    kind: str
    target_checksum: str
    previous_target_checksum: str | None

    def __post_init__(self) -> None:
        for name in ("start_date", "end_date", "effective_date"):
            object.__setattr__(
                self,
                name,
                pd.Timestamp(getattr(self, name)).normalize(),
            )
        if self.next_effective_date is not None:
            object.__setattr__(
                self,
                "next_effective_date",
                pd.Timestamp(self.next_effective_date).normalize(),
            )
        if self.start_date > self.end_date:
            raise ValueError("simulation segment start must not be after its end")


def calendar_month_partitions(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
):
    """Yield closed calendar-month partitions over an inclusive date range."""

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        return
    cursor = start
    while cursor <= end:
        month_end = (cursor + pd.offsets.MonthEnd(0)).normalize()
        partition_end = min(month_end, end)
        yield cursor, partition_end
        cursor = partition_end + pd.Timedelta(days=1)


__all__ = ["ImmutableSimulationSegment", "calendar_month_partitions"]
