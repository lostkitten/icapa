"""Provider-neutral review calendars."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Union

import pandas as pd
from icapa.data_sources.contracts import validate_review_schedule
from icapa.data_sources.registry import registry


@dataclass
class AbstractCalendar:
    dates: pd.DataFrame = field(default_factory=pd.DataFrame)

    def concat(self, other):
        combined = pd.concat([self.dates, other.dates], ignore_index=True)
        conflicting = combined.groupby("effective_date")["reference_date"].nunique()
        if (conflicting > 1).any():
            raise ValueError("calendars contain conflicting reference dates")
        combined = combined.drop_duplicates("effective_date")
        return to_calendar(combined)


@dataclass
class Calendar(AbstractCalendar):
    """Review schedule loaded through an explicitly selected provider.

    Use :meth:`from_dates` for CSV/Excel/manual schedules.  Otherwise set
    ``provider_name`` and ``calendar_id``; no database calendar is selected by
    default.
    """

    start_date: Optional[Union[datetime, str]] = None
    end_date: Optional[Union[datetime, str]] = None
    calendar_id: str = ""
    provider_name: str | None = None
    provider_parameters: dict = field(default_factory=dict)
    command: str = "Calendar"

    def __post_init__(self):
        if not self.dates.empty:
            self.dates = self._normalise(self.dates)
            if self.start_date is None:
                self.start_date = self.dates["effective_date"].min()
            if self.end_date is None:
                self.end_date = self.dates["effective_date"].max()
            return
        if not self.provider_name:
            raise ValueError("Calendar requires provider_name or Calendar.from_dates(...)")
        if not self.calendar_id:
            raise ValueError("calendar_id must be supplied for a provider calendar")
        if self.start_date is None or self.end_date is None:
            raise ValueError("start_date and end_date are required for a provider calendar")
        self.start_date = pd.Timestamp(self.start_date).normalize()
        self.end_date = pd.Timestamp(self.end_date).normalize()
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        provider = registry.resolve("load_review_schedule", self.provider_name)
        schedule = provider.load_review_schedule(
            calendar_id=self.calendar_id,
            start_date=self.start_date,
            end_date=self.end_date,
            **self.provider_parameters,
        )
        self.dates = self._normalise(schedule)
        if (
            self.dates["effective_date"].min() < self.start_date
            or self.dates["effective_date"].max() > self.end_date
        ):
            raise ValueError("provider calendar falls outside the requested date range")

    @staticmethod
    def _normalise(dates: pd.DataFrame) -> pd.DataFrame:
        result = validate_review_schedule(dates)
        result["previous_effective_date"] = result["effective_date"].shift(1)
        result["next_effective_date"] = result["effective_date"].shift(-1)
        result["previous_reference_date"] = result["reference_date"].shift(1)
        result["next_reference_date"] = result["reference_date"].shift(-1)
        return result

    @classmethod
    def from_dates(cls, dates):
        return cls(dates=pd.DataFrame(dates))


def to_calendar(dates):
    if isinstance(dates, Calendar):
        return dates
    if isinstance(dates, str):
        raise TypeError("string calendars are not supported; provide explicit date rows")
    return Calendar.from_dates(dates)
