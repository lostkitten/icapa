"""Provider-neutral review calendars."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from pandas.tseries.offsets import MonthEnd, QuarterEnd, Week, YearEnd

from icapa.data_sources.contracts import validate_review_schedule
from icapa.data_sources.providers.registry import registry

from .frequency import RebalanceFrequency


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

    start_date: datetime | str | None = None
    end_date: datetime | str | None = None
    calendar_id: str = ""
    provider_name: str | None = None
    provider_parameters: dict = field(default_factory=dict)
    command: str = "Calendar"

    def __post_init__(self):
        if not self.dates.empty:
            self.dates = self._normalize(self.dates)
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
        self.dates = self._normalize(schedule)
        if (
            self.dates["effective_date"].min() < self.start_date
            or self.dates["effective_date"].max() > self.end_date
        ):
            raise ValueError("provider calendar falls outside the requested date range")

    @staticmethod
    def _normalize(dates: pd.DataFrame) -> pd.DataFrame:
        result = validate_review_schedule(dates)
        result["previous_effective_date"] = result["effective_date"].shift(1)
        result["next_effective_date"] = result["effective_date"].shift(-1)
        result["previous_reference_date"] = result["reference_date"].shift(1)
        result["next_reference_date"] = result["reference_date"].shift(-1)
        return result

    @classmethod
    def from_dates(cls, dates):
        return cls(dates=pd.DataFrame(dates))

    @classmethod
    def from_frequency(
        cls,
        *,
        start_date,
        end_date,
        frequency: RebalanceFrequency | str,
        reference_lag_business_days: int = 0,
        business_days=None,
    ):
        """Explicitly generate a period-end review schedule.

        Existing effective-date schedules remain authoritative. This helper is
        used only when called directly and never rewrites provider or manual
        dates. When ``business_days`` is omitted, a Monday-Friday calendar is
        used; production deployments should pass their explicit calendar.
        """

        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        if pd.isna(start) or pd.isna(end):
            raise ValueError("start_date and end_date must not be null")
        if start > end:
            raise ValueError("start_date must not be after end_date")
        if (
            not isinstance(reference_lag_business_days, int)
            or reference_lag_business_days < 0
        ):
            raise ValueError(
                "reference_lag_business_days must be a non-negative integer"
            )
        selected_frequency = RebalanceFrequency(frequency)
        if selected_frequency is RebalanceFrequency.CUSTOM:
            raise ValueError(
                "CUSTOM schedules must be supplied with Calendar.from_dates(...)"
            )
        if business_days is None:
            padding = 14 + 3 * reference_lag_business_days
            days = pd.bdate_range(start - pd.Timedelta(days=padding), end)
        else:
            days = pd.DatetimeIndex(pd.to_datetime(list(business_days))).normalize()
            days = days.sort_values().drop_duplicates()
        if days.empty:
            raise ValueError("business_days must contain at least one date")

        anchors = _frequency_anchors(start, end, selected_frequency)
        rows: list[dict[str, pd.Timestamp]] = []
        for anchor in anchors:
            position = int(days.searchsorted(anchor, side="right")) - 1
            if position < 0:
                raise ValueError(
                    f"business_days has no date on or before anchor {anchor.date()}"
                )
            effective = days[position]
            if effective < start or effective > end:
                continue
            reference_position = position - reference_lag_business_days
            if reference_position < 0:
                raise ValueError(
                    "business_days does not contain enough history for the "
                    "reference-date lag"
                )
            rows.append(
                {
                    "reference_date": days[reference_position],
                    "effective_date": effective,
                }
            )
        if not rows:
            raise ValueError("frequency produced no effective dates in the range")
        return cls.from_dates(pd.DataFrame.from_records(rows).drop_duplicates())

    def frequency_diagnostics(
        self,
        frequency: RebalanceFrequency | str,
    ) -> tuple[dict[str, object], ...]:
        """Describe additional reviews and missing expected periods.

        The supplied effective-date schedule remains authoritative. These
        diagnostics identify likely configuration mistakes without treating
        holiday adjustments or intentionally missing periods as new dates.
        """

        selected = RebalanceFrequency(frequency)
        if selected is RebalanceFrequency.CUSTOM:
            return ()
        period_keys = self.dates["effective_date"].map(
            lambda value: _period_key(pd.Timestamp(value), selected)
        )
        records: list[dict[str, object]] = []
        duplicates = sorted(set(period_keys[period_keys.duplicated(False)]))
        for period in duplicates:
            positions = period_keys[period_keys == period].index
            records.append(
                {
                    "code": "additional_review_in_period",
                    "severity": "warning",
                    "frequency": selected.value,
                    "period": period,
                    "effective_dates": tuple(
                        pd.Timestamp(value).normalize()
                        for value in self.dates.loc[
                            positions,
                            "effective_date",
                        ]
                    ),
                }
            )
        unique_dates = (
            pd.DatetimeIndex(self.dates["effective_date"])
            .normalize()
            .sort_values()
        )
        ordinals = [
            _period_ordinal(value, selected)
            for value in unique_dates
        ]
        for previous_date, current_date, previous, current in zip(
            unique_dates[:-1],
            unique_dates[1:],
            ordinals[:-1],
            ordinals[1:],
        ):
            missing = current - previous - 1
            if missing > 0:
                records.append(
                    {
                        "code": "missing_rebalance_periods",
                        "severity": "warning",
                        "frequency": selected.value,
                        "previous_effective_date": pd.Timestamp(
                            previous_date
                        ).normalize(),
                        "next_effective_date": pd.Timestamp(
                            current_date
                        ).normalize(),
                        "missing_period_count": int(missing),
                    }
                )
        return tuple(records)

    def validate_frequency(
        self,
        frequency: RebalanceFrequency | str,
        *,
        allow_additional_reviews: bool = False,
        allow_gaps: bool = True,
    ) -> None:
        """Validate periodic reasonableness without rewriting the schedule."""

        selected = RebalanceFrequency(frequency)
        diagnostics = self.frequency_diagnostics(selected)
        additional = [
            item
            for item in diagnostics
            if item["code"] == "additional_review_in_period"
        ]
        if additional and not allow_additional_reviews:
            duplicates = [str(item["period"]) for item in additional]
            raise ValueError(
                f"review schedule has multiple effective dates in {selected.value} "
                f"periods: {duplicates}"
            )
        gaps = [
            item
            for item in diagnostics
            if item["code"] == "missing_rebalance_periods"
        ]
        if gaps and not allow_gaps:
            raise ValueError(
                f"review schedule has missing {selected.value} periods"
            )


def to_calendar(dates):
    if isinstance(dates, Calendar):
        return dates
    if isinstance(dates, str):
        raise TypeError("string calendars are not supported; provide explicit date rows")
    return Calendar.from_dates(dates)


def _frequency_anchors(
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: RebalanceFrequency,
) -> pd.DatetimeIndex:
    if frequency is RebalanceFrequency.CUSTOM:
        raise ValueError("CUSTOM schedules do not have generated frequency anchors")
    if frequency is RebalanceFrequency.WEEKLY:
        offset = Week(weekday=4)
        first = start + offset
        if start.weekday() == 4:
            first = start
        return pd.date_range(first, end, freq=offset)
    if frequency is RebalanceFrequency.MONTHLY:
        return pd.date_range(start, end, freq=MonthEnd())
    if frequency is RebalanceFrequency.QUARTERLY:
        return pd.date_range(start, end, freq=QuarterEnd())
    if frequency is RebalanceFrequency.ANNUAL:
        return pd.date_range(start, end, freq=YearEnd())
    semi_annual = pd.date_range(start, end, freq=QuarterEnd())
    return semi_annual[semi_annual.month.isin((6, 12))]


def _period_key(
    value: pd.Timestamp,
    frequency: RebalanceFrequency,
) -> str:
    if frequency is RebalanceFrequency.WEEKLY:
        year, week, _ = value.isocalendar()
        return f"{year}-W{week:02d}"
    if frequency is RebalanceFrequency.MONTHLY:
        return value.strftime("%Y-%m")
    if frequency is RebalanceFrequency.QUARTERLY:
        return f"{value.year}-Q{value.quarter}"
    if frequency is RebalanceFrequency.SEMI_ANNUAL:
        return f"{value.year}-H{1 if value.month <= 6 else 2}"
    return str(value.year)


def _period_ordinal(
    value: pd.Timestamp,
    frequency: RebalanceFrequency,
) -> int:
    if frequency is RebalanceFrequency.WEEKLY:
        monday = value - pd.Timedelta(days=value.weekday())
        return int(monday.toordinal() // 7)
    if frequency is RebalanceFrequency.MONTHLY:
        return value.year * 12 + value.month - 1
    if frequency is RebalanceFrequency.QUARTERLY:
        return value.year * 4 + value.quarter - 1
    if frequency is RebalanceFrequency.SEMI_ANNUAL:
        return value.year * 2 + (0 if value.month <= 6 else 1)
    if frequency is RebalanceFrequency.ANNUAL:
        return value.year
    raise ValueError("CUSTOM schedules do not have period ordinals")
