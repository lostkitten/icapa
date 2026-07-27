"""Small in-memory data context used by portfolio construction rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from typing import Iterable

import pandas as pd


@dataclass
class DataContext:
    """Hold point-in-time constituents and optional daily market observations."""

    reference_date: object | None = None
    effective_date: object | None = None
    index_id: str = ""
    universe_id: str = ""
    calendar: object | None = None
    provider_name: str | None = None
    provider_parameters: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    daily: pd.DataFrame | None = None
    _df: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def __post_init__(self) -> None:
        if self.reference_date is not None:
            self.reference_date = pd.Timestamp(self.reference_date).normalize()
        if self.effective_date is not None:
            self.effective_date = pd.Timestamp(self.effective_date).normalize()

    @property
    def cons(self) -> pd.DataFrame:
        """Return the constituent frame."""

        return self._df

    def get_dataframe(
        self,
        columns: Iterable[str] | str | None = None,
        include_excluded_instruments: bool = False,
    ) -> pd.DataFrame:
        """Return a defensive copy of requested constituent columns."""

        if columns is None or columns == [] or columns == ():
            result = self._df.copy()
        else:
            requested = [columns] if isinstance(columns, str) else list(columns)
            missing = set(requested) - set(self._df.columns)
            if missing:
                raise KeyError(f"constituent data is missing columns: {sorted(missing)}")
            result = self._df.loc[:, requested].copy()

        if not include_excluded_instruments and "excluded" in result.columns:
            result = result.loc[~result["excluded"].fillna(False).astype(bool)]
        elif not include_excluded_instruments and "excluded" in self._df.columns:
            result = result.loc[~self._df["excluded"].fillna(False).astype(bool)]
        return result

    def set_dataframe(
        self,
        frame: pd.DataFrame,
        columns: Iterable[str] | str | None = None,
    ) -> None:
        """Merge constituent data or store instrument-by-business-date data."""

        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")

        if isinstance(frame.index, pd.MultiIndex) and "business_date" in frame.index.names:
            self.daily = frame.copy()
            return

        incoming = frame.copy()
        if incoming.index.name != "instrument_id":
            if "instrument_id" not in incoming.columns:
                raise ValueError("constituent data must use instrument_id as its index or column")
            incoming = incoming.set_index("instrument_id", verify_integrity=True)

        if incoming.index.has_duplicates:
            raise ValueError("constituent data contains duplicate instrument_id values")

        if columns is None:
            selected = list(incoming.columns)
        elif isinstance(columns, str):
            selected = [columns]
        else:
            selected = list(columns)
        selected = [column for column in selected if column != "instrument_id"]

        missing = set(selected) - set(incoming.columns)
        if missing:
            raise KeyError(f"incoming data is missing columns: {sorted(missing)}")

        if self._df.empty:
            self._df = incoming.loc[:, selected].copy()
            self._df.index.name = "instrument_id"
            return

        unknown_ids = incoming.index.difference(self._df.index)
        if len(unknown_ids):
            raise ValueError(f"incoming data contains unknown instrument_id values: {list(unknown_ids)}")

        for column in selected:
            self._df.loc[incoming.index, column] = incoming[column]

    def copy(self) -> "DataContext":
        """Return a deep-enough copy for an independent review calculation."""

        duplicate = DataContext(
            reference_date=self.reference_date,
            effective_date=self.effective_date,
            index_id=self.index_id,
            universe_id=self.universe_id,
            calendar=self.calendar,
            provider_name=self.provider_name,
            provider_parameters=dict(self.provider_parameters),
            diagnostics=deepcopy(self.diagnostics),
        )
        duplicate._df = self._df.copy(deep=True)
        duplicate.daily = None if self.daily is None else self.daily.copy(deep=True)
        return duplicate


DataContainer = DataContext
