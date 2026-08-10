"""Ad-hoc CSV/Excel input rule."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from icapa.data_sources.providers.registry import get_provider
from icapa.portfolio_construction.rules.data_loading.base import DataLoadingRule


@dataclass
class ImportData(DataLoadingRule):
    """Read a CSV/Excel file through the registered file provider and merge it."""

    file_name: str = ""
    provider_name: str = "file"
    kwargs: dict = field(default_factory=dict)
    rename: dict = field(default_factory=dict)
    imported_keys: list = field(default_factory=list)
    import_columns: Optional[list] = None
    date_column: Optional[str] = None
    include_excluded_instruments: bool = False
    command: str = "ImportData"

    def get_output_field_names(self):
        return self.import_columns or []

    def _render_file_name(self, data_context) -> str:
        file_name = str(Path(self.file_name))
        values = {
            "effective_date": pd.Timestamp(data_context.effective_date).date(),
            "reference_date": pd.Timestamp(data_context.reference_date).date(),
            "universe_id": getattr(data_context, "universe_id", ""),
        }
        if getattr(data_context, "calendar", None) is not None:
            row = data_context.calendar.dates
            row = row[row.effective_date == data_context.effective_date].iloc[0]
            for name in ("previous_effective_date", "next_effective_date"):
                values[name] = pd.Timestamp(row[name]).date()
        return file_name.format(**values)

    def execute(self, data_context):
        provider = get_provider(self.provider_name)
        read = getattr(provider, "read", None)
        if not callable(read):
            raise TypeError(f"provider {self.provider_name!r} does not implement file reading")
        adhoc_df = read(self._render_file_name(data_context), **self.kwargs).rename(columns=self.rename)

        if "effective_date" in adhoc_df:
            adhoc_df["effective_date"] = pd.to_datetime(adhoc_df["effective_date"])
        if self.date_column:
            adhoc_df[self.date_column] = pd.to_datetime(adhoc_df[self.date_column])
            available = adhoc_df.loc[
                adhoc_df[self.date_column] <= data_context.reference_date,
                self.date_column,
            ]
            if available.empty:
                raise ValueError(f"no {self.date_column} on or before reference_date")
            adhoc_df = adhoc_df.loc[adhoc_df[self.date_column] == available.max()]

        current = data_context.get_dataframe(
            self.imported_keys,
            include_excluded_instruments=self.include_excluded_instruments,
        )
        if not current.empty and not self.imported_keys:
            raise ValueError("imported_keys must be supplied when merging into existing data")
        if current.empty and not self.imported_keys:
            merged = adhoc_df.set_index(self.keys)
        else:
            merged = current.reset_index().merge(
                adhoc_df,
                on=self.imported_keys,
                how="left",
            ).set_index(self.keys)

        if "excluded" in merged:
            merged["excluded"] = merged["excluded"].fillna(False)
        if self.import_columns:
            merged = merged[self.import_columns]
        data_context.set_dataframe(merged)

        if "excluded" in merged:
            weights = data_context.get_dataframe(
                ["index_weight", "excluded"], include_excluded_instruments=True
            )
            weights["index_weight"] = weights["index_weight"].where(~weights["excluded"], 0.0)
            total_weight = weights["index_weight"].sum()
            if total_weight <= 0:
                raise ValueError("imported exclusions removed all positive index weight")
            weights["index_weight"] /= total_weight
            data_context.set_dataframe(weights)
        return data_context
