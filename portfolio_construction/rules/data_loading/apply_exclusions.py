"""Provider-neutral constituent exclusions."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from icapa.data_sources.registry import registry
from icapa.portfolio_construction.rules.data_loading.base import DataLoadingRule


@dataclass
class ApplyExclusions(DataLoadingRule):
    """Exclude instruments by column values or by another index membership.

    ``index_id`` is a virtual filter.  When present, a provider implementing
    ``load_membership`` must be selected.  All other keys are ordinary
    canonical DataFrame columns such as ``country`` or ``industry``.
    """

    exclude: Optional[dict] = None
    only_include: Optional[dict] = None
    provider_name: str | None = None
    provider_parameters: dict = field(default_factory=dict)
    allow_null: bool = False
    command: str = "ApplyExclusions"

    def __post_init__(self):
        self.exclude = self._normalise_filters(self.exclude)
        self.only_include = self._normalise_filters(self.only_include)
        if (
            self.exclude.get("index_id") or self.only_include.get("index_id")
        ) and (not self.provider_name or not self.provider_name.strip()):
            raise ValueError(
                "provider_name is required when exclusions use index membership"
            )
        self.excluded_constituents: dict[str, set] = {}
        super().__post_init__()

    @staticmethod
    def _normalise_filters(filters):
        filters = filters or {}
        return {
            key: value if isinstance(value, (list, tuple, set)) else [value]
            for key, value in filters.items()
        }

    def _load_membership(self, index_id, data_context) -> set:
        provider = registry.resolve("load_membership", self.provider_name)
        membership = provider.load_membership(
            index_id=index_id,
            start_date=data_context.effective_date,
            end_date=data_context.effective_date,
            **self.provider_parameters,
        )
        if membership is None:
            if self.allow_null:
                return set()
            raise ValueError(f"provider returned no membership for {index_id}")
        if isinstance(membership, pd.DataFrame):
            if "instrument_id" in membership.columns:
                return set(membership["instrument_id"])
            return set(membership.index)
        return set(membership)

    def get_input_fact_names(self):
        columns = (set(self.exclude) | set(self.only_include)) - {"index_id"}
        return sorted(columns) + ["instrument_id", "excluded", "exclusion_reason", "index_weight"]

    def get_output_fact_names(self):
        return ["index_weight", "excluded", "exclusion_reason"]

    @staticmethod
    def _existing_reasons(value):
        if isinstance(value, list):
            return value
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        if isinstance(value, str):
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else [str(parsed)]
        return [str(value)]

    def execute(self, data_context):
        df = data_context.cons.copy()
        if "excluded" not in df:
            df["excluded"] = False
        if "exclusion_reason" not in df:
            df["exclusion_reason"] = [[] for _ in range(len(df))]

        data_context.exclusion_info = {
            "exclude": self.exclude,
            "only_include": self.only_include,
        }

        current_ids = set(df.index)
        for index_id in self.exclude.get("index_id", []):
            self.excluded_constituents[f"member of {index_id}"] = self._load_membership(index_id, data_context)

        included_ids: set = set()
        for index_id in self.only_include.get("index_id", []):
            included_ids |= self._load_membership(index_id, data_context)
        if self.only_include.get("index_id"):
            labels = ", ".join(map(str, self.only_include["index_id"]))
            self.excluded_constituents[f"not a member of {labels}"] = current_ids - included_ids

        for column, values in self.exclude.items():
            if column == "index_id":
                continue
            if column not in df:
                raise ValueError(f"{column!r} is not a valid exclusion column")
            self.excluded_constituents[column] = set(df.index[df[column].isin(values)])

        for column, values in self.only_include.items():
            if column == "index_id":
                continue
            if column not in df:
                raise ValueError(f"{column!r} is not a valid only_include column")
            self.excluded_constituents[f"not in {column}"] = set(df.index[~df[column].isin(values)])

        new_reasons = df.index.to_series().map(
            lambda instrument_id: [
                reason for reason, ids in self.excluded_constituents.items() if instrument_id in ids
            ]
        )
        df["exclusion_reason"] = [
            self._existing_reasons(old) + new
            for old, new in zip(df["exclusion_reason"], new_reasons)
        ]
        df["excluded"] = df["excluded"].fillna(False) | df["exclusion_reason"].map(bool)
        df.loc[df["excluded"], "index_weight"] = 0.0

        total_weight = df["index_weight"].sum()
        if total_weight <= 0:
            raise ValueError("exclusions removed all positive index weight")
        df["index_weight"] /= total_weight
        data_context.set_dataframe(df, columns=self.get_output_fact_names())
        return data_context

    def final_checks(self, data_context, final_weights: pd.Series) -> dict:
        excluded = data_context._df["excluded"].fillna(True)
        if set(excluded.index) != set(final_weights.index):
            raise ValueError("final index and starting universe have different instrument ids")
        df = pd.concat([excluded, final_weights], axis=1)
        df["is_constraint_met"] = ~((df["excluded"]) & (df["index_weight"] > 0))
        df["violation_no_tol"] = df["excluded"].astype(int) * df["index_weight"]
        return {self.command: df}
