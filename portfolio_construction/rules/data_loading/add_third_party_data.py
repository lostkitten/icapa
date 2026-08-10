"""Typed loading rule for non-canonical external datasets."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from icapa.data_sources.contracts import require_columns
from icapa.data_sources.providers.registry import registry
from icapa.data_sources.provenance import automatic_data_identity
from icapa.portfolio_construction.rules.data_loading.base import DataLoadingRule
from icapa.data_sources.contracts import ThirdPartyDataType


@dataclass
class AddThirdPartyData(DataLoadingRule):
    """Load an explicitly classified external dataset from a named provider.

    The data category, requested fields, and provider are mandatory so that
    external dependencies remain visible in code and configuration.
    """

    data_type: ThirdPartyDataType | str | None = None
    fields: list[str] = field(default_factory=list)
    provider_name: str = ""
    provider_parameters: dict = field(default_factory=dict)
    include_excluded_instruments: bool = True
    command: str = "AddThirdPartyData"

    def __post_init__(self):
        super().__post_init__()
        if self.data_type is None:
            raise ValueError("data_type is required for AddThirdPartyData")
        self.data_type = ThirdPartyDataType(self.data_type)
        if not self.provider_name or not self.provider_name.strip():
            raise ValueError("provider_name is required for AddThirdPartyData")
        if not self.fields:
            raise ValueError("fields must contain at least one requested field")
        if any(
            not isinstance(field_name, str) or not field_name.strip()
            for field_name in self.fields
        ):
            raise ValueError("fields must contain non-empty strings")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("fields must not contain duplicates")
        reserved = {"instrument_id", "reference_date", "effective_date", "business_date"}
        overlap = reserved.intersection(self.fields)
        if overlap:
            raise ValueError(
                f"fields must not replace canonical key/date columns: {sorted(overlap)}"
            )

    def get_output_field_names(self):
        return list(self.fields)

    def execute(self, data_context):
        provider = registry.resolve("load_third_party_data", self.provider_name)
        current = data_context.get_dataframe(
            [],
            include_excluded_instruments=self.include_excluded_instruments,
        )
        facts = provider.load_third_party_data(
            data_type=self.data_type.value,
            instrument_ids=tuple(current.index),
            fields=tuple(self.fields),
            reference_date=data_context.reference_date,
            parameters=dict(self.provider_parameters),
        )
        if facts is None:
            raise ValueError(
                f"provider {self.provider_name!r} returned no {self.data_type.value} data"
            )
        facts = facts.copy()
        require_columns(facts, ["instrument_id", *self.fields], self.data_type.value)
        if facts.index.name != "instrument_id":
            facts = facts.set_index("instrument_id", verify_integrity=True)
        elif facts.index.duplicated().any():
            raise ValueError(
                f"{self.data_type.value} contains duplicate instrument_id values"
            )
        if "reference_date" in facts.columns:
            source_dates = pd.to_datetime(facts["reference_date"])
            if (source_dates > pd.Timestamp(data_context.reference_date)).any():
                raise ValueError(
                    f"{self.data_type.value} contains reference_date after the review cutoff"
                )
        unexpected_ids = facts.index.difference(current.index)
        if len(unexpected_ids):
            raise ValueError(
                f"{self.data_type.value} contains instruments outside the universe"
            )
        data_context.provenance.record_provider_call(
            automatic_data_identity(
                provider_name=self.provider_name,
                provider=provider,
                capability="load_third_party_data",
                request={
                    "data_type": self.data_type.value,
                    "fields": tuple(self.fields),
                    "reference_date": data_context.reference_date,
                    **self.provider_parameters,
                },
                frame=facts,
                sort_by=["instrument_id"],
            )
        )
        current = current.join(facts[self.fields], how="left")
        data_context.set_dataframe(current, columns=self.fields)
        return data_context
