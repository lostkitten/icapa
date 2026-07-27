from dataclasses import dataclass, field
from typing import List

from icapa.data_sources.contracts import require_columns
from icapa.data_sources.registry import registry
from icapa.portfolio_construction.rules.data_loading.base import DataLoadingRule


@dataclass
class AddIdentifierFacts(DataLoadingRule):
    """Load provider-neutral reference fields such as ISIN or classifications."""

    fields: List[str] = field(default_factory=list)
    provider_name: str = ""
    provider_parameters: dict = field(default_factory=dict)
    include_excluded_instruments: bool = True
    command: str = "AddIdentifierFacts"

    def __post_init__(self):
        super().__post_init__()
        if not self.provider_name or not self.provider_name.strip():
            raise ValueError("provider_name is required for AddIdentifierFacts")
        if not self.fields:
            raise ValueError("fields must contain at least one reference-data field")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("fields must not contain duplicates")
        if "instrument_id" in self.fields:
            raise ValueError("instrument_id is a key and must not be requested as a field")

    def get_output_fact_names(self):
        return list(self.fields)

    def execute(self, data_context):
        provider = registry.resolve("load_reference_data", self.provider_name)
        current = data_context.get_dataframe(
            [],
            include_excluded_instruments=self.include_excluded_instruments,
        )
        reference = provider.load_reference_data(
            instrument_ids=tuple(current.index),
            reference_date=data_context.reference_date,
            fields=self.fields,
            **self.provider_parameters,
        )
        if reference is None:
            raise ValueError(f"provider {self.provider_name!r} returned no reference data")
        reference = reference.copy()
        require_columns(reference, ["instrument_id", *self.fields], "reference data")
        if reference.index.name != "instrument_id":
            reference = reference.set_index("instrument_id", verify_integrity=True)
        elif reference.index.duplicated().any():
            raise ValueError("reference data contains duplicate instrument_id values")
        if len(reference.index.difference(current.index)):
            raise ValueError("reference data contains instruments outside the universe")
        current = current.join(reference[self.fields], how="left")
        data_context.set_dataframe(current, columns=self.fields)
        return data_context
