from dataclasses import dataclass, field
from typing import List

from icapa.data_sources.registry import registry
from icapa.portfolio_construction.rules.data_loading.base import DataLoadingRule


@dataclass
class AddIndexMemberships(DataLoadingRule):
    """Add membership flags supplied by a configured MembershipProvider."""

    indices: List[str] = field(default_factory=list)
    provider_name: str = ""
    provider_parameters: dict = field(default_factory=dict)
    command: str = "AddIndexMemberships"

    def __post_init__(self):
        super().__post_init__()
        if not self.provider_name or not self.provider_name.strip():
            raise ValueError("provider_name is required for AddIndexMemberships")
        if not self.indices:
            raise ValueError("indices must contain at least one membership identifier")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("indices must not contain duplicates")

    def get_output_fact_names(self):
        return list(self.indices)

    def execute(self, data_context):
        provider = registry.resolve("load_membership", self.provider_name)
        result = data_context.get_dataframe([], include_excluded_instruments=True)
        for index_id in self.indices:
            membership = provider.load_membership(
                index_id=index_id,
                start_date=data_context.effective_date,
                end_date=data_context.effective_date,
                **self.provider_parameters,
            )
            if membership is None:
                raise ValueError(f"provider returned no membership for {index_id}")
            required = {"instrument_id", "is_member"}
            missing = required - set(membership.columns)
            if missing:
                raise ValueError(f"membership data is missing columns: {sorted(missing)}")
            if membership["instrument_id"].duplicated().any():
                raise ValueError(
                    f"membership data contains duplicate instrument_id values for {index_id}"
                )
            flags = membership.set_index("instrument_id")["is_member"]
            result[index_id] = result.index.to_series().map(flags).fillna(False).astype(bool)
        data_context.set_dataframe(result, columns=self.get_output_fact_names())
        return data_context
