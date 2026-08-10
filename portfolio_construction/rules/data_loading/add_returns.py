from dataclasses import dataclass, field

import pandas as pd

from icapa.data_sources.contracts import validate_daily_market_data
from icapa.data_sources.providers.registry import registry
from icapa.data_sources.provenance import automatic_data_identity
from icapa.portfolio_construction.rules.data_loading.base import DataLoadingRule


@dataclass
class AddReturns(DataLoadingRule):
    """Load canonical daily market data from an explicitly selected provider."""

    provider_name: str = ""
    start_date: str | None = None
    end_date: str | None = None
    provider_parameters: dict = field(default_factory=dict)
    enforce_reference_cutoff: bool = True
    command: str = "AddReturns"

    def __post_init__(self):
        super().__post_init__()
        if not self.provider_name or not self.provider_name.strip():
            raise ValueError("provider_name is required for AddReturns")

    def get_output_field_names(self):
        return [
            "price_return",
            "gross_dividend",
            "net_dividend",
            "market_cap",
            "gross_total_return",
            "net_total_return",
            "total_return_gross",
            "total_return_net",
        ]

    def execute(self, data_context):
        provider = registry.resolve("load_daily_market_data", self.provider_name)
        instruments = data_context.constituents.index
        start_date = pd.Timestamp(self.start_date or data_context.reference_date).normalize()
        end_date = pd.Timestamp(self.end_date or data_context.reference_date).normalize()
        reference_date = pd.Timestamp(data_context.reference_date).normalize()
        if start_date > end_date:
            raise ValueError("start_date must not be after end_date")
        if self.enforce_reference_cutoff and end_date > reference_date:
            raise ValueError("end_date must not be after reference_date")
        daily = provider.load_daily_market_data(
            instrument_ids=tuple(instruments),
            start_date=start_date,
            end_date=end_date,
            **self.provider_parameters,
        )
        daily = validate_daily_market_data(
            daily,
            reference_date=reference_date if self.enforce_reference_cutoff else None,
        )
        if (daily["business_date"] < start_date).any() or (
            daily["business_date"] > end_date
        ).any():
            raise ValueError("daily market data falls outside the requested date range")
        unexpected_ids = set(daily["instrument_id"]) - set(instruments)
        if unexpected_ids:
            raise ValueError("daily market data contains instruments outside the universe")
        daily = daily.copy()
        daily["gross_total_return"] = (
            daily["price_return"] + daily["gross_dividend"]
        )
        daily["net_total_return"] = (
            daily["price_return"] + daily["net_dividend"]
        )
        daily["total_return_gross"] = daily["gross_total_return"]
        daily["total_return_net"] = daily["net_total_return"]
        data_context.provenance.record_provider_call(
            automatic_data_identity(
                provider_name=self.provider_name,
                provider=provider,
                capability="load_daily_market_data",
                request={
                    "start_date": start_date,
                    "end_date": end_date,
                    "reference_date": reference_date,
                    **self.provider_parameters,
                },
                frame=daily,
                sort_by=["business_date", "instrument_id"],
            )
        )
        daily = daily.set_index(["instrument_id", "business_date"], verify_integrity=True)
        data_context.daily = daily
        return data_context
