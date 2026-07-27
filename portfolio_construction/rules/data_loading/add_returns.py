from dataclasses import dataclass, field

import pandas as pd

from icapa.data_sources.contracts import validate_daily_market_data
from icapa.data_sources.registry import registry
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

    def get_output_fact_names(self):
        return [
            "price_return",
            "gross_dividend",
            "net_dividend",
            "market_cap",
            "total_return_gross",
            "total_return_net",
        ]

    def execute(self, data_context):
        provider = registry.resolve("load_daily_market_data", self.provider_name)
        instruments = data_context.cons.index
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
        daily["total_return_gross"] = daily["price_return"] + daily["gross_dividend"]
        daily["total_return_net"] = daily["price_return"] + daily["net_dividend"]
        daily = daily.set_index(["instrument_id", "business_date"], verify_integrity=True)
        data_context.daily = daily
        return data_context
