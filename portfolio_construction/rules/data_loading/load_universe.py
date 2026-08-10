"""Canonical provider-backed universe loading rule."""

from dataclasses import dataclass, field

import pandas as pd

from icapa.data_sources.contracts import UNIVERSE_COLUMNS, validate_universe
from icapa.data_sources.providers.registry import registry
from icapa.data_sources.provenance import automatic_data_identity
from icapa.portfolio_construction.rules.data_loading.base import DataLoadingRule


@dataclass
class LoadUniverse(DataLoadingRule):
    """Load a canonical point-in-time universe from a configured provider."""

    universe_id: str = ""
    provider_name: str = ""
    provider_parameters: dict = field(default_factory=dict)
    command: str = "LoadUniverse"

    def __post_init__(self):
        super().__post_init__()
        if not self.universe_id or not self.universe_id.strip():
            raise ValueError("universe_id is required for LoadUniverse")
        if not self.provider_name or not self.provider_name.strip():
            raise ValueError("provider_name is required for LoadUniverse")

    def get_output_field_names(self):
        return list(UNIVERSE_COLUMNS) + [
            "index_weight",
            "excluded",
            "exclusion_reason",
        ]

    def execute(self, data_context):
        provider = registry.resolve("load_universe", self.provider_name)
        universe = provider.load_universe(
            universe_id=self.universe_id,
            reference_date=data_context.reference_date,
            effective_date=data_context.effective_date,
            **self.provider_parameters,
        )
        if universe is None or universe.empty:
            raise ValueError(
                f"provider returned no universe rows for {self.universe_id}"
            )

        universe = validate_universe(universe)
        expected_reference_date = pd.Timestamp(
            data_context.reference_date
        ).normalize()
        expected_effective_date = pd.Timestamp(
            data_context.effective_date
        ).normalize()
        if not universe["reference_date"].eq(expected_reference_date).all():
            raise ValueError(
                "universe reference_date does not match the review context"
            )
        if not universe["effective_date"].eq(expected_effective_date).all():
            raise ValueError(
                "universe effective_date does not match the review context"
            )
        if universe.index.name != "instrument_id":
            universe = universe.set_index(
                "instrument_id",
                verify_integrity=True,
            )

        if "index_weight" not in universe.columns:
            universe["index_weight"] = universe["benchmark_weight"]
        if "excluded" not in universe.columns:
            universe["excluded"] = False
        if "exclusion_reason" not in universe.columns:
            universe["exclusion_reason"] = [[] for _ in range(len(universe))]

        data_context.provenance.record_provider_call(
            automatic_data_identity(
                provider_name=self.provider_name,
                provider=provider,
                capability="load_universe",
                request={
                    "universe_id": self.universe_id,
                    "reference_date": data_context.reference_date,
                    "effective_date": data_context.effective_date,
                    **self.provider_parameters,
                },
                frame=universe,
                sort_by=["instrument_id"],
            )
        )
        data_context.universe_id = self.universe_id
        data_context.set_dataframe(
            universe,
            columns=list(universe.columns),
        )
        return data_context


__all__ = ["LoadUniverse"]
