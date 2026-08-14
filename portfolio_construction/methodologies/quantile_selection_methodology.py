"""Generic quantile-selection methodology."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from icapa.portfolio_construction.recipes.factors import (
    FactorStandardizationVariant,
    StandardizeFactors,
)
from icapa.portfolio_construction.rules.data_loading.add_third_party_data import (
    AddThirdPartyData,
)
from icapa.portfolio_construction.rules.data_loading.load_universe import (
    LoadUniverse,
)
from icapa.portfolio_construction.engines.quantile_selection_engine import (
    QuantileSelectionEngine,
    SelectionCriterion,
    SelectionScope,
    SelectionWeighting,
)
from icapa.data_sources.contracts import ThirdPartyDataType


@dataclass
class QuantileSelectionMethodology:
    """Load signal data, standardize it cross-sectionally, and select the top score quantile.

    ``zscore_clip`` bounds each standardized signal during cross-sectional
    standardization before the weighted composite selection score is formed.
    """

    signal_weights: Mapping[str, float]
    universe_id: str
    universe_provider_name: str
    signal_provider_name: str
    index_id: str | None = None
    universe_provider_parameters: dict = field(default_factory=dict)
    signal_provider_parameters: dict = field(default_factory=dict)
    selection_fraction: float = 0.4
    selection_criterion: SelectionCriterion = SelectionCriterion.COUNT
    selection_scope: SelectionScope = SelectionScope.UNIVERSE
    selection_weighting: SelectionWeighting = SelectionWeighting.PROPORTIONAL
    group_column: str = "industry"
    zscore_clip: float | None = 3.0
    factor_standardization_variant: FactorStandardizationVariant = (
        FactorStandardizationVariant.STANDARD
    )
    liquidity_field: str | None = None
    minimum_liquidity: float | None = None
    liquidity_provider_name: str | None = None
    liquidity_provider_parameters: dict = field(default_factory=dict)

    def to_recipe(self):
        """Return the provider-aware recipe preset for this methodology."""

        from icapa.portfolio_construction.recipes.methodology_presets import (
            methodology_provider_request,
            methodology_recipe_preset,
        )

        requests = [
            methodology_provider_request(
                "load_universe",
                provider_name=self.universe_provider_name,
                provider_parameters=self.universe_provider_parameters,
                request_parameters={"universe_id": self.universe_id},
                review_dimensions=frozenset(
                    {"reference_date", "effective_date"}
                ),
            ),
            methodology_provider_request(
                "load_third_party_data",
                provider_name=self.signal_provider_name,
                provider_parameters=self.signal_provider_parameters,
                request_parameters={
                    "data_type": ThirdPartyDataType.FACTOR_DATA.value,
                    "fields": tuple(self.signal_weights),
                },
                review_dimensions=frozenset({"reference_date"}),
                provider_parameters_key="parameters",
                covers_all_instruments=True,
            ),
        ]
        if self.liquidity_field is not None:
            requests.append(
                methodology_provider_request(
                    "load_third_party_data",
                    provider_name=self.liquidity_provider_name or "",
                    provider_parameters=self.liquidity_provider_parameters,
                    request_parameters={
                        "data_type": ThirdPartyDataType.LIQUIDITY_DATA.value,
                        "fields": (self.liquidity_field,),
                    },
                    review_dimensions=frozenset({"reference_date"}),
                    provider_parameters_key="parameters",
                    covers_all_instruments=True,
                )
            )
        return methodology_recipe_preset(
            self,
            provider_requests=tuple(requests),
        )

    def execute(self, data_context):
        if not self.universe_id:
            raise ValueError("universe_id is required")
        if not self.signal_provider_name:
            raise ValueError("signal_provider_name is required")
        if self.index_id is not None:
            data_context.index_id = self.index_id
        LoadUniverse(
            universe_id=self.universe_id,
            provider_name=self.universe_provider_name,
            provider_parameters=dict(self.universe_provider_parameters),
        ).execute(data_context)
        AddThirdPartyData(
            data_type=ThirdPartyDataType.FACTOR_DATA,
            fields=list(self.signal_weights),
            provider_name=self.signal_provider_name,
            provider_parameters=dict(self.signal_provider_parameters),
        ).execute(data_context)
        if self.liquidity_field is not None:
            if not self.liquidity_provider_name:
                raise ValueError(
                    "liquidity_provider_name is required when liquidity_field is supplied"
                )
            AddThirdPartyData(
                data_type=ThirdPartyDataType.LIQUIDITY_DATA,
                fields=[self.liquidity_field],
                provider_name=self.liquidity_provider_name,
                provider_parameters=dict(self.liquidity_provider_parameters),
            ).execute(data_context)
        StandardizeFactors(
            factor_fields=list(self.signal_weights),
            zscore_clip=self.zscore_clip,
            standardization_variant=self.factor_standardization_variant,
        ).execute(data_context)
        QuantileSelectionEngine(
            signal_weights=self.signal_weights,
            selection_fraction=self.selection_fraction,
            selection_criterion=self.selection_criterion,
            selection_scope=self.selection_scope,
            selection_weighting=self.selection_weighting,
            group_column=self.group_column,
            liquidity_field=self.liquidity_field,
            minimum_liquidity=self.minimum_liquidity,
        ).execute(data_context)
        return data_context


__all__ = ["QuantileSelectionMethodology"]
