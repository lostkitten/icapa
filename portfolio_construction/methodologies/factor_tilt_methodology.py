"""Generic factor-tilt methodology."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from icapa.portfolio_construction.recipes.factors import (
    FactorStandardizationVariant,
    StandardizeFactors,
)
from icapa.portfolio_construction.optimization import PortfolioSolver
from icapa.portfolio_construction.rules.data_loading.add_third_party_data import (
    AddThirdPartyData,
)
from icapa.portfolio_construction.rules.data_loading.load_universe import (
    LoadUniverse,
)
from icapa.portfolio_construction.engines.factor_tilt_engine import (
    FactorTiltEngine,
    TiltScheme,
)
from icapa.data_sources.contracts import ThirdPartyDataType


@dataclass
class FactorTiltMethodology:
    """Load factor data, standardize it cross-sectionally, and apply multiplicative factor tilts.

    ``zscore_clip`` bounds each standardized factor during cross-sectional
    standardization; ``score_clip`` bounds the weighted composite factor
    score before the tilt multiplier is applied. The two limits act at
    different stages and are configured independently.
    """

    factor_tilts: Mapping[str, float]
    universe_id: str
    universe_provider_name: str
    factor_provider_name: str
    index_id: str | None = None
    universe_provider_parameters: dict = field(default_factory=dict)
    factor_provider_parameters: dict = field(default_factory=dict)
    tilt_scheme: TiltScheme = TiltScheme.EXPONENTIAL
    tilt_strength: float = 1.0
    score_clip: float = 20.0
    zscore_clip: float | None = 3.0
    factor_standardization_variant: FactorStandardizationVariant = (
        FactorStandardizationVariant.STANDARD
    )
    minimum_weight: float = 0.0
    maximum_weight: float = 1.0
    capacity_multiple: float | None = None
    group_tolerances: Mapping[str, float] = field(default_factory=dict)
    solver: PortfolioSolver | None = field(default=None, repr=False)

    def to_recipe(self):
        """Return the provider-aware recipe preset for this methodology."""

        from icapa.portfolio_construction.recipes.methodology_presets import (
            methodology_provider_request,
            methodology_recipe_preset,
        )

        return methodology_recipe_preset(
            self,
            provider_requests=(
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
                    provider_name=self.factor_provider_name,
                    provider_parameters=self.factor_provider_parameters,
                    request_parameters={
                        "data_type": ThirdPartyDataType.FACTOR_DATA.value,
                        "fields": tuple(self.factor_tilts),
                    },
                    review_dimensions=frozenset({"reference_date"}),
                    provider_parameters_key="parameters",
                    covers_all_instruments=True,
                ),
            ),
        )

    def execute(self, data_context):
        if not self.universe_id:
            raise ValueError("universe_id is required")
        if not self.factor_provider_name:
            raise ValueError("factor_provider_name is required")
        if self.index_id is not None:
            data_context.index_id = self.index_id
        LoadUniverse(
            universe_id=self.universe_id,
            provider_name=self.universe_provider_name,
            provider_parameters=dict(self.universe_provider_parameters),
        ).execute(data_context)
        AddThirdPartyData(
            data_type=ThirdPartyDataType.FACTOR_DATA,
            fields=list(self.factor_tilts),
            provider_name=self.factor_provider_name,
            provider_parameters=dict(self.factor_provider_parameters),
        ).execute(data_context)
        StandardizeFactors(
            factor_fields=list(self.factor_tilts),
            zscore_clip=self.zscore_clip,
            standardization_variant=self.factor_standardization_variant,
        ).execute(data_context)
        FactorTiltEngine(
            factor_tilts=self.factor_tilts,
            tilt_scheme=self.tilt_scheme,
            tilt_strength=self.tilt_strength,
            score_clip=self.score_clip,
            minimum_weight=self.minimum_weight,
            maximum_weight=self.maximum_weight,
            capacity_multiple=self.capacity_multiple,
            group_tolerances=self.group_tolerances,
            solver=self.solver,
        ).execute(data_context)
        return data_context


__all__ = ["FactorTiltMethodology"]
