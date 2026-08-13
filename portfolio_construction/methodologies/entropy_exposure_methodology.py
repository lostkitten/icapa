"""Minimum-relative-entropy exposure-targeting methodology."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from icapa.data_sources.contracts import ThirdPartyDataType
from icapa.portfolio_construction.engines.entropy_exposure_engine import (
    EntropyExposureEngine,
    EntropyExposureMode,
    ExposureTarget,
)
from icapa.portfolio_construction.optimization import (
    EGMUConstrainedElasticSolver,
    EGMUNewtonSolver,
    EGMUProjectionSolver,
)
from icapa.portfolio_construction.rules.data_loading.add_third_party_data import (
    AddThirdPartyData,
)
from icapa.portfolio_construction.rules.data_loading.load_universe import (
    LoadUniverse,
)


@dataclass
class EntropyExposureMethodology:
    """Load exposure fields and construct an EGMU target portfolio."""

    targets: Sequence[ExposureTarget]
    universe_id: str
    universe_provider_name: str
    index_id: str | None = None
    target_provider_name: str | None = None
    target_data_type: ThirdPartyDataType = ThirdPartyDataType.FACTOR_DATA
    universe_provider_parameters: dict = field(default_factory=dict)
    target_provider_parameters: dict = field(default_factory=dict)
    mode: EntropyExposureMode = EntropyExposureMode.HARD
    elastic_penalty: float = 100.0
    minimum_weight: float = 0.0
    maximum_weight: float = 1.0
    capacity_multiple: float | None = None
    group_tolerances: Mapping[str, float] = field(default_factory=dict)
    max_iterations: int = 1_000
    tolerance: float = 1.0e-8
    newton_ridge: float = 1.0e-10
    newton_solver: EGMUNewtonSolver | None = field(default=None, repr=False)
    elastic_solver: EGMUConstrainedElasticSolver | None = field(
        default=None,
        repr=False,
    )
    projection_solver: EGMUProjectionSolver | None = field(
        default=None,
        repr=False,
    )

    def to_recipe(self):
        """Return a provider-aware recipe for this methodology."""

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
            )
        ]
        if self.target_provider_name:
            requests.append(
                methodology_provider_request(
                    "load_third_party_data",
                    provider_name=self.target_provider_name,
                    provider_parameters=self.target_provider_parameters,
                    request_parameters={
                        "data_type": ThirdPartyDataType(
                            self.target_data_type
                        ).value,
                        "fields": tuple(
                            dict.fromkeys(
                                target.field for target in self.targets
                            )
                        ),
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
        if not self.targets:
            raise ValueError("targets cannot be empty")
        if self.index_id is not None:
            data_context.index_id = self.index_id
        LoadUniverse(
            universe_id=self.universe_id,
            provider_name=self.universe_provider_name,
            provider_parameters=dict(self.universe_provider_parameters),
        ).execute(data_context)
        if self.target_provider_name:
            AddThirdPartyData(
                data_type=ThirdPartyDataType(self.target_data_type),
                fields=list(
                    dict.fromkeys(target.field for target in self.targets)
                ),
                provider_name=self.target_provider_name,
                provider_parameters=dict(self.target_provider_parameters),
            ).execute(data_context)
        EntropyExposureEngine(
            targets=self.targets,
            mode=self.mode,
            elastic_penalty=self.elastic_penalty,
            minimum_weight=self.minimum_weight,
            maximum_weight=self.maximum_weight,
            capacity_multiple=self.capacity_multiple,
            group_tolerances=self.group_tolerances,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            newton_ridge=self.newton_ridge,
            newton_solver=self.newton_solver,
            elastic_solver=self.elastic_solver,
            projection_solver=self.projection_solver,
        ).execute(data_context)
        return data_context


__all__ = ["EntropyExposureMethodology"]
