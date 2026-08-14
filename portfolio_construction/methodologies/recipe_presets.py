"""Recipe presets with specialized stages for public methodologies."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pandas as pd

from icapa.portfolio_construction.recipes.artifacts import (
    Artifact,
    ArtifactKey,
    ArtifactOutput,
    ArtifactRequirement,
    CORE_DAILY_DATA,
    CORE_DIAGNOSTICS,
    CORE_FINAL_CONSTITUENTS,
    CORE_TARGET_WEIGHTS,
    canonicalize,
)
from icapa.portfolio_construction.recipes import (
    IndexRecipe,
    StageCacheScope,
    StageDescriptor,
    StageDiagnostic,
    StageInputs,
    StageNode,
    StageRequirements,
    StageResult,
    StageRuntime,
    StageSideEffect,
)
from icapa.portfolio_construction.recipes.fingerprints import (
    component_tree_identity,
)
from icapa.portfolio_construction.recipes.methodology_presets import (
    MethodologyPresetStage,
    methodology_provider_request,
    methodology_recipe_preset,
)
from icapa.portfolio_construction.context import DataContext


_MINVAR_CONSTITUENTS = ArtifactKey(
    "icapa.minimum_variance",
    "constituents",
)
_MINVAR_DAILY = ArtifactKey(
    "icapa.minimum_variance",
    "daily_returns",
)
_MINVAR_COVARIANCE = ArtifactKey(
    "icapa.minimum_variance",
    "covariance",
)
_MINVAR_RISK_DIAGNOSTICS = ArtifactKey(
    "icapa.minimum_variance",
    "risk_diagnostics",
)


@dataclass(frozen=True)
class MinimumVarianceDataStage:
    """Load point-in-time constituents and return history."""

    universe_id: str
    universe_provider_name: str
    universe_provider_parameters: dict
    returns_provider_name: str
    returns_provider_parameters: dict
    start_date: object | None
    end_date: object | None

    @property
    def descriptor(self) -> StageDescriptor:
        return StageDescriptor(
            kind="icapa.minimum_variance.data",
            version="1",
            cache_scope=StageCacheScope.CONTENT,
            side_effect=StageSideEffect.READ_ONLY_IO,
            parallel_safe=False,
        )

    @property
    def requirements(self) -> StageRequirements:
        start_date = self.start_date
        end_date = self.end_date
        review_parameter_map = {}
        returns_request = {}
        if start_date is None:
            review_parameter_map["start_date"] = "reference_date"
        else:
            returns_request["start_date"] = pd.Timestamp(
                start_date
            ).normalize()
        if end_date is None:
            review_parameter_map["end_date"] = "reference_date"
        else:
            returns_request["end_date"] = pd.Timestamp(end_date).normalize()
        return StageRequirements(
            provider_capabilities=(
                "load_universe",
                "load_daily_market_data",
            ),
            provider_requests=(
                methodology_provider_request(
                    "load_universe",
                    provider_name=self.universe_provider_name,
                    provider_parameters=self.universe_provider_parameters,
                    request_parameters={
                        "universe_id": self.universe_id,
                    },
                    review_dimensions=frozenset(
                        {"reference_date", "effective_date"}
                    ),
                ),
                methodology_provider_request(
                    "load_daily_market_data",
                    provider_name=self.returns_provider_name,
                    provider_parameters=self.returns_provider_parameters,
                    request_parameters=returns_request,
                    review_parameter_map=review_parameter_map,
                    covers_all_instruments=True,
                ),
            ),
        )

    @property
    def outputs(self) -> tuple[ArtifactOutput, ...]:
        return (
            ArtifactOutput(_MINVAR_CONSTITUENTS),
            ArtifactOutput(_MINVAR_DAILY),
        )

    def canonical_configuration(self) -> Any:
        return {
            "universe_id": self.universe_id,
            "universe_provider_name": self.universe_provider_name,
            "universe_provider_parameters": canonicalize(
                self.universe_provider_parameters
            ),
            "returns_provider_name": self.returns_provider_name,
            "returns_provider_parameters": canonicalize(
                self.returns_provider_parameters
            ),
            "start_date": self.start_date,
            "end_date": self.end_date,
        }

    def wrapped_implementation_identity(self) -> dict[str, Any]:
        return component_tree_identity(self)

    def run(
        self,
        inputs: StageInputs,
        runtime: StageRuntime,
    ) -> StageResult:
        from icapa.portfolio_construction.rules.data_loading.add_returns import (
            AddReturns,
        )
        from icapa.portfolio_construction.rules.data_loading.load_universe import (
            LoadUniverse,
        )

        context = DataContext(
            index_id=inputs.review.index_id,
            reference_date=inputs.review.reference_date,
            effective_date=inputs.review.effective_date,
        )
        LoadUniverse(
            universe_id=self.universe_id,
            provider_name=self.universe_provider_name,
            provider_parameters=dict(self.universe_provider_parameters),
        ).execute(context)
        AddReturns(
            provider_name=self.returns_provider_name,
            start_date=self.start_date,
            end_date=self.end_date,
            provider_parameters=dict(self.returns_provider_parameters),
        ).execute(context)
        return StageResult(
            {
                _MINVAR_CONSTITUENTS: Artifact.from_value(
                    _MINVAR_CONSTITUENTS,
                    context.constituents.copy(deep=True),
                ),
                _MINVAR_DAILY: Artifact.from_value(
                    _MINVAR_DAILY,
                    context.daily.copy(deep=True),
                ),
            }
        )


@dataclass(frozen=True)
class MinimumVarianceCovarianceStage:
    """Estimate and cache the formal point-in-time covariance window."""

    return_column: str
    minimum_observations: int
    covariance_ridge: float
    return_window: object | None
    covariance_estimator: object | None

    @property
    def descriptor(self) -> StageDescriptor:
        return StageDescriptor(
            kind="icapa.minimum_variance.covariance",
            version="1",
            cache_scope=StageCacheScope.CONTENT,
        )

    @property
    def requirements(self) -> StageRequirements:
        return StageRequirements(
            artifacts=(
                ArtifactRequirement(_MINVAR_CONSTITUENTS),
                ArtifactRequirement(_MINVAR_DAILY),
            )
        )

    @property
    def outputs(self) -> tuple[ArtifactOutput, ...]:
        return (
            ArtifactOutput(_MINVAR_COVARIANCE),
            ArtifactOutput(_MINVAR_RISK_DIAGNOSTICS),
        )

    def canonical_configuration(self) -> Any:
        return {
            "return_column": self.return_column,
            "minimum_observations": self.minimum_observations,
            "covariance_ridge": self.covariance_ridge,
            "return_window": canonicalize(self.return_window),
            "covariance_estimator": canonicalize(self.covariance_estimator),
        }

    def wrapped_implementation_identity(self) -> dict[str, Any]:
        return component_tree_identity(self)

    def run(
        self,
        inputs: StageInputs,
        runtime: StageRuntime,
    ) -> StageResult:
        from icapa.portfolio_construction.optimization import (
            ReturnWindowSpec,
            SampleCovarianceEstimator,
            estimate_covariance_for_window,
        )

        constituents = inputs.value(_MINVAR_CONSTITUENTS)
        daily = inputs.value(_MINVAR_DAILY).reset_index()
        returns = daily.pivot(
            index="business_date",
            columns="instrument_id",
            values=self.return_column,
        ).reindex(columns=constituents.index)
        window = self.return_window or ReturnWindowSpec(
            lookback=252,
            minimum_observations=self.minimum_observations,
        )
        estimator = (
            self.covariance_estimator
            or SampleCovarianceEstimator(
                minimum_observations=self.minimum_observations,
                ridge=self.covariance_ridge,
            )
        )
        resolved, estimate = estimate_covariance_for_window(
            estimator,
            returns,
            window,
            inputs.review.reference_date,
        )
        covariance = pd.DataFrame(
            estimate.matrix,
            index=pd.Index(
                estimate.instrument_ids,
                name="instrument_id",
            ),
            columns=pd.Index(
                estimate.instrument_ids,
                name="instrument_id",
            ),
        )
        diagnostics = {
            "observations": estimate.observations,
            "return_window": {
                "start_date": resolved.start_date.isoformat(),
                "end_date": resolved.end_date.isoformat(),
                "observation_count": resolved.observation_count,
                "kind": window.kind.value,
                "lookback": window.lookback,
            },
            "covariance_estimator": estimate.estimator_name,
            "covariance_metadata": canonicalize(dict(estimate.metadata)),
        }
        return StageResult(
            {
                _MINVAR_COVARIANCE: Artifact.from_value(
                    _MINVAR_COVARIANCE,
                    covariance,
                ),
                _MINVAR_RISK_DIAGNOSTICS: Artifact.from_value(
                    _MINVAR_RISK_DIAGNOSTICS,
                    diagnostics,
                ),
            },
            diagnostics=(
                StageDiagnostic(
                    code="minimum_variance_covariance_estimated",
                    metrics={
                        "observation_count": estimate.observations,
                        "instrument_count": len(estimate.instrument_ids),
                    },
                ),
            ),
        )


@dataclass(frozen=True)
class MinimumVarianceWeightStage:
    """Solve minimum variance from a cached covariance artifact."""

    universe_id: str
    minimum_weight: float
    maximum_weight: float
    capacity_multiple: float | None
    group_tolerances: object
    solver: object | None
    maximum_tracking_error: float | None = None
    maximum_one_way_turnover: float | None = None
    turnover_reference_column: str | None = None

    @property
    def descriptor(self) -> StageDescriptor:
        return StageDescriptor(
            kind="icapa.minimum_variance.weighting",
            version="1",
            cache_scope=StageCacheScope.CONTENT,
        )

    @property
    def requirements(self) -> StageRequirements:
        return StageRequirements(
            artifacts=(
                ArtifactRequirement(_MINVAR_CONSTITUENTS),
                ArtifactRequirement(_MINVAR_DAILY),
                ArtifactRequirement(_MINVAR_COVARIANCE),
                ArtifactRequirement(_MINVAR_RISK_DIAGNOSTICS),
            )
        )

    @property
    def outputs(self) -> tuple[ArtifactOutput, ...]:
        return (
            ArtifactOutput(CORE_FINAL_CONSTITUENTS),
            ArtifactOutput(CORE_TARGET_WEIGHTS),
            ArtifactOutput(CORE_DIAGNOSTICS),
            ArtifactOutput(CORE_DAILY_DATA),
        )

    def canonical_configuration(self) -> Any:
        return {
            "universe_id": self.universe_id,
            "minimum_weight": self.minimum_weight,
            "maximum_weight": self.maximum_weight,
            "capacity_multiple": self.capacity_multiple,
            "group_tolerances": canonicalize(self.group_tolerances),
            "maximum_tracking_error": self.maximum_tracking_error,
            "maximum_one_way_turnover": self.maximum_one_way_turnover,
            "turnover_reference_column": self.turnover_reference_column,
            "solver": canonicalize(self.solver),
        }

    def wrapped_implementation_identity(self) -> dict[str, Any]:
        return component_tree_identity(self)

    def run(
        self,
        inputs: StageInputs,
        runtime: StageRuntime,
    ) -> StageResult:
        from icapa.portfolio_construction.engines.minimum_variance_engine import (
            MinimumVarianceEngine,
        )

        context = DataContext(
            index_id=inputs.review.index_id,
            universe_id=self.universe_id,
            reference_date=inputs.review.reference_date,
            effective_date=inputs.review.effective_date,
        )
        constituents = inputs.value(_MINVAR_CONSTITUENTS).copy(deep=True)
        context.set_dataframe(constituents)
        context.daily = inputs.value(_MINVAR_DAILY).copy(deep=True)
        covariance = inputs.value(_MINVAR_COVARIANCE).reindex(
            index=constituents.index,
            columns=constituents.index,
        )
        MinimumVarianceEngine(
            minimum_weight=self.minimum_weight,
            maximum_weight=self.maximum_weight,
            capacity_multiple=self.capacity_multiple,
            group_tolerances=self.group_tolerances,
            maximum_tracking_error=self.maximum_tracking_error,
            maximum_one_way_turnover=self.maximum_one_way_turnover,
            turnover_reference_column=self.turnover_reference_column,
            solver=self.solver,
        ).execute_with_covariance(
            context,
            covariance.to_numpy(dtype=float),
            risk_diagnostics=inputs.value(
                _MINVAR_RISK_DIAGNOSTICS
            ),
        )
        return StageResult(
            {
                CORE_FINAL_CONSTITUENTS: Artifact.from_value(
                    CORE_FINAL_CONSTITUENTS,
                    context.constituents.copy(deep=True),
                ),
                CORE_TARGET_WEIGHTS: Artifact.from_value(
                    CORE_TARGET_WEIGHTS,
                    context.constituents[["index_weight"]].copy(),
                ),
                CORE_DIAGNOSTICS: Artifact.from_value(
                    CORE_DIAGNOSTICS,
                    deepcopy(context.diagnostics),
                ),
                CORE_DAILY_DATA: Artifact.from_value(
                    CORE_DAILY_DATA,
                    context.daily.copy(deep=True),
                ),
            }
        )


def minimum_variance_recipe_preset(methodology: object) -> IndexRecipe:
    """Build the cacheable data, risk, and weighting minimum-variance DAG."""

    return IndexRecipe(
        nodes=(
            StageNode(
                "load_data",
                MinimumVarianceDataStage(
                    universe_id=methodology.universe_id,
                    universe_provider_name=methodology.universe_provider_name,
                    universe_provider_parameters=dict(
                        methodology.universe_provider_parameters
                    ),
                    returns_provider_name=methodology.returns_provider_name,
                    returns_provider_parameters=dict(
                        methodology.returns_provider_parameters
                    ),
                    start_date=methodology.start_date,
                    end_date=methodology.end_date,
                ),
            ),
            StageNode(
                "estimate_covariance",
                MinimumVarianceCovarianceStage(
                    return_column=methodology.return_column,
                    minimum_observations=methodology.minimum_observations,
                    covariance_ridge=methodology.covariance_ridge,
                    return_window=methodology.return_window,
                    covariance_estimator=methodology.covariance_estimator,
                ),
            ),
            StageNode(
                "solve_weights",
                MinimumVarianceWeightStage(
                    universe_id=methodology.universe_id,
                    minimum_weight=methodology.minimum_weight,
                    maximum_weight=methodology.maximum_weight,
                    capacity_multiple=methodology.capacity_multiple,
                    group_tolerances=dict(methodology.group_tolerances),
                    solver=methodology.solver,
                    maximum_tracking_error=methodology.maximum_tracking_error,
                    maximum_one_way_turnover=methodology.maximum_one_way_turnover,
                    turnover_reference_column=methodology.turnover_reference_column,
                ),
            ),
        )
    )


__all__ = [
    "MethodologyPresetStage",
    "MinimumVarianceCovarianceStage",
    "MinimumVarianceDataStage",
    "MinimumVarianceWeightStage",
    "minimum_variance_recipe_preset",
    "methodology_provider_request",
    "methodology_recipe_preset",
]
