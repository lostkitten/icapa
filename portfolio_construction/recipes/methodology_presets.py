"""Provider-aware recipe presets for executable methodologies."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from icapa.portfolio_construction.context import DataContext

from .artifacts import (
    Artifact,
    ArtifactOutput,
    CORE_DAILY_DATA,
    CORE_DIAGNOSTICS,
    CORE_FINAL_CONSTITUENTS,
    CORE_TARGET_WEIGHTS,
    canonicalize,
    qualified_name,
)
from .contracts import (
    ProviderRequestSpec,
    StageCacheScope,
    StageDescriptor,
    StageDiagnostic,
    StageInputs,
    StageRequirements,
    StageResult,
    StageRuntime,
    StageSideEffect,
)
from .fingerprints import component_tree_identity
from .graph import IndexRecipe, StageNode


def methodology_provider_request(
    capability: str,
    *,
    provider_name: str,
    provider_parameters: dict,
    request_parameters: dict,
    review_dimensions: frozenset[str] = frozenset(),
    review_parameter_map: dict[str, str] | None = None,
    provider_parameters_key: str | None = None,
    covers_all_instruments: bool = False,
) -> ProviderRequestSpec:
    """Build one provider request used by a methodology preset."""

    return ProviderRequestSpec(
        capability=capability,
        review_dimensions=review_dimensions,
        request_parameters=request_parameters,
        provider_parameters_key=provider_parameters_key,
        review_parameter_map=(
            {} if review_parameter_map is None else review_parameter_map
        ),
        expected_provider_name=provider_name,
        expected_provider_parameters=dict(provider_parameters),
        covers_all_instruments=covers_all_instruments,
    )


@dataclass(frozen=True)
class MethodologyPresetStage:
    """Run a methodology with explicit provider requirements and outputs."""

    methodology: object
    provider_requests: tuple[ProviderRequestSpec, ...]

    def __post_init__(self) -> None:
        if not callable(getattr(self.methodology, "execute", None)):
            raise TypeError("methodology must implement execute(data_context)")
        requests = tuple(self.provider_requests)
        if not requests:
            raise ValueError(
                "a methodology recipe preset must declare provider requests"
            )
        if any(not isinstance(item, ProviderRequestSpec) for item in requests):
            raise TypeError(
                "provider_requests must contain ProviderRequestSpec values"
            )
        object.__setattr__(self, "provider_requests", requests)

    @property
    def provider_capabilities(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                request.capability for request in self.provider_requests
            )
        )

    @property
    def descriptor(self) -> StageDescriptor:
        return StageDescriptor(
            kind=f"{qualified_name(self.methodology)}.recipe_preset",
            version="1",
            deterministic=True,
            cache_scope=StageCacheScope.CONTENT,
            side_effect=StageSideEffect.READ_ONLY_IO,
            parallel_safe=False,
        )

    @property
    def requirements(self) -> StageRequirements:
        return StageRequirements(
            provider_capabilities=self.provider_capabilities,
            provider_requests=self.provider_requests,
        )

    @property
    def outputs(self) -> tuple[ArtifactOutput, ...]:
        return (
            ArtifactOutput(CORE_FINAL_CONSTITUENTS),
            ArtifactOutput(CORE_TARGET_WEIGHTS),
            ArtifactOutput(CORE_DIAGNOSTICS),
            ArtifactOutput(CORE_DAILY_DATA, optional=True),
        )

    def canonical_configuration(self) -> Any:
        return {
            "methodology_type": qualified_name(self.methodology),
            "methodology": canonicalize(self.methodology),
            "provider_capabilities": list(self.provider_capabilities),
        }

    def wrapped_implementation_identity(self) -> dict[str, Any]:
        """Fingerprint methodology, engine, and injected behavior source."""

        return component_tree_identity(self.methodology)

    def run(
        self,
        inputs: StageInputs,
        runtime: StageRuntime,
    ) -> StageResult:
        context = DataContext(
            index_id=inputs.review.index_id,
            universe_id=inputs.review.universe_id,
            reference_date=inputs.review.reference_date,
            effective_date=inputs.review.effective_date,
        )
        result = self.methodology.execute(context)
        if result is not None:
            if not isinstance(result, DataContext):
                raise TypeError(
                    "methodology recipe presets must return DataContext or None"
                )
            context = result
        constituents = context.cons.copy(deep=True)
        if constituents.empty or "index_weight" not in constituents:
            raise ValueError(
                "methodology preset did not produce canonical index weights"
            )
        artifacts = {
            CORE_FINAL_CONSTITUENTS: Artifact.from_value(
                CORE_FINAL_CONSTITUENTS,
                constituents,
            ),
            CORE_TARGET_WEIGHTS: Artifact.from_value(
                CORE_TARGET_WEIGHTS,
                constituents[["index_weight"]].copy(),
            ),
            CORE_DIAGNOSTICS: Artifact.from_value(
                CORE_DIAGNOSTICS,
                deepcopy(context.diagnostics),
            ),
        }
        if context.daily is not None:
            artifacts[CORE_DAILY_DATA] = Artifact.from_value(
                CORE_DAILY_DATA,
                context.daily.copy(deep=True),
            )
        return StageResult(
            artifacts=artifacts,
            diagnostics=(
                StageDiagnostic(
                    code="methodology_recipe_preset_completed",
                    metrics={
                        "constituent_count": int(len(constituents)),
                        "has_daily_data": context.daily is not None,
                    },
                ),
            ),
        )


def methodology_recipe_preset(
    methodology: object,
    *,
    provider_requests: tuple[ProviderRequestSpec, ...],
) -> IndexRecipe:
    """Build a canonical provider-aware recipe for one methodology."""

    return IndexRecipe(
        nodes=(
            StageNode(
                "construct_target_weights",
                MethodologyPresetStage(
                    methodology=methodology,
                    provider_requests=provider_requests,
                ),
            ),
        )
    )


__all__ = [
    "MethodologyPresetStage",
    "methodology_provider_request",
    "methodology_recipe_preset",
]
