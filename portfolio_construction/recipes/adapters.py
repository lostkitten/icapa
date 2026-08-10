"""Adapters between recipe stages and ``execute(DataContext)`` components."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import pandas as pd

from icapa.portfolio_construction.context import DataContext

from .artifacts import (
    Artifact,
    ArtifactKey,
    ArtifactOutput,
    ArtifactRequirement,
    CORE_CONSTITUENTS,
    CORE_DAILY_DATA,
    CORE_DIAGNOSTICS,
    CORE_FINAL_CONSTITUENTS,
    CORE_TARGET_WEIGHTS,
    JsonValue,
    canonicalize,
    qualified_name,
)
from .contracts import (
    ReviewIdentity,
    StageCacheScope,
    StageDescriptor,
    StageDiagnostic,
    StageInputs,
    StageRequirements,
    StageResult,
    StageRuntime,
    StageSideEffect,
)
from .execution import (
    MemoryStageCache,
    PreviousReviewState,
    RecipeRunner,
    ReviewConstructionResult,
    StageCache,
)
from .graph import IndexRecipe


@dataclass(frozen=True)
class MethodologyExecutionStage:
    """Adapt an ``execute(DataContext)`` methodology to an opaque recipe stage."""

    methodology: object
    kind: str | None = None
    version: str = "1"

    def __post_init__(self) -> None:
        if not callable(getattr(self.methodology, "execute", None)):
            raise TypeError("methodology must implement execute(data_context)")

    @property
    def descriptor(self) -> StageDescriptor:
        return StageDescriptor(
            kind=self.kind or qualified_name(self.methodology),
            version=self.version,
            deterministic=True,
            cache_scope=StageCacheScope.DISABLED,
            side_effect=StageSideEffect.OPAQUE,
            parallel_safe=False,
        )

    @property
    def requirements(self) -> StageRequirements:
        return StageRequirements(consume_all_current=True)

    @property
    def completion_diagnostic_code(self) -> str:
        """Return the neutral completion code."""

        return "methodology_execute_completed"

    @property
    def outputs(self) -> tuple[ArtifactOutput, ...]:
        return (
            ArtifactOutput(CORE_FINAL_CONSTITUENTS),
            ArtifactOutput(CORE_TARGET_WEIGHTS),
            ArtifactOutput(CORE_DIAGNOSTICS),
            ArtifactOutput(CORE_DAILY_DATA, optional=True),
        )

    def canonical_configuration(self) -> JsonValue:
        return {
            "methodology_type": qualified_name(self.methodology),
            "methodology": canonicalize(self.methodology),
        }

    def run(self, inputs: StageInputs, runtime: StageRuntime) -> StageResult:
        context = _context_for_stage(inputs, runtime)
        result = self.methodology.execute(context)
        if result is not None:
            if not isinstance(result, DataContext):
                raise TypeError(
                    "methodology must return DataContext or None"
                )
            context = result
        constituents = context.cons.copy(deep=True)
        if constituents.empty:
            raise ValueError("methodology produced no constituent data")
        if "index_weight" not in constituents:
            raise ValueError("methodology did not produce index_weight")
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
                    code=self.completion_diagnostic_code,
                    metrics={
                        "constituent_count": int(len(constituents)),
                        "has_daily_data": context.daily is not None,
                    },
                ),
            ),
        )


@dataclass(frozen=True)
class RuleExecutionStage:
    """Adapt an ``execute(DataContext)`` rule to an opaque recipe stage."""

    rule: object
    input_key: ArtifactKey
    output_key: ArtifactKey
    daily_input_key: ArtifactKey | None = None
    daily_output_key: ArtifactKey | None = None
    kind: str | None = None
    version: str = "1"

    def __post_init__(self) -> None:
        if not callable(getattr(self.rule, "execute", None)):
            raise TypeError("rule must implement execute(data_context)")
        if self.output_key == self.input_key:
            raise ValueError("rule output_key must differ from input_key")

    @property
    def descriptor(self) -> StageDescriptor:
        return StageDescriptor(
            kind=self.kind or qualified_name(self.rule),
            version=self.version,
            deterministic=True,
            cache_scope=StageCacheScope.DISABLED,
            side_effect=StageSideEffect.OPAQUE,
            parallel_safe=False,
        )

    @property
    def requirements(self) -> StageRequirements:
        artifacts = [ArtifactRequirement(self.input_key)]
        if self.daily_input_key is not None:
            artifacts.append(ArtifactRequirement(self.daily_input_key, optional=True))
        return StageRequirements(artifacts=tuple(artifacts))

    @property
    def outputs(self) -> tuple[ArtifactOutput, ...]:
        outputs = [ArtifactOutput(self.output_key)]
        if self.daily_output_key is not None:
            outputs.append(ArtifactOutput(self.daily_output_key, optional=True))
        return tuple(outputs)

    def canonical_configuration(self) -> JsonValue:
        return {
            "rule_type": qualified_name(self.rule),
            "rule": canonicalize(self.rule),
            "input_key": self.input_key.canonical_name,
            "output_key": self.output_key.canonical_name,
            "daily_input_key": (
                None
                if self.daily_input_key is None
                else self.daily_input_key.canonical_name
            ),
            "daily_output_key": (
                None
                if self.daily_output_key is None
                else self.daily_output_key.canonical_name
            ),
        }

    def run(self, inputs: StageInputs, runtime: StageRuntime) -> StageResult:
        context = DataContext(
            reference_date=inputs.review.reference_date,
            effective_date=inputs.review.effective_date,
            index_id=inputs.review.index_id,
            universe_id=inputs.review.universe_id,
        )
        context.set_dataframe(_as_constituent_frame(inputs.value(self.input_key)))
        if self.daily_input_key is not None and self.daily_input_key in inputs.artifacts:
            context.daily = deepcopy(inputs.value(self.daily_input_key))
        result = self.rule.execute(context)
        if result is not None:
            if not isinstance(result, DataContext):
                raise TypeError("rule must return DataContext or None")
            context = result
        artifacts = {
            self.output_key: Artifact.from_value(
                self.output_key,
                context.cons.copy(deep=True),
            )
        }
        if self.daily_output_key is not None and context.daily is not None:
            artifacts[self.daily_output_key] = Artifact.from_value(
                self.daily_output_key,
                context.daily.copy(deep=True),
            )
        return StageResult(artifacts)


class RecipeWeightProducer:
    """Expose a native recipe through the existing ``execute(DataContext)`` API."""

    def __init__(
        self,
        recipe: IndexRecipe,
        *,
        runtime: StageRuntime | None = None,
        cache: StageCache | None = None,
        previous_state_provider: (
            Callable[[ReviewIdentity], PreviousReviewState | None] | None
        ) = None,
        runtime_provider: (
            Callable[[ReviewIdentity], StageRuntime] | None
        ) = None,
        allow_empty_initial_state: bool = False,
        memory_cache_when_unspecified: bool = True,
    ) -> None:
        self.recipe = recipe
        self._runtime = runtime or StageRuntime()
        self._runner = RecipeRunner(
            runtime=self._runtime,
            cache=(
                cache
                if cache is not None
                else (
                    MemoryStageCache()
                    if memory_cache_when_unspecified
                    else None
                )
            ),
        )
        self._previous_state_provider = previous_state_provider
        self._runtime_provider = runtime_provider
        self._allow_empty_initial_state = bool(allow_empty_initial_state)
        self._last_result: ReviewConstructionResult | None = None

    @property
    def last_result(self) -> ReviewConstructionResult | None:
        """Return the most recent native result for notebook inspection."""

        return self._last_result

    def execute(self, data_context: DataContext) -> DataContext:
        """Run the recipe and merge its canonical result into ``data_context``."""

        if not isinstance(data_context, DataContext):
            raise TypeError("data_context must be a DataContext")
        review = ReviewIdentity(
            index_id=data_context.index_id,
            universe_id=data_context.universe_id,
            reference_date=data_context.reference_date,
            effective_date=data_context.effective_date,
        )
        initial: dict[ArtifactKey, Any] = {}
        if not data_context.cons.empty:
            initial[CORE_CONSTITUENTS] = data_context.cons.copy(deep=True)
        if data_context.daily is not None:
            initial[CORE_DAILY_DATA] = data_context.daily.copy(deep=True)
        previous = (
            None
            if self._previous_state_provider is None
            else self._previous_state_provider(review)
        )
        if (
            previous is None
            and self._last_result is not None
            and self._last_result.review.effective_date
            < review.effective_date
        ):
            previous = self._last_result.as_previous_state()
        selected_runtime = (
            self._runtime
            if self._runtime_provider is None
            else self._runtime_provider(review)
        )
        if not isinstance(selected_runtime, StageRuntime):
            raise TypeError(
                "runtime_provider must return a StageRuntime"
            )
        runtime = _runtime_with_context(selected_runtime, data_context)
        result = self._runner.run_review(
            self.recipe,
            review,
            initial_artifacts=initial,
            previous_state=previous,
            allow_empty_initial_state=(
                self._allow_empty_initial_state
                and self._last_result is None
                and previous is None
            ),
            runtime=runtime,
        )
        self._last_result = result

        final_constituents = result.artifacts.get(CORE_FINAL_CONSTITUENTS)
        if final_constituents is not None:
            data_context.set_dataframe(
                _as_constituent_frame(final_constituents.value)
            )
        weights = result.target_weights.to_frame("index_weight")
        weights.index.name = "instrument_id"
        data_context.set_dataframe(weights, columns=["index_weight"])
        daily = result.artifacts.get(CORE_DAILY_DATA)
        if daily is not None:
            data_context.daily = deepcopy(daily.value)
        methodology_diagnostics = result.artifacts.get(CORE_DIAGNOSTICS)
        if methodology_diagnostics is not None and isinstance(
            methodology_diagnostics.value, Mapping
        ):
            data_context.diagnostics.update(deepcopy(methodology_diagnostics.value))
        data_context.diagnostics["index_recipe"] = {
            "recipe_id": self.recipe.recipe_id,
            "recipe_version": self.recipe.recipe_version,
            "recipe_digest": result.recipe_digest,
            "stages": [
                {
                    "node_id": item.node_id,
                    "stage_kind": item.stage_kind,
                    "stage_version": item.stage_version,
                    "implementation_digest": item.implementation_digest,
                    "cache_source": item.cache_source.value,
                    "cache_key": item.cache_key,
                    "elapsed_seconds": item.elapsed_seconds,
                    "input_digests": dict(item.input_digests),
                    "output_digests": dict(item.output_digests),
                    "random_seed": item.random_seed,
                }
                for item in result.stages
            ],
        }
        return data_context


def _context_for_stage(inputs: StageInputs, runtime: StageRuntime) -> DataContext:
    template = runtime.services.get("data_context")
    if template is None:
        context = DataContext()
    elif isinstance(template, DataContext):
        context = template.copy()
    else:
        raise TypeError("data_context runtime service must be a DataContext")
    context.reference_date = pd.Timestamp(inputs.review.reference_date).normalize()
    context.effective_date = pd.Timestamp(inputs.review.effective_date).normalize()
    context.index_id = inputs.review.index_id
    context.universe_id = inputs.review.universe_id
    constituents = inputs.artifacts.get(CORE_CONSTITUENTS)
    if constituents is not None:
        incoming = _as_constituent_frame(constituents.value)
        if context.cons.empty:
            context.set_dataframe(incoming)
        else:
            context.set_dataframe(incoming)
    daily = inputs.artifacts.get(CORE_DAILY_DATA)
    if daily is not None:
        context.daily = deepcopy(daily.value)
    return context


def _runtime_with_context(runtime: StageRuntime, context: DataContext) -> StageRuntime:
    services = dict(runtime.services)
    services["data_context"] = context
    return StageRuntime(
        providers=runtime.providers,
        provider_parameters=runtime.provider_parameters,
        provider_revisions=runtime.provider_revisions,
        services=services,
        data_revision=runtime.data_revision,
        code_revision=runtime.code_revision,
        random_seed_revision=runtime.random_seed_revision,
        random_seed=runtime.random_seed,
        run_id=runtime.run_id,
    )


def _as_constituent_frame(value: Any) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError("constituent artifacts must contain a pandas DataFrame")
    frame = value.copy(deep=True)
    if frame.index.name != "instrument_id":
        if "instrument_id" not in frame.columns:
            raise ValueError(
                "constituent artifact must use instrument_id as index or column"
            )
        frame = frame.set_index("instrument_id", verify_integrity=True)
    if not frame.index.is_unique:
        raise ValueError("constituent artifact contains duplicate instrument_id values")
    return frame


__all__ = [
    "MethodologyExecutionStage",
    "RecipeWeightProducer",
    "RuleExecutionStage",
]
