"""State-aware execution for compiled index recipes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from threading import RLock
from time import perf_counter
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

import numpy as np
import pandas as pd

from .artifacts import Artifact, ArtifactKey, canonical_digest, canonicalize
from .contracts import (
    IndexStage,
    PriorReviewStateError,
    PriorStatePolicy,
    ReviewIdentity,
    StageCacheScope,
    StageCacheSource,
    StageDiagnostic,
    StageExecutionError,
    StageInputs,
    StageResult,
    StageRuntime,
)
from .graph import (
    CompiledStageNode,
    ExecutionPlan,
    IndexRecipe,
    RecipeCompiler,
    StageNode,
)


@runtime_checkable
class StageCache(Protocol):
    """Storage interface for immutable stage results."""

    def load(self, key: str) -> StageResult | None: ...

    def save(self, key: str, result: StageResult) -> None: ...


class MemoryStageCache:
    """Thread-safe process-local stage cache used by tests and notebooks."""

    def __init__(self) -> None:
        self._results: dict[str, StageResult] = {}
        self._lock = RLock()

    def load(self, key: str) -> StageResult | None:
        with self._lock:
            result = self._results.get(key)
            return None if result is None else _copy_stage_result(result)

    def save(self, key: str, result: StageResult) -> None:
        with self._lock:
            self._results[key] = _copy_stage_result(result)

    def clear(self) -> None:
        with self._lock:
            self._results.clear()


@dataclass(frozen=True)
class PreviousReviewState:
    """Immutable artifacts made available to a later review."""

    review: ReviewIdentity
    artifacts: Mapping[ArtifactKey, Artifact]

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))


@dataclass(frozen=True)
class StageExecutionRecord:
    """Execution and cache provenance for one recipe node."""

    node_id: str
    stage_kind: str
    stage_version: str
    implementation_digest: str | None
    cache_source: StageCacheSource
    cache_key: str | None
    elapsed_seconds: float
    input_digests: Mapping[str, str]
    output_digests: Mapping[str, str]
    diagnostics: tuple[StageDiagnostic, ...] = ()
    random_seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_source", StageCacheSource(self.cache_source))
        object.__setattr__(
            self,
            "input_digests",
            MappingProxyType(dict(self.input_digests)),
        )
        object.__setattr__(
            self,
            "output_digests",
            MappingProxyType(dict(self.output_digests)),
        )
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


@dataclass(frozen=True)
class ReviewConstructionResult:
    """Final artifacts and stage provenance for one review."""

    review: ReviewIdentity
    recipe_digest: str
    final_weights_key: ArtifactKey
    artifacts: Mapping[ArtifactKey, Artifact]
    stages: tuple[StageExecutionRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))
        object.__setattr__(self, "stages", tuple(self.stages))

    @property
    def target_weights(self) -> pd.Series:
        """Return a defensive copy of the canonical final weights."""

        return extract_weights(
            self.artifacts[self.final_weights_key].value
        ).copy()

    def as_previous_state(self) -> PreviousReviewState:
        """Expose this review's artifacts to explicitly stateful later stages."""

        return PreviousReviewState(self.review, self.artifacts)


class RecipeRunner:
    """Execute compiled recipes with explicit artifacts, state, and caching."""

    def __init__(
        self,
        *,
        runtime: StageRuntime | None = None,
        cache: StageCache | None = None,
        compiler: RecipeCompiler | None = None,
    ) -> None:
        self.runtime = runtime or StageRuntime()
        self.cache = cache
        self.compiler = compiler or RecipeCompiler()
        self._run_token = self.runtime.run_id or uuid4().hex

    def run_review(
        self,
        recipe: IndexRecipe | ExecutionPlan,
        review: ReviewIdentity,
        *,
        initial_artifacts: Mapping[ArtifactKey, Artifact | Any] | None = None,
        previous_state: PreviousReviewState | None = None,
        allow_empty_initial_state: bool = False,
        runtime: StageRuntime | None = None,
    ) -> ReviewConstructionResult:
        """Execute one review and validate its final target weights."""

        plan = (
            recipe
            if isinstance(recipe, ExecutionPlan)
            else self.compiler.compile(
                recipe,
                allow_unfingerprintable=self.cache is None,
            )
        )
        active_runtime = self._resolved_runtime(plan, runtime or self.runtime)
        if previous_state is not None and (
            previous_state.review.effective_date >= review.effective_date
        ):
            raise PriorReviewStateError(
                "previous review effective_date must be before the current review"
            )
        artifacts = normalize_artifacts(initial_artifacts or {})
        missing_initial = set(plan.recipe.required_artifacts) - set(artifacts)
        if missing_initial:
            raise StageExecutionError(
                "recipe is missing required initial artifacts: "
                + ", ".join(sorted(map(str, missing_initial)))
            )

        records: list[StageExecutionRecord] = []
        for compiled in plan.nodes:
            record, result = self._run_stage(
                plan=plan,
                compiled=compiled,
                review=review,
                artifacts=artifacts,
                previous_state=previous_state,
                allow_empty_initial_state=allow_empty_initial_state,
                runtime=active_runtime,
            )
            collisions = set(result.artifacts) & set(artifacts)
            if collisions:
                raise StageExecutionError(
                    f"stage {compiled.node.node_id!r} attempted to replace "
                    f"existing artifacts: {sorted(map(str, collisions))}"
                )
            artifacts.update(result.artifacts)
            records.append(record)

        if plan.recipe.final_weights not in artifacts:
            raise StageExecutionError("recipe did not produce its final weight artifact")
        if plan.recipe.validate_final_weights:
            validate_weights(artifacts[plan.recipe.final_weights].value)
        return ReviewConstructionResult(
            review=review,
            recipe_digest=plan.recipe_digest,
            final_weights_key=plan.recipe.final_weights,
            artifacts=artifacts,
            stages=tuple(records),
        )

    def run_sequence(
        self,
        recipe: IndexRecipe | ExecutionPlan,
        reviews: Sequence[ReviewIdentity],
        *,
        initial_artifacts: Mapping[
            ReviewIdentity, Mapping[ArtifactKey, Artifact | Any]
        ]
        | None = None,
        initial_previous_state: PreviousReviewState | None = None,
        allow_empty_initial_state: bool = False,
        runtime: StageRuntime | None = None,
    ) -> dict[pd.Timestamp, ReviewConstructionResult]:
        """Execute reviews in effective-date order with explicit prior state."""

        ordered = sorted(reviews, key=lambda item: item.effective_date)
        effective_dates = [item.effective_date for item in ordered]
        if len(set(effective_dates)) != len(effective_dates):
            raise ValueError("review sequence contains duplicate effective_date values")
        supplied = initial_artifacts or {}
        previous = initial_previous_state
        results: dict[pd.Timestamp, ReviewConstructionResult] = {}
        for position, review in enumerate(ordered):
            result = self.run_review(
                recipe,
                review,
                initial_artifacts=supplied.get(review),
                previous_state=previous,
                allow_empty_initial_state=(
                    allow_empty_initial_state
                    and position == 0
                    and previous is None
                ),
                runtime=runtime,
            )
            results[review.effective_date] = result
            previous = result.as_previous_state()
        return results

    def _run_stage(
        self,
        *,
        plan: ExecutionPlan,
        compiled: CompiledStageNode,
        review: ReviewIdentity,
        artifacts: Mapping[ArtifactKey, Artifact],
        previous_state: PreviousReviewState | None,
        allow_empty_initial_state: bool,
        runtime: StageRuntime,
    ) -> tuple[StageExecutionRecord, StageResult]:
        node = compiled.node
        missing_providers = set(node.stage.requirements.provider_capabilities) - set(
            runtime.providers
        )
        if missing_providers:
            raise StageExecutionError(
                f"node {node.node_id!r} is missing provider capabilities: "
                f"{sorted(missing_providers)}"
            )
        current = self._resolve_current(node, artifacts)
        prior = self._resolve_prior(
            node,
            previous_state,
            allow_empty_initial_state=allow_empty_initial_state,
        )
        inputs = StageInputs(
            review=review,
            artifacts=current,
            prior_artifacts=prior,
        )
        cache_key = self._cache_key(plan, compiled, inputs, runtime)
        result = None
        source = StageCacheSource.EXECUTED
        started = perf_counter()
        if cache_key is not None and self.cache is not None:
            result = self.cache.load(cache_key)
            if result is not None:
                source = StageCacheSource.CACHE
        if result is None:
            try:
                result = node.stage.run(inputs, runtime)
            except Exception as exc:
                raise StageExecutionError(
                    f"stage {node.node_id!r} ({node.stage.descriptor.kind}) failed"
                ) from exc
            if not isinstance(result, StageResult):
                raise StageExecutionError(
                    f"stage {node.node_id!r} did not return StageResult"
                )
            self._validate_stage_outputs(node, result)
            if cache_key is not None and self.cache is not None:
                self.cache.save(cache_key, result)
        else:
            self._validate_stage_outputs(node, result)
        elapsed = perf_counter() - started
        input_digests = {
            f"current:{key.canonical_name}": artifact.digest
            for key, artifact in current.items()
        }
        input_digests.update(
            {
                f"prior:{key.canonical_name}": artifact.digest
                for key, artifact in prior.items()
            }
        )
        return (
            StageExecutionRecord(
                node_id=node.node_id,
                stage_kind=node.stage.descriptor.kind,
                stage_version=node.stage.descriptor.version,
                implementation_digest=compiled.implementation_digest,
                cache_source=source,
                cache_key=cache_key,
                elapsed_seconds=elapsed,
                input_digests=input_digests,
                output_digests={
                    key.canonical_name: artifact.digest
                    for key, artifact in result.artifacts.items()
                },
                diagnostics=result.diagnostics,
                random_seed=runtime.random_seed,
            ),
            result,
        )

    @staticmethod
    def _resolved_runtime(
        plan: ExecutionPlan,
        runtime: StageRuntime,
    ) -> StageRuntime:
        if runtime.random_seed is not None and runtime.code_revision != "unversioned":
            return runtime
        seed_identity = plan.recipe_digest
        if runtime.random_seed_revision != "unversioned":
            seed_identity = canonical_digest(
                {
                    "definition_fingerprint": canonicalize(
                        runtime.random_seed_revision
                    )
                }
            )
        elif runtime.code_revision != "unversioned":
            seed_identity = canonical_digest(
                {
                    "code_revision": canonicalize(runtime.code_revision),
                    "recipe_digest": plan.recipe_digest,
                }
            )
        return replace(
            runtime,
            random_seed=(
                runtime.random_seed
                if runtime.random_seed is not None
                else int(seed_identity[:16], 16) % (2**32)
            ),
            code_revision=(
                runtime.code_revision
                if runtime.code_revision != "unversioned"
                else plan.recipe_digest
            ),
        )

    @staticmethod
    def _resolve_current(
        node: StageNode,
        artifacts: Mapping[ArtifactKey, Artifact],
    ) -> dict[ArtifactKey, Artifact]:
        requirements = node.stage.requirements
        if requirements.consume_all_current:
            return dict(artifacts)
        selected: dict[ArtifactKey, Artifact] = {}
        for requirement in requirements.artifacts:
            artifact = artifacts.get(requirement.key)
            if artifact is None:
                if requirement.optional:
                    continue
                raise StageExecutionError(
                    f"node {node.node_id!r} is missing artifact {requirement.key}"
                )
            selected[requirement.key] = artifact
        return selected

    @staticmethod
    def _resolve_prior(
        node: StageNode,
        previous_state: PreviousReviewState | None,
        *,
        allow_empty_initial_state: bool,
    ) -> dict[ArtifactKey, Artifact]:
        requirements = node.stage.requirements
        if requirements.consume_all_previous:
            if previous_state is None:
                if allow_empty_initial_state:
                    return {}
                raise PriorReviewStateError(
                    f"node {node.node_id!r} requires previous-review state"
                )
            return dict(previous_state.artifacts)
        selected: dict[ArtifactKey, Artifact] = {}
        for requirement in requirements.prior_artifacts:
            artifact = (
                None
                if previous_state is None
                else previous_state.artifacts.get(requirement.key)
            )
            if artifact is not None:
                selected[requirement.key] = artifact
                continue
            if requirement.policy is PriorStatePolicy.OPTIONAL:
                continue
            if (
                requirement.policy is PriorStatePolicy.EMPTY_INITIAL
                and previous_state is None
                and allow_empty_initial_state
            ):
                continue
            raise PriorReviewStateError(
                f"node {node.node_id!r} requires prior artifact {requirement.key}"
            )
        return selected

    @staticmethod
    def _validate_stage_outputs(node: StageNode, result: StageResult) -> None:
        outputs = {output.key: output for output in node.stage.outputs}
        missing = {
            key
            for key, output in outputs.items()
            if not output.optional and key not in result.artifacts
        }
        if missing:
            raise StageExecutionError(
                f"stage {node.node_id!r} omitted required outputs: "
                f"{sorted(map(str, missing))}"
            )
        unexpected = set(result.artifacts) - set(outputs)
        if unexpected and not node.stage.descriptor.allows_dynamic_outputs:
            raise StageExecutionError(
                f"stage {node.node_id!r} returned undeclared outputs: "
                f"{sorted(map(str, unexpected))}"
            )

    def _cache_key(
        self,
        plan: ExecutionPlan,
        compiled: CompiledStageNode,
        inputs: StageInputs,
        runtime: StageRuntime,
    ) -> str | None:
        node = compiled.node
        descriptor = node.stage.descriptor
        if descriptor.cache_scope is StageCacheScope.DISABLED:
            return None
        if (
            node.stage.requirements.provider_capabilities
            and runtime.data_revision == "unversioned"
        ):
            return None
        payload: dict[str, Any] = {
            "contract_version": "1",
            "stage": {
                "kind": descriptor.kind,
                "version": descriptor.version,
                "implementation_digest": compiled.implementation_digest,
                "configuration": node.stage.canonical_configuration(),
            },
            "review": inputs.review.selected_dimensions(
                node.stage.requirements.review_dimensions
            ),
            "inputs": {
                key.canonical_name: artifact.digest
                for key, artifact in sorted(inputs.artifacts.items())
            },
            "prior_inputs": {
                key.canonical_name: artifact.digest
                for key, artifact in sorted(inputs.prior_artifacts.items())
            },
            "code_revision": canonicalize(runtime.code_revision),
        }
        capabilities = tuple(node.stage.requirements.provider_capabilities)
        if capabilities:
            payload["data_revision"] = canonicalize(runtime.data_revision)
            payload["provider_revisions"] = canonicalize(
                {
                    capability: runtime.provider_revisions.get(
                        capability,
                        "unversioned",
                    )
                    for capability in capabilities
                }
            )
        if descriptor.uses_randomness:
            payload["random_seed"] = runtime.random_seed
        if descriptor.cache_scope in {StageCacheScope.RECIPE, StageCacheScope.RUN}:
            payload["recipe_digest"] = plan.recipe_digest
        if descriptor.cache_scope is StageCacheScope.RUN:
            payload["run_id"] = self._run_token
        return canonical_digest(payload)


class StageRegistry:
    """Run-scoped stage factory registry for serialized configurations."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[Mapping[str, Any]], IndexStage]] = {}

    def register(
        self,
        kind: str,
        factory: Callable[[Mapping[str, Any]], IndexStage],
        *,
        replace: bool = False,
    ) -> None:
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("stage kind must be a non-empty string")
        if not callable(factory):
            raise TypeError("stage factory must be callable")
        if kind in self._factories and not replace:
            raise KeyError(f"stage factory is already registered: {kind}")
        self._factories[kind] = factory

    def create(self, kind: str, configuration: Mapping[str, Any]) -> IndexStage:
        try:
            factory = self._factories[kind]
        except KeyError as exc:
            raise KeyError(f"stage factory is not registered: {kind}") from exc
        stage = factory(configuration)
        if not isinstance(stage, IndexStage):
            raise TypeError(f"factory for {kind!r} did not return an IndexStage")
        return stage

    def items(self):
        return iter(self._factories.items())


def normalize_artifacts(
    values: Mapping[ArtifactKey, Artifact | Any],
) -> dict[ArtifactKey, Artifact]:
    """Normalize initial values to immutable recipe artifacts."""

    artifacts: dict[ArtifactKey, Artifact] = {}
    for key, value in values.items():
        if not isinstance(key, ArtifactKey):
            raise TypeError("initial artifact keys must be ArtifactKey instances")
        artifact = (
            value
            if isinstance(value, Artifact)
            else Artifact.from_value(key, value)
        )
        if artifact.key != key:
            raise ValueError(f"initial artifact key mismatch for {key}")
        artifacts[key] = artifact
    return artifacts


def _copy_stage_result(result: StageResult) -> StageResult:
    artifacts = {
        key: Artifact(
            key=artifact.key,
            value=deepcopy(artifact.value),
            digest=artifact.digest,
            metadata=dict(artifact.metadata),
        )
        for key, artifact in result.artifacts.items()
    }
    diagnostics = tuple(
        StageDiagnostic(
            code=item.code,
            message=item.message,
            severity=item.severity,
            metrics=dict(item.metrics),
        )
        for item in result.diagnostics
    )
    return StageResult(artifacts=artifacts, diagnostics=diagnostics)


def extract_weights(value: Any) -> pd.Series:
    """Extract canonical index weights from a Series or DataFrame."""

    if isinstance(value, pd.Series):
        weights = value.copy()
    elif isinstance(value, pd.DataFrame):
        if "index_weight" not in value.columns:
            raise StageExecutionError(
                "final weight DataFrame must contain an index_weight column"
            )
        if "instrument_id" in value.columns:
            if value["instrument_id"].duplicated().any():
                raise StageExecutionError(
                    "final weights contain duplicate instrument_id values"
                )
            weights = value.set_index("instrument_id")["index_weight"]
        else:
            weights = value["index_weight"].copy()
    else:
        raise StageExecutionError(
            "final weight artifact must contain a pandas Series or DataFrame"
        )
    if not weights.index.is_unique:
        raise StageExecutionError("final weights must use a unique instrument index")
    return pd.to_numeric(weights, errors="coerce")


def validate_weights(value: Any) -> None:
    """Validate the canonical non-negative, fully invested weight contract."""

    array = extract_weights(value).to_numpy(dtype=float)
    if len(array) == 0:
        raise StageExecutionError("final weights must not be empty")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise StageExecutionError("final weights must be finite and non-negative")
    if not np.isclose(float(array.sum()), 1.0, atol=1e-8, rtol=0.0):
        raise StageExecutionError("final weights must sum to one")


__all__ = [
    "MemoryStageCache",
    "PreviousReviewState",
    "RecipeRunner",
    "ReviewConstructionResult",
    "StageCache",
    "StageExecutionRecord",
    "StageRegistry",
]
