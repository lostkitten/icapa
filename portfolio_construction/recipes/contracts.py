"""Stage contracts shared by recipe graphs and execution runtimes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from .artifacts import (
    Artifact,
    ArtifactKey,
    ArtifactOutput,
    ArtifactRequirement,
    JsonValue,
    canonicalize,
    qualified_name,
)


class RecipeError(RuntimeError):
    """Base class for recipe compilation and execution errors."""


class RecipeCompilationError(RecipeError):
    """Raised when a recipe graph or stage contract is invalid."""


class StageExecutionError(RecipeError):
    """Raised when a stage fails its declared execution contract."""


class PriorReviewStateError(RecipeError):
    """Raised when a state-dependent stage cannot resolve prior artifacts."""


class StageCacheScope(StrEnum):
    """Scope in which a deterministic stage result may be reused."""

    CONTENT = "content"
    RECIPE = "recipe"
    RUN = "run"
    DISABLED = "disabled"


class StageSideEffect(StrEnum):
    """Declared side-effect class for scheduling and review."""

    PURE = "pure"
    READ_ONLY_IO = "read_only_io"
    OPAQUE = "opaque"


class PriorStatePolicy(StrEnum):
    """Behavior when a requested prior-review artifact is unavailable."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    EMPTY_INITIAL = "empty_initial"


class StageCacheSource(StrEnum):
    """How a stage result was obtained."""

    EXECUTED = "executed"
    CACHE = "cache"


REVIEW_DIMENSIONS = frozenset(
    {"index_id", "universe_id", "reference_date", "effective_date"}
)


@dataclass(frozen=True)
class ReviewIdentity:
    """Canonical identity of one index review."""

    index_id: str
    reference_date: object
    effective_date: object
    universe_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.index_id, str) or not self.index_id.strip():
            raise ValueError("index_id must be a non-empty string")
        reference = pd.Timestamp(self.reference_date).normalize()
        effective = pd.Timestamp(self.effective_date).normalize()
        if pd.isna(reference) or pd.isna(effective):
            raise ValueError("review dates must not be null")
        if reference > effective:
            raise ValueError("reference_date must not be after effective_date")
        object.__setattr__(self, "reference_date", reference)
        object.__setattr__(self, "effective_date", effective)

    def selected_dimensions(self, dimensions: Iterable[str]) -> dict[str, JsonValue]:
        """Return the review fields explicitly used by a stage cache key."""

        requested = frozenset(dimensions)
        unknown = requested - REVIEW_DIMENSIONS
        if unknown:
            raise ValueError(f"unknown review dimensions: {sorted(unknown)}")
        return {
            name: canonicalize(getattr(self, name))
            for name in sorted(requested)
        }


@dataclass(frozen=True)
class PriorArtifactRequirement:
    """Declare one artifact read from the previous review."""

    key: ArtifactKey
    policy: PriorStatePolicy = PriorStatePolicy.REQUIRED

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", PriorStatePolicy(self.policy))


@dataclass(frozen=True)
class ProviderRequestSpec:
    """Describe the immutable request made to one recipe provider."""

    capability: str
    review_dimensions: frozenset[str] = frozenset()
    include_provider_parameters: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.capability, str) or not self.capability.strip():
            raise ValueError("provider request capability must be a non-empty string")
        dimensions = frozenset(self.review_dimensions)
        unknown = dimensions - REVIEW_DIMENSIONS
        if unknown:
            raise ValueError(
                f"unknown provider request review dimensions: {sorted(unknown)}"
            )
        if not isinstance(self.include_provider_parameters, bool):
            raise TypeError("include_provider_parameters must be a bool")
        object.__setattr__(self, "capability", self.capability.strip())
        object.__setattr__(self, "review_dimensions", dimensions)

    def build_request(
        self,
        review: ReviewIdentity,
        provider_parameters: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build the exact keyword mapping declared for one provider call."""

        request = (
            dict(provider_parameters)
            if self.include_provider_parameters
            else {}
        )
        overlap = set(request).intersection(self.review_dimensions)
        if overlap:
            raise ValueError(
                "provider parameters must not override declared review "
                f"dimensions: {sorted(overlap)}"
            )
        request.update(
            {
                name: getattr(review, name)
                for name in sorted(self.review_dimensions)
            }
        )
        return request


@dataclass(frozen=True)
class StageRequirements:
    """Inputs, prior state, providers, and review dimensions used by a stage."""

    artifacts: tuple[ArtifactRequirement, ...] = ()
    prior_artifacts: tuple[PriorArtifactRequirement, ...] = ()
    provider_capabilities: tuple[str, ...] = ()
    provider_requests: tuple[ProviderRequestSpec, ...] = ()
    review_dimensions: frozenset[str] = REVIEW_DIMENSIONS
    consume_all_current: bool = False
    consume_all_previous: bool = False

    def __post_init__(self) -> None:
        dimensions = frozenset(self.review_dimensions)
        unknown = dimensions - REVIEW_DIMENSIONS
        if unknown:
            raise ValueError(f"unknown review dimensions: {sorted(unknown)}")
        providers = tuple(self.provider_capabilities)
        if any(not isinstance(item, str) or not item.strip() for item in providers):
            raise ValueError("provider capabilities must be non-empty strings")
        if len(set(providers)) != len(providers):
            raise ValueError("provider capabilities must not contain duplicates")
        requests = tuple(self.provider_requests)
        if any(not isinstance(item, ProviderRequestSpec) for item in requests):
            raise TypeError(
                "provider_requests must contain ProviderRequestSpec values"
            )
        request_capabilities = [item.capability for item in requests]
        if len(set(request_capabilities)) != len(request_capabilities):
            raise ValueError(
                "provider request capabilities must not contain duplicates"
            )
        undeclared = set(request_capabilities) - set(providers)
        if undeclared:
            raise ValueError(
                "provider requests require matching provider capabilities: "
                f"{sorted(undeclared)}"
            )
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "prior_artifacts", tuple(self.prior_artifacts))
        object.__setattr__(self, "provider_capabilities", providers)
        object.__setattr__(self, "provider_requests", requests)
        object.__setattr__(self, "review_dimensions", dimensions)


@dataclass(frozen=True)
class StageDescriptor:
    """Stable implementation identity and execution behavior."""

    kind: str
    version: str = "1"
    deterministic: bool = True
    cache_scope: StageCacheScope = StageCacheScope.CONTENT
    side_effect: StageSideEffect = StageSideEffect.PURE
    parallel_safe: bool = True
    allows_dynamic_outputs: bool = False
    uses_randomness: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("stage kind must be a non-empty string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("stage version must be a non-empty string")
        object.__setattr__(self, "cache_scope", StageCacheScope(self.cache_scope))
        object.__setattr__(self, "side_effect", StageSideEffect(self.side_effect))
        if not self.deterministic and self.cache_scope is not StageCacheScope.DISABLED:
            raise ValueError("non-deterministic stages must disable caching")


@dataclass(frozen=True)
class StageDiagnostic:
    """Compact structured diagnostic emitted by a stage."""

    code: str
    message: str = ""
    severity: str = "info"
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("diagnostic code must be a non-empty string")
        if self.severity not in {"info", "warning", "error"}:
            raise ValueError("diagnostic severity must be info, warning, or error")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


@dataclass(frozen=True)
class StageInputs:
    """Read-only artifacts and review identity supplied to a stage."""

    review: ReviewIdentity
    artifacts: Mapping[ArtifactKey, Artifact]
    prior_artifacts: Mapping[ArtifactKey, Artifact] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))
        object.__setattr__(
            self,
            "prior_artifacts",
            MappingProxyType(dict(self.prior_artifacts)),
        )

    def value(self, key: ArtifactKey) -> Any:
        """Return the current-review artifact value for ``key``."""

        return self.artifacts[key].value

    def prior_value(self, key: ArtifactKey) -> Any:
        """Return the prior-review artifact value for ``key``."""

        return self.prior_artifacts[key].value


@dataclass(frozen=True)
class StageResult:
    """Artifacts and domain diagnostics returned by one stage."""

    artifacts: Mapping[ArtifactKey, Artifact]
    diagnostics: tuple[StageDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        artifacts = dict(self.artifacts)
        for key, artifact in artifacts.items():
            if not isinstance(key, ArtifactKey) or not isinstance(artifact, Artifact):
                raise TypeError("stage artifacts must map ArtifactKey to Artifact")
            if artifact.key != key:
                raise ValueError(
                    f"stage artifact mapping key {key} does not match artifact key "
                    f"{artifact.key}"
                )
        object.__setattr__(self, "artifacts", MappingProxyType(artifacts))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


@dataclass(frozen=True)
class StageRuntime:
    """Run-scoped services and safe revision identities available to stages."""

    providers: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    provider_parameters: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    provider_revisions: Mapping[str, Any] = field(default_factory=dict)
    services: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    data_revision: Any = "unversioned"
    code_revision: Any = "unversioned"
    random_seed_revision: Any = "unversioned"
    random_seed: int | None = None
    run_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "providers", MappingProxyType(dict(self.providers)))
        object.__setattr__(
            self,
            "provider_parameters",
            MappingProxyType(
                {
                    str(capability): MappingProxyType(dict(parameters))
                    for capability, parameters in self.provider_parameters.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "provider_revisions",
            MappingProxyType(dict(self.provider_revisions)),
        )
        object.__setattr__(self, "services", MappingProxyType(dict(self.services)))
        if self.random_seed is not None and not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be an integer or None")


@runtime_checkable
class IndexStage(Protocol):
    """Structural contract implemented by native and private recipe stages."""

    @property
    def descriptor(self) -> StageDescriptor: ...

    @property
    def requirements(self) -> StageRequirements: ...

    @property
    def outputs(self) -> tuple[ArtifactOutput, ...]: ...

    def canonical_configuration(self) -> JsonValue: ...

    def run(self, inputs: StageInputs, runtime: StageRuntime) -> StageResult: ...


@dataclass(frozen=True)
class CallableStage:
    """Adapt an arbitrary callable to the canonical stage contract."""

    function: Callable[[StageInputs, StageRuntime], StageResult | Mapping]
    output_specs: tuple[ArtifactOutput, ...]
    input_requirements: StageRequirements = StageRequirements()
    configuration: Mapping[str, Any] = field(default_factory=dict)
    kind: str | None = None
    version: str = "1"
    cache_scope: StageCacheScope = StageCacheScope.DISABLED
    deterministic: bool = True
    side_effect: StageSideEffect = StageSideEffect.OPAQUE
    parallel_safe: bool = False
    allows_dynamic_outputs: bool = False
    uses_randomness: bool = False

    @property
    def descriptor(self) -> StageDescriptor:
        return StageDescriptor(
            kind=self.kind or qualified_name(self.function),
            version=self.version,
            deterministic=self.deterministic,
            cache_scope=self.cache_scope,
            side_effect=self.side_effect,
            parallel_safe=self.parallel_safe,
            allows_dynamic_outputs=self.allows_dynamic_outputs,
            uses_randomness=self.uses_randomness,
        )

    @property
    def requirements(self) -> StageRequirements:
        return self.input_requirements

    @property
    def outputs(self) -> tuple[ArtifactOutput, ...]:
        return tuple(self.output_specs)

    def canonical_configuration(self) -> JsonValue:
        return {
            "function": qualified_name(self.function),
            "configuration": canonicalize(self.configuration),
        }

    def run(self, inputs: StageInputs, runtime: StageRuntime) -> StageResult:
        raw = self.function(inputs, runtime)
        if isinstance(raw, StageResult):
            return raw
        if not isinstance(raw, Mapping):
            raise TypeError("callable stages must return StageResult or a mapping")
        artifacts: dict[ArtifactKey, Artifact] = {}
        for key, value in raw.items():
            if not isinstance(key, ArtifactKey):
                raise TypeError("callable stage output keys must be ArtifactKey instances")
            artifacts[key] = (
                value if isinstance(value, Artifact) else Artifact.from_value(key, value)
            )
        return StageResult(artifacts)


__all__ = [
    "CallableStage",
    "IndexStage",
    "PriorArtifactRequirement",
    "PriorReviewStateError",
    "PriorStatePolicy",
    "ProviderRequestSpec",
    "RecipeCompilationError",
    "RecipeError",
    "ReviewIdentity",
    "StageCacheScope",
    "StageCacheSource",
    "StageDescriptor",
    "StageDiagnostic",
    "StageExecutionError",
    "StageInputs",
    "StageRequirements",
    "StageResult",
    "StageRuntime",
    "StageSideEffect",
]
