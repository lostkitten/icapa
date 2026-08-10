"""Stage contracts shared by recipe graphs and execution runtimes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import InitVar, dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from icapa.data_sources.provenance import private_parameter_digest

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
class _FrozenParameterValue:
    """Type-tagged value retained inside immutable parameter mappings."""

    kind: str
    payload: Any


class _FrozenParameterMapping(Mapping[str, Any]):
    """Deeply immutable, hashable provider-request parameter mapping."""

    __slots__ = ("_items", "_lookup")

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        selected = dict(values or {})
        if any(not isinstance(key, str) for key in selected):
            raise TypeError("provider parameter mappings require string keys")
        private_parameter_digest(selected)
        items = tuple(
            (key, _freeze_parameter_value(value))
            for key, value in sorted(selected.items())
        )
        self._items = items
        self._lookup = dict(items)

    def __getitem__(self, key: str) -> Any:
        return _immutable_parameter_view(self._lookup[key])

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _FrozenParameterMapping):
            return self._items == other._items
        if isinstance(other, Mapping):
            try:
                return self._items == _FrozenParameterMapping(other)._items
            except (TypeError, ValueError):
                return False
        return False

    def __repr__(self) -> str:
        return repr(_thaw_parameter_value(self))


def _freeze_parameter_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenParameterMapping(value)
    if isinstance(value, list):
        return _FrozenParameterValue(
            "list",
            tuple(_freeze_parameter_value(item) for item in value),
        )
    if isinstance(value, tuple):
        return _FrozenParameterValue(
            "tuple",
            tuple(_freeze_parameter_value(item) for item in value),
        )
    if isinstance(value, set):
        return _FrozenParameterValue(
            "set",
            frozenset(_freeze_parameter_value(item) for item in value),
        )
    if isinstance(value, frozenset):
        return _FrozenParameterValue(
            "frozenset",
            frozenset(_freeze_parameter_value(item) for item in value),
        )
    if isinstance(value, bytearray):
        return _FrozenParameterValue("bytearray", bytes(value))
    return _FrozenParameterValue(
        f"{type(value).__module__}.{type(value).__qualname__}",
        value,
    )


def _thaw_parameter_value(value: Any) -> Any:
    if isinstance(value, _FrozenParameterMapping):
        return {
            key: _thaw_parameter_value(item)
            for key, item in value._items
        }
    if isinstance(value, _FrozenParameterValue):
        if value.kind == "list":
            return [_thaw_parameter_value(item) for item in value.payload]
        if value.kind == "tuple":
            return tuple(
                _thaw_parameter_value(item) for item in value.payload
            )
        if value.kind == "set":
            return {
                _thaw_parameter_value(item) for item in value.payload
            }
        if value.kind == "frozenset":
            return frozenset(
                _thaw_parameter_value(item) for item in value.payload
            )
        if value.kind == "bytearray":
            return bytearray(value.payload)
        return value.payload
    return value


def _immutable_parameter_view(value: Any) -> Any:
    if isinstance(value, _FrozenParameterMapping):
        return value
    if isinstance(value, _FrozenParameterValue):
        if value.kind in {"list", "tuple"}:
            return tuple(
                _immutable_parameter_view(item) for item in value.payload
            )
        if value.kind in {"set", "frozenset"}:
            return frozenset(
                _immutable_parameter_view(item) for item in value.payload
            )
        if value.kind == "bytearray":
            return bytes(value.payload)
        return value.payload
    return value


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
    """Describe one immutable provider snapshot request.

    ``request_parameters`` contains fixed, non-secret request arguments.
    Provider binding parameters are merged either at the request top level or
    beneath ``provider_parameters_key``.  The optional expected provider name
    and parameter digest prevent a recipe from fingerprinting one provider
    binding while its implementation reads from another. A request marked
    ``covers_all_instruments`` uses a provider-acknowledged dataset snapshot;
    cache preflight requires it to be paired with an exact universe request.
    """

    capability: str
    review_dimensions: frozenset[str] = frozenset()
    include_provider_parameters: bool = True
    request_parameters: Mapping[str, Any] = field(default_factory=dict)
    provider_parameters_key: str | None = None
    review_parameter_map: Mapping[str, str] = field(default_factory=dict)
    expected_provider_name: str | None = None
    expected_provider_parameters_digest: str | None = None
    covers_all_instruments: bool = False
    expected_provider_parameters: InitVar[Mapping[str, Any] | None] = None
    _declared_provider_parameters: Mapping[str, Any] | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(
        self,
        expected_provider_parameters: Mapping[str, Any] | None,
    ) -> None:
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
        request_parameters = _FrozenParameterMapping(
            self.request_parameters
        )
        if any(
            not isinstance(name, str) or not name.strip()
            for name in request_parameters
        ):
            raise ValueError(
                "provider request parameter names must be non-empty strings"
            )
        provider_parameters_key = self.provider_parameters_key
        if provider_parameters_key is not None:
            if (
                not isinstance(provider_parameters_key, str)
                or not provider_parameters_key.strip()
            ):
                raise ValueError(
                    "provider_parameters_key must be a non-empty string or None"
                )
            if not self.include_provider_parameters:
                raise ValueError(
                    "provider_parameters_key requires include_provider_parameters=True"
                )
            provider_parameters_key = provider_parameters_key.strip()
        review_parameter_map = _FrozenParameterMapping(
            self.review_parameter_map
        )
        if any(
            not isinstance(name, str) or not name.strip()
            for name in review_parameter_map
        ):
            raise ValueError("review parameter names must be non-empty strings")
        unknown_mapped = set(review_parameter_map.values()) - REVIEW_DIMENSIONS
        if unknown_mapped:
            raise ValueError(
                "unknown mapped provider request review dimensions: "
                f"{sorted(unknown_mapped)}"
            )
        expected_provider_name = self.expected_provider_name
        if expected_provider_name is not None:
            if (
                not isinstance(expected_provider_name, str)
                or not expected_provider_name.strip()
            ):
                raise ValueError(
                    "expected_provider_name must be a non-empty string or None"
                )
            expected_provider_name = expected_provider_name.strip().lower()
        expected_digest = self.expected_provider_parameters_digest
        declared_parameters = (
            None
            if expected_provider_parameters is None
            else _FrozenParameterMapping(expected_provider_parameters)
        )
        if declared_parameters is not None:
            declared_digest = private_parameter_digest(
                _thaw_parameter_value(declared_parameters)
            )
            if expected_digest is not None and expected_digest != declared_digest:
                raise ValueError(
                    "expected provider parameter mapping does not match its digest"
                )
            expected_digest = declared_digest
        if expected_digest is not None:
            if not isinstance(expected_digest, str) or len(expected_digest) != 64:
                raise ValueError(
                    "expected_provider_parameters_digest must be a SHA-256 hex digest"
                )
            try:
                int(expected_digest, 16)
            except ValueError as exc:
                raise ValueError(
                    "expected_provider_parameters_digest must be a SHA-256 hex digest"
                ) from exc
            expected_digest = expected_digest.lower()
        if not isinstance(self.covers_all_instruments, bool):
            raise TypeError("covers_all_instruments must be a bool")
        if self.covers_all_instruments and "instrument_scope" in request_parameters:
            raise ValueError(
                "request_parameters must not override dataset instrument scope"
            )
        object.__setattr__(self, "capability", self.capability.strip())
        object.__setattr__(self, "review_dimensions", dimensions)
        object.__setattr__(
            self,
            "request_parameters",
            request_parameters,
        )
        object.__setattr__(
            self,
            "provider_parameters_key",
            provider_parameters_key,
        )
        object.__setattr__(
            self,
            "review_parameter_map",
            review_parameter_map,
        )
        object.__setattr__(self, "expected_provider_name", expected_provider_name)
        object.__setattr__(
            self,
            "expected_provider_parameters_digest",
            expected_digest,
        )
        object.__setattr__(
            self,
            "_declared_provider_parameters",
            (
                None
                if declared_parameters is None
                else declared_parameters
            ),
        )

    @property
    def declared_provider_parameters(self) -> Mapping[str, Any] | None:
        """Return in-memory parameters for manifest/provider alignment."""

        if self._declared_provider_parameters is None:
            return None
        return _thaw_parameter_value(self._declared_provider_parameters)

    def validate_binding(
        self,
        provider_name: str,
        provider_parameters: Mapping[str, Any],
    ) -> None:
        """Fail when the runtime binding differs from the declared reader."""

        selected_name = str(provider_name).strip().lower()
        if (
            self.expected_provider_name is not None
            and selected_name != self.expected_provider_name
        ):
            raise ValueError(
                f"provider request {self.capability!r} expects provider "
                f"{self.expected_provider_name!r}, not {selected_name!r}"
            )
        self._validate_provider_parameters(provider_parameters)

    def _validate_provider_parameters(
        self,
        provider_parameters: Mapping[str, Any],
    ) -> None:
        if self.expected_provider_parameters_digest is None:
            return
        actual_digest = private_parameter_digest(dict(provider_parameters))
        if actual_digest != self.expected_provider_parameters_digest:
            raise ValueError(
                f"provider request {self.capability!r} binding parameters "
                "do not match the parameters used by the recipe stage"
            )

    def build_request(
        self,
        review: ReviewIdentity,
        provider_parameters: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build the fixed and review-bound provider snapshot request."""

        parameters = dict(provider_parameters)
        self._validate_provider_parameters(parameters)
        request = _thaw_parameter_value(self.request_parameters)
        if self.covers_all_instruments:
            request["instrument_scope"] = "all_instruments"
        if self.include_provider_parameters:
            if self.provider_parameters_key is None:
                overlap = set(request).intersection(parameters)
                if overlap:
                    raise ValueError(
                        "fixed request parameters must not overlap provider "
                        f"parameters: {sorted(overlap)}"
                    )
                request.update(parameters)
            else:
                if self.provider_parameters_key in request:
                    raise ValueError(
                        "fixed request parameters must not contain the provider "
                        f"parameter container {self.provider_parameters_key!r}"
                    )
                request[self.provider_parameters_key] = parameters
        review_values = {
            name: getattr(review, name)
            for name in sorted(self.review_dimensions)
        }
        for parameter_name, dimension_name in sorted(
            self.review_parameter_map.items()
        ):
            if parameter_name in review_values:
                raise ValueError(
                    f"duplicate review-bound provider request parameter: {parameter_name!r}"
                )
            review_values[parameter_name] = getattr(review, dimension_name)
        overlap = set(request).intersection(review_values)
        if overlap:
            raise ValueError(
                "provider request parameters must not override review-bound "
                f"parameters: {sorted(overlap)}"
            )
        request.update(review_values)
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
        for position, request in enumerate(requests):
            if request in requests[:position]:
                raise ValueError("provider requests must not contain duplicates")
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
