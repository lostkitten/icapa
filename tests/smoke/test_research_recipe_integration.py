"""High-level recipe provider, state, cache, and schedule contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

import icapa.research.runners.identity as workflow_module
from icapa.backtesting import Calendar, RebalanceFrequency
from icapa.data_sources import register_provider, registry
from icapa.data_sources.provenance import private_parameter_digest
from icapa.portfolio_construction import (
    Artifact,
    ArtifactKey,
    ArtifactOutput,
    ArtifactRequirement,
    CallableStage,
    CORE_TARGET_WEIGHTS,
    IndexRecipe,
    PriorArtifactRequirement,
    PriorReviewStateError,
    PriorStatePolicy,
    ProviderRequestSpec,
    ReviewIdentity,
    StageCacheScope,
    StageDescriptor,
    StageNode,
    StageRequirements,
    StageResult,
    StageSideEffect,
)
from icapa.research import (
    IndexDefinition,
    RecipeProviderBinding,
    ResearchSpec,
    ResearchWorkspace,
    UnsafeCacheReuseError,
)
from icapa.workspace import (
    CacheMode,
    CacheOptions,
    CacheSource,
    CacheStage,
    automatic_digest,
    clear_memory_cache,
)
from icapa.workspace.caches.source_identity import (
    private_parameter_scope_digest,
)


_PROVIDER_NAME = "recipe_integration_provider"
_CALENDAR_PROVIDER_NAME = "recipe_integration_calendar_provider"
_STATE = ArtifactKey("test", "review_state")
_FEATURES = ArtifactKey("test", "features")


class _SnapshotUniverseProvider:
    def __init__(self) -> None:
        self._load_calls = 0
        self._snapshot_calls = 0
        self._snapshot_requests = []
        self._load_requests = []

    @property
    def load_calls(self):
        return self._load_calls

    @property
    def snapshot_requests(self):
        return tuple(dict(item) for item in self._snapshot_requests)

    @property
    def load_requests(self):
        return tuple(dict(item) for item in self._load_requests)

    def describe_snapshot(self, *, capability, request):
        self._snapshot_calls += 1
        self._snapshot_requests.append(dict(request))
        return {
            "revision": "recipe-provider-v1",
            "capability": capability,
            "universe_id": request["universe_id"],
            "reference_date": str(
                pd.Timestamp(request["reference_date"]).date()
            ),
            "effective_date": str(
                pd.Timestamp(request["effective_date"]).date()
            ),
        }

    def load_universe(
        self,
        *,
        universe_id,
        reference_date,
        effective_date,
        **kwargs,
    ):
        self._load_calls += 1
        self._load_requests.append(
            {
                **kwargs,
                "universe_id": universe_id,
                "reference_date": reference_date,
                "effective_date": effective_date,
            }
        )
        if universe_id != "RECIPE_UNIVERSE":
            raise ValueError("unexpected universe_id")
        return pd.DataFrame(
            {
                "instrument_id": ["A", "B"],
                "benchmark_weight": [0.4, 0.6],
            }
        )


class _ContentOnlyUniverseProvider:
    def __init__(self) -> None:
        self._load_calls = 0

    @property
    def load_calls(self):
        return self._load_calls

    def load_universe(
        self,
        *,
        universe_id,
        reference_date,
        effective_date,
        **kwargs,
    ):
        del universe_id, reference_date, effective_date, kwargs
        self._load_calls += 1
        return pd.DataFrame(
            {
                "instrument_id": ["A", "B"],
                "benchmark_weight": [0.4, 0.6],
            }
        )


class _AlternativeSnapshotUniverseProvider(_SnapshotUniverseProvider):
    """A distinct adapter implementation returning the same logical data."""


class _CredentialScopedUniverseProvider(_SnapshotUniverseProvider):
    """Expose one public token while credential scope changes the source."""

    def load_universe(
        self,
        *,
        universe_id,
        reference_date,
        effective_date,
        **kwargs,
    ):
        password = kwargs.get("password")
        frame = super().load_universe(
            universe_id=universe_id,
            reference_date=reference_date,
            effective_date=effective_date,
            **kwargs,
        )
        if password == "credential-beta":
            frame["benchmark_weight"] = [0.7, 0.3]
        return frame


class _AlternativeCalendar(Calendar):
    """A distinct calendar implementation with identical supplied dates."""


class _CalendarProvider:
    def load_review_schedule(
        self,
        *,
        calendar_id,
        start_date,
        end_date,
        **kwargs,
    ):
        del start_date, end_date, kwargs
        if calendar_id != "RESEARCH_CALENDAR":
            raise ValueError("unexpected calendar_id")
        return _calendar().dates.loc[
            :,
            ["reference_date", "effective_date"],
        ]


class _AlternativeCalendarProvider(_CalendarProvider):
    """A distinct provider adapter returning the same review schedule."""


@dataclass
class _ProviderFeatureStage:
    calls: ClassVar[int] = 0

    @property
    def descriptor(self):
        return StageDescriptor(
            "test.provider_feature_stage",
            cache_scope=StageCacheScope.CONTENT,
            side_effect=StageSideEffect.READ_ONLY_IO,
        )

    @property
    def requirements(self):
        return StageRequirements(
            provider_capabilities=("load_universe",),
            provider_requests=(
                ProviderRequestSpec(
                    "load_universe",
                    review_dimensions=frozenset(
                        {"reference_date", "effective_date"}
                    ),
                ),
            ),
        )

    @property
    def outputs(self):
        return (ArtifactOutput(_FEATURES),)

    def canonical_configuration(self):
        return {}

    def run(self, inputs, runtime):
        type(self).calls += 1
        provider = runtime.providers["load_universe"]
        parameters = runtime.provider_parameters["load_universe"]
        frame = provider.load_universe(
            reference_date=inputs.review.reference_date,
            effective_date=inputs.review.effective_date,
            **parameters,
        )
        features = frame.set_index("instrument_id").loc[
            :,
            ["benchmark_weight"],
        ]
        return StageResult(
            {
                _FEATURES: Artifact.from_value(
                    _FEATURES,
                    features,
                )
            }
        )


@dataclass
class _StaticFeatureStage:
    calls: ClassVar[int] = 0

    @property
    def descriptor(self):
        return StageDescriptor(
            "test.static_feature_stage",
            cache_scope=StageCacheScope.CONTENT,
            side_effect=StageSideEffect.PURE,
        )

    @property
    def requirements(self):
        return StageRequirements()

    @property
    def outputs(self):
        return (ArtifactOutput(_FEATURES),)

    def canonical_configuration(self):
        return {}

    def run(self, inputs, runtime):
        del inputs, runtime
        type(self).calls += 1
        features = pd.DataFrame(
            {"benchmark_weight": [0.4, 0.6]},
            index=pd.Index(["A", "B"], name="instrument_id"),
        )
        return StageResult(
            {
                _FEATURES: Artifact.from_value(
                    _FEATURES,
                    features,
                )
            }
        )


@dataclass(frozen=True)
class _FeatureWeightStage:
    tilt: float
    calls: ClassVar[int] = 0

    @property
    def descriptor(self):
        return StageDescriptor(
            "test.feature_weight_stage",
            cache_scope=StageCacheScope.CONTENT,
            side_effect=StageSideEffect.PURE,
        )

    @property
    def requirements(self):
        return StageRequirements(
            artifacts=(ArtifactRequirement(_FEATURES),)
        )

    @property
    def outputs(self):
        return (ArtifactOutput(CORE_TARGET_WEIGHTS),)

    def canonical_configuration(self):
        return {"tilt": self.tilt}

    def run(self, inputs, runtime):
        del runtime
        type(self).calls += 1
        benchmark = inputs.value(_FEATURES)["benchmark_weight"]
        desired = benchmark.pow(self.tilt)
        weights = (desired / desired.sum()).rename("index_weight")
        return StageResult(
            {
                CORE_TARGET_WEIGHTS: Artifact.from_value(
                    CORE_TARGET_WEIGHTS,
                    weights,
                )
            }
        )


@dataclass
class _ProviderWeightStage:
    calls: int = 0

    @property
    def descriptor(self):
        return StageDescriptor(
            "test.provider_weight_stage",
            cache_scope=StageCacheScope.CONTENT,
            side_effect=StageSideEffect.READ_ONLY_IO,
        )

    @property
    def requirements(self):
        return StageRequirements(
            provider_capabilities=("load_universe",),
            provider_requests=(
                ProviderRequestSpec(
                    "load_universe",
                    review_dimensions=frozenset(
                        {"reference_date", "effective_date"}
                    ),
                ),
            ),
        )

    @property
    def outputs(self):
        return (ArtifactOutput(CORE_TARGET_WEIGHTS),)

    def canonical_configuration(self):
        return {}

    def run(self, inputs, runtime):
        type(self).calls += 1
        provider = runtime.providers["load_universe"]
        parameters = runtime.provider_parameters["load_universe"]
        frame = provider.load_universe(
            reference_date=inputs.review.reference_date,
            effective_date=inputs.review.effective_date,
            **parameters,
        )
        weights = frame.set_index("instrument_id")[
            "benchmark_weight"
        ]
        return StageResult(
            {
                CORE_TARGET_WEIGHTS: Artifact.from_value(
                    CORE_TARGET_WEIGHTS,
                    weights,
                )
            }
        )


@dataclass
class _MultiRequestProviderWeightStage:
    calls: ClassVar[int] = 0

    @property
    def descriptor(self):
        return StageDescriptor(
            "test.multi_request_provider_weight_stage",
            cache_scope=StageCacheScope.CONTENT,
            side_effect=StageSideEffect.READ_ONLY_IO,
        )

    @property
    def requirements(self):
        requests = tuple(
            ProviderRequestSpec(
                "load_universe",
                request_parameters={
                    "universe_id": "RECIPE_UNIVERSE",
                    "variant": variant,
                },
                review_dimensions=frozenset(
                    {"reference_date", "effective_date"}
                ),
            )
            for variant in ("primary", "secondary")
        )
        return StageRequirements(
            provider_capabilities=("load_universe",),
            provider_requests=requests,
        )

    @property
    def outputs(self):
        return (ArtifactOutput(CORE_TARGET_WEIGHTS),)

    def canonical_configuration(self):
        return {}

    def run(self, inputs, runtime):
        type(self).calls += 1
        provider = runtime.providers["load_universe"]
        frames = [
            provider.load_universe(
                universe_id="RECIPE_UNIVERSE",
                variant=variant,
                reference_date=inputs.review.reference_date,
                effective_date=inputs.review.effective_date,
            )
            for variant in ("primary", "secondary")
        ]
        weights = frames[0].set_index("instrument_id")[
            "benchmark_weight"
        ]
        return StageResult(
            {
                CORE_TARGET_WEIGHTS: Artifact.from_value(
                    CORE_TARGET_WEIGHTS,
                    weights,
                )
            }
        )


@dataclass
class _UndeclaredProviderWeightStage(_ProviderWeightStage):
    @property
    def requirements(self):
        return StageRequirements(
            provider_capabilities=("load_universe",)
        )


@dataclass
class _StatefulWeightStage:
    calls: int = 0

    @property
    def descriptor(self):
        return StageDescriptor(
            "test.stateful_weight_stage",
            cache_scope=StageCacheScope.CONTENT,
        )

    @property
    def requirements(self):
        return StageRequirements(
            prior_artifacts=(
                PriorArtifactRequirement(
                    _STATE,
                    policy=PriorStatePolicy.EMPTY_INITIAL,
                ),
            )
        )

    @property
    def outputs(self):
        return (
            ArtifactOutput(CORE_TARGET_WEIGHTS),
            ArtifactOutput(_STATE),
        )

    def canonical_configuration(self):
        return {}

    def run(self, inputs, runtime):
        del runtime
        type(self).calls += 1
        prior = inputs.prior_artifacts.get(_STATE)
        count = 0 if prior is None else int(prior.value)
        weights = pd.Series(
            [0.5 + 0.1 * count, 0.5 - 0.1 * count],
            index=pd.Index(["A", "B"], name="instrument_id"),
            name="index_weight",
        )
        return StageResult(
            {
                CORE_TARGET_WEIGHTS: Artifact.from_value(
                    CORE_TARGET_WEIGHTS,
                    weights,
                ),
                _STATE: Artifact.from_value(_STATE, count + 1),
            }
        )


@dataclass
class _OpaqueWeightStage:
    calls: int = 0

    @property
    def descriptor(self):
        return StageDescriptor(
            "test.opaque_weight_stage",
            cache_scope=StageCacheScope.DISABLED,
            side_effect=StageSideEffect.OPAQUE,
        )

    @property
    def requirements(self):
        return StageRequirements()

    @property
    def outputs(self):
        return (ArtifactOutput(CORE_TARGET_WEIGHTS),)

    def canonical_configuration(self):
        return {}

    def run(self, inputs, runtime):
        del inputs, runtime
        type(self).calls += 1
        weights = pd.Series(
            [0.4, 0.6],
            index=pd.Index(["A", "B"], name="instrument_id"),
            name="index_weight",
        )
        return StageResult(
            {
                CORE_TARGET_WEIGHTS: Artifact.from_value(
                    CORE_TARGET_WEIGHTS,
                    weights,
                )
            }
        )


@dataclass
class _RandomWeightStage:
    calls: ClassVar[int] = 0

    @property
    def descriptor(self):
        return StageDescriptor(
            "test.random_weight_stage",
            cache_scope=StageCacheScope.CONTENT,
            uses_randomness=True,
        )

    @property
    def requirements(self):
        return StageRequirements()

    @property
    def outputs(self):
        return (ArtifactOutput(CORE_TARGET_WEIGHTS),)

    def canonical_configuration(self):
        return {}

    def run(self, inputs, runtime):
        del inputs
        type(self).calls += 1
        fraction = (int(runtime.random_seed) % 500 + 250) / 1000.0
        weights = pd.Series(
            [fraction, 1.0 - fraction],
            index=pd.Index(["A", "B"], name="instrument_id"),
            name="index_weight",
        )
        return StageResult(
            {
                CORE_TARGET_WEIGHTS: Artifact.from_value(
                    CORE_TARGET_WEIGHTS,
                    weights,
                )
            }
        )


def _calendar() -> Calendar:
    return Calendar.from_dates(
        [
            {
                "reference_date": "2026-04-24",
                "effective_date": "2026-05-01",
            },
            {
                "reference_date": "2026-05-22",
                "effective_date": "2026-06-01",
            },
        ]
    )


def _recipe(stage) -> IndexRecipe:
    return IndexRecipe(nodes=(StageNode("weights", stage),))


def _feature_recipe(source_stage, *, tilt: float) -> IndexRecipe:
    return IndexRecipe(
        nodes=(
            StageNode("weights", _FeatureWeightStage(tilt)),
            StageNode("features", source_stage),
        )
    )


def _reuse() -> CacheOptions:
    return CacheOptions(
        stage_modes={CacheStage.REVIEWS: CacheMode.REUSE}
    )


def test_provider_request_contract_merges_fixed_binding_and_review_parameters():
    parameters = {"tenant": "alpha"}
    request = ProviderRequestSpec(
        "load_universe",
        review_dimensions=frozenset(
            {"reference_date", "effective_date"}
        ),
        request_parameters={"universe_id": "RECIPE_UNIVERSE"},
        expected_provider_name=_PROVIDER_NAME,
        expected_provider_parameters=parameters,
    )
    review = ReviewIdentity(
        index_id="RECIPE_INDEX",
        reference_date="2026-04-24",
        effective_date="2026-05-01",
    )

    request.validate_binding(_PROVIDER_NAME, parameters)
    assert request.build_request(review, parameters) == {
        "universe_id": "RECIPE_UNIVERSE",
        "tenant": "alpha",
        "reference_date": pd.Timestamp("2026-04-24"),
        "effective_date": pd.Timestamp("2026-05-01"),
    }
    with pytest.raises(ValueError, match="expects provider"):
        request.validate_binding("different_provider", parameters)
    with pytest.raises(ValueError, match="binding parameters"):
        request.validate_binding(_PROVIDER_NAME, {"tenant": "beta"})


def test_provider_request_contract_supports_same_capability_snapshot_scopes():
    first = ProviderRequestSpec(
        "load_third_party_data",
        request_parameters={"data_type": "factor", "fields": ("score",)},
    )
    second = ProviderRequestSpec(
        "load_third_party_data",
        request_parameters={
            "data_type": "liquidity",
            "fields": ("average_daily_value",),
        },
    )

    requirements = StageRequirements(
        provider_capabilities=("load_third_party_data",),
        provider_requests=(first, second),
    )

    assert requirements.provider_requests == (first, second)


def test_provider_request_contract_supports_nested_parameters_and_mapped_dates():
    request = ProviderRequestSpec(
        "load_third_party_data",
        request_parameters={
            "data_type": "factor",
            "fields": ("quality",),
        },
        provider_parameters_key="parameters",
        review_parameter_map={"as_of_date": "reference_date"},
    )
    review = ReviewIdentity(
        index_id="RECIPE_INDEX",
        reference_date="2026-04-24",
        effective_date="2026-05-01",
    )

    assert request.build_request(review, {"tenant": "alpha"}) == {
        "data_type": "factor",
        "fields": ("quality",),
        "parameters": {"tenant": "alpha"},
        "as_of_date": pd.Timestamp("2026-04-24"),
    }
    with pytest.raises(ValueError, match="must not overlap"):
        ProviderRequestSpec(
            "load_universe",
            request_parameters={"tenant": "fixed"},
        ).build_request(review, {"tenant": "binding"})
    with pytest.raises(ValueError, match="must not override review-bound"):
        ProviderRequestSpec(
            "load_universe",
            review_dimensions=frozenset({"reference_date"}),
            request_parameters={"reference_date": "fixed"},
        ).build_request(review, {})


def test_provider_request_parameter_identity_is_exact_and_fail_closed():
    request = ProviderRequestSpec(
        "load_universe",
        expected_provider_parameters={"credential": b"alpha"},
    )

    request.validate_binding("provider", {"credential": b"alpha"})
    with pytest.raises(ValueError, match="binding parameters"):
        request.validate_binding("provider", {"credential": b"beta"})
    with pytest.raises(TypeError, match="unsupported value type"):
        ProviderRequestSpec(
            "load_universe",
            expected_provider_parameters={"opaque": object()},
        )
    assert private_parameter_scope_digest(
        {"credential": b"alpha"}
    ) == private_parameter_digest({"credential": b"alpha"})
    assert private_parameter_digest({"value": (1, 2)}) != (
        private_parameter_digest({"value": [1, 2]})
    )
    assert private_parameter_digest({"value": Path("/tmp/example")}) != (
        private_parameter_digest({"value": "/tmp/example"})
    )
    assert private_parameter_digest({"value": b"alpha"}) != (
        private_parameter_digest(
            {
                "value": {
                    "type": "bytes",
                    "sha256": "not-a-byte-digest",
                    "length": 5,
                }
            }
        )
    )
    with pytest.raises(TypeError, match="require string keys"):
        private_parameter_digest({"value": {1: "alpha"}})


def test_provider_request_contract_is_deeply_immutable_and_hashable():
    source = {
        "nested": {
            "values": [1, 2],
        }
    }
    first = ProviderRequestSpec(
        "load_universe",
        request_parameters=source,
        expected_provider_parameters={"credential": b"alpha"},
    )
    equivalent = ProviderRequestSpec(
        "load_universe",
        request_parameters={"nested": {"values": [1, 2]}},
        expected_provider_parameters={"credential": b"alpha"},
    )
    original_hash = hash(first)

    source["nested"]["values"].append(3)
    with pytest.raises(TypeError):
        first.request_parameters["nested"]["extra"] = "mutation"
    built = first.build_request(
        ReviewIdentity(
            index_id="RECIPE_INDEX",
            reference_date="2026-04-24",
            effective_date="2026-05-01",
        ),
        {"credential": b"alpha"},
    )
    built["nested"]["values"].append(4)

    assert first == equivalent
    assert hash(first) == original_hash == hash(equivalent)
    assert first.build_request(
        ReviewIdentity(
            index_id="RECIPE_INDEX",
            reference_date="2026-04-24",
            effective_date="2026-05-01",
        ),
        {"credential": b"alpha"},
    )["nested"]["values"] == [1, 2]


def test_recipe_provider_binding_is_explicit_manifested_and_reused(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _SnapshotUniverseProvider()
    register_provider(_PROVIDER_NAME, provider, replace=True)
    _ProviderWeightStage.calls = 0
    try:
        spec = ResearchSpec(
            definition=IndexDefinition(
                "RECIPE_PROVIDER_INDEX",
                _recipe(_ProviderWeightStage()),
                rebalance_frequency=RebalanceFrequency.MONTHLY,
            ),
            calendar=_calendar(),
            recipe_providers={
                "load_universe": RecipeProviderBinding(
                    _PROVIDER_NAME,
                    {
                        "universe_id": "RECIPE_UNIVERSE",
                        "password": "must-not-be-persisted",
                    },
                )
            },
            analytics=None,
            cache=_reuse(),
        )
        workspace = ResearchWorkspace.open("recipe_provider")
        first = workspace.run(spec)
        calls_after_first = provider.load_calls
        second = workspace.run(spec)
    finally:
        registry.unregister(_PROVIDER_NAME)

    assert calls_after_first == 2
    assert provider.load_calls == calls_after_first
    assert _ProviderWeightStage.calls == 2
    assert first.manifest.definition_fingerprint == (
        second.manifest.definition_fingerprint
    )
    assert any(
        record["capability"] == "load_universe"
        for record in second.manifest.providers
    )
    assert "must-not-be-persisted" not in str(second.manifest)


def test_recipe_snapshot_uses_the_declared_exact_provider_request(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _SnapshotUniverseProvider()
    register_provider(_PROVIDER_NAME, provider, replace=True)
    try:
        spec = ResearchSpec(
            definition=IndexDefinition(
                "RECIPE_EXACT_REQUEST_INDEX",
                _recipe(_ProviderWeightStage()),
            ),
            calendar=_calendar(),
            recipe_providers={
                "load_universe": RecipeProviderBinding(
                    _PROVIDER_NAME,
                    {"universe_id": "RECIPE_UNIVERSE"},
                )
            },
            analytics=None,
            cache=_reuse(),
        )
        ResearchWorkspace.open("recipe_exact_request").run(spec)
    finally:
        registry.unregister(_PROVIDER_NAME)

    assert provider.snapshot_requests
    assert provider.load_requests
    expected_keys = {
        "universe_id",
        "reference_date",
        "effective_date",
    }
    assert all(set(request) == expected_keys for request in provider.snapshot_requests)
    assert all(set(request) == expected_keys for request in provider.load_requests)
    assert {
        (
            request["universe_id"],
            pd.Timestamp(request["reference_date"]).normalize(),
            pd.Timestamp(request["effective_date"]).normalize(),
        )
        for request in provider.snapshot_requests
    } == {
        (
            request["universe_id"],
            pd.Timestamp(request["reference_date"]).normalize(),
            pd.Timestamp(request["effective_date"]).normalize(),
        )
        for request in provider.load_requests
    }


def test_multiple_same_capability_requests_are_snapshotted_and_reused(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _SnapshotUniverseProvider()
    register_provider(_PROVIDER_NAME, provider, replace=True)
    _MultiRequestProviderWeightStage.calls = 0
    spec = ResearchSpec(
        definition=IndexDefinition(
            "RECIPE_MULTI_REQUEST_INDEX",
            _recipe(_MultiRequestProviderWeightStage()),
        ),
        calendar=_calendar(),
        recipe_providers={"load_universe": _PROVIDER_NAME},
        analytics=None,
        cache=_reuse(),
    )
    try:
        workspace = ResearchWorkspace.open("recipe_multi_request")
        first = workspace.run(spec)
        calls_after_first = provider.load_calls
        clear_memory_cache()
        second = workspace.run(spec)
    finally:
        registry.unregister(_PROVIDER_NAME)

    assert calls_after_first == 4
    assert provider.load_calls == calls_after_first
    assert _MultiRequestProviderWeightStage.calls == 2
    assert {
        item.cache_source
        for item in first.backtest.metadata.reviews.values()
    } == {CacheSource.COMPUTED}
    assert {
        item.cache_source
        for item in second.backtest.metadata.reviews.values()
    } == {CacheSource.DISK}
    assert {request["variant"] for request in provider.snapshot_requests} == {
        "primary",
        "secondary",
    }


def test_undeclared_recipe_provider_request_is_off_only(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _SnapshotUniverseProvider()
    register_provider(_PROVIDER_NAME, provider, replace=True)

    def spec(cache):
        return ResearchSpec(
            definition=IndexDefinition(
                "RECIPE_UNDECLARED_REQUEST_INDEX",
                _recipe(_UndeclaredProviderWeightStage()),
            ),
            calendar=_calendar(),
            recipe_providers={
                "load_universe": RecipeProviderBinding(
                    _PROVIDER_NAME,
                    {"universe_id": "RECIPE_UNIVERSE"},
                )
            },
            analytics=None,
            cache=cache,
        )

    try:
        workspace = ResearchWorkspace.open("recipe_undeclared_request")
        workspace.run(spec(CacheOptions.off()))
        calls_after_off_run = provider.load_calls
        with pytest.raises(UnsafeCacheReuseError, match="exact provider requests"):
            workspace.run(spec(_reuse()))
    finally:
        registry.unregister(_PROVIDER_NAME)

    assert calls_after_off_run == 2
    assert provider.load_calls == calls_after_off_run
    assert provider.snapshot_requests == ()


def test_recipe_provider_cache_is_scoped_by_private_parameters(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _CredentialScopedUniverseProvider()
    register_provider(_PROVIDER_NAME, provider, replace=True)
    _ProviderWeightStage.calls = 0
    workspace = ResearchWorkspace.open("recipe_provider_private_scope")

    def spec(password):
        return ResearchSpec(
            definition=IndexDefinition(
                "RECIPE_PRIVATE_SCOPE_INDEX",
                _recipe(_ProviderWeightStage()),
            ),
            calendar=_calendar(),
            recipe_providers={
                "load_universe": RecipeProviderBinding(
                    _PROVIDER_NAME,
                    {
                        "universe_id": "RECIPE_UNIVERSE",
                        "password": password,
                    },
                )
            },
            analytics=None,
            cache=_reuse(),
        )

    try:
        first = workspace.run(spec("credential-alpha"))
        second = workspace.run(spec("credential-beta"))
    finally:
        registry.unregister(_PROVIDER_NAME)

    assert provider.load_calls == 4
    assert _ProviderWeightStage.calls == 4
    assert first.manifest.definition_fingerprint == (
        second.manifest.definition_fingerprint
    )
    assert not first.backtest.weights.equals(second.backtest.weights)
    persisted = b"".join(
        path.read_bytes()
        for path in workspace.workspace_path.rglob("*")
        if path.is_file()
    )
    assert b"credential-alpha" not in persisted
    assert b"credential-beta" not in persisted


def test_recipe_provider_without_snapshot_executes_provider_before_reuse(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _ContentOnlyUniverseProvider()
    register_provider(_PROVIDER_NAME, provider, replace=True)
    _ProviderWeightStage.calls = 0
    try:
        spec = ResearchSpec(
            definition=IndexDefinition(
                "RECIPE_CONTENT_INDEX",
                _recipe(_ProviderWeightStage()),
            ),
            calendar=_calendar(),
            recipe_providers={
                "load_universe": RecipeProviderBinding(
                    _PROVIDER_NAME,
                    {"universe_id": "RECIPE_UNIVERSE"},
                )
            },
            analytics=None,
            cache=_reuse(),
        )
        workspace = ResearchWorkspace.open("recipe_content")
        first = workspace.run(spec)
        second = workspace.run(spec)
    finally:
        registry.unregister(_PROVIDER_NAME)

    assert provider.load_calls == 4
    assert _ProviderWeightStage.calls == 4
    for run in (first, second):
        decision = next(
            item
            for item in run.manifest.cache_decisions
            if item["stage"] == "reviews"
        )
        assert decision["mode"] == "reuse"
        assert "content-identify" in decision["reason"]


def test_content_identified_provider_output_reuses_downstream_stage(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _ContentOnlyUniverseProvider()
    register_provider(_PROVIDER_NAME, provider, replace=True)
    _ProviderFeatureStage.calls = 0
    _FeatureWeightStage.calls = 0
    try:
        spec = ResearchSpec(
            definition=IndexDefinition(
                "RECIPE_CONTENT_PIPELINE",
                _feature_recipe(_ProviderFeatureStage(), tilt=1.0),
            ),
            calendar=_calendar(),
            recipe_providers={
                "load_universe": RecipeProviderBinding(
                    _PROVIDER_NAME,
                    {"universe_id": "RECIPE_UNIVERSE"},
                )
            },
            analytics=None,
            cache=_reuse(),
        )
        workspace = ResearchWorkspace.open("recipe_content_pipeline")
        workspace.run(spec)
        workspace.run(spec)
    finally:
        registry.unregister(_PROVIDER_NAME)

    assert provider.load_calls == 4
    assert _ProviderFeatureStage.calls == 4
    assert _FeatureWeightStage.calls == 2


def test_downstream_recipe_parameter_change_reuses_upstream_stage(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    _StaticFeatureStage.calls = 0
    _FeatureWeightStage.calls = 0
    workspace = ResearchWorkspace.open("recipe_parameter_scenarios")
    baseline = ResearchSpec(
        definition=IndexDefinition(
            "RECIPE_PARAMETER_INDEX",
            _feature_recipe(_StaticFeatureStage(), tilt=1.0),
        ),
        calendar=_calendar(),
        analytics=None,
        cache=_reuse(),
    )
    candidate = ResearchSpec(
        definition=IndexDefinition(
            "RECIPE_PARAMETER_INDEX",
            _feature_recipe(_StaticFeatureStage(), tilt=2.0),
        ),
        calendar=_calendar(),
        analytics=None,
        cache=_reuse(),
    )

    first = workspace.run(baseline)
    second = workspace.run(candidate)

    assert first.manifest.definition_fingerprint != (
        second.manifest.definition_fingerprint
    )
    assert _StaticFeatureStage.calls == 2
    assert _FeatureWeightStage.calls == 4


def test_provider_and_calendar_implementations_are_definition_scoped(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _SnapshotUniverseProvider()
    register_provider(_PROVIDER_NAME, provider, replace=True)
    provider_spec = ResearchSpec(
        definition=IndexDefinition(
            "RECIPE_PROVIDER_IDENTITY",
            _recipe(_ProviderWeightStage()),
        ),
        calendar=_calendar(),
        recipe_providers={
            "load_universe": RecipeProviderBinding(
                _PROVIDER_NAME,
                {"universe_id": "RECIPE_UNIVERSE"},
            )
        },
        analytics=None,
    )
    workspace = ResearchWorkspace.open("provider_calendar_identity")
    try:
        first_provider = workspace.run(provider_spec)
        register_provider(
            _PROVIDER_NAME,
            _AlternativeSnapshotUniverseProvider(),
            replace=True,
        )
        second_provider = workspace.run(provider_spec)
    finally:
        registry.unregister(_PROVIDER_NAME)

    assert first_provider.manifest.definition_fingerprint != (
        second_provider.manifest.definition_fingerprint
    )

    standard_calendar = _calendar()
    alternative_calendar = _AlternativeCalendar.from_dates(
        standard_calendar.dates.loc[
            :,
            ["reference_date", "effective_date"],
        ]
    )
    definition = IndexDefinition(
        "RECIPE_CALENDAR_IDENTITY",
        _feature_recipe(_StaticFeatureStage(), tilt=1.0),
    )
    standard = workspace.run(
        ResearchSpec(
            definition=definition,
            calendar=standard_calendar,
            analytics=None,
        )
    )
    alternative = workspace.run(
        ResearchSpec(
            definition=definition,
            calendar=alternative_calendar,
            analytics=None,
        )
    )
    assert standard.manifest.definition_fingerprint != (
        alternative.manifest.definition_fingerprint
    )

    register_provider(
        _CALENDAR_PROVIDER_NAME,
        _CalendarProvider(),
        replace=True,
    )
    try:
        provider_calendar = Calendar(
            start_date="2026-05-01",
            end_date="2026-06-01",
            calendar_id="RESEARCH_CALENDAR",
            provider_name=_CALENDAR_PROVIDER_NAME,
        )
        first_calendar_provider = workspace.run(
            ResearchSpec(
                definition=definition,
                calendar=provider_calendar,
                analytics=None,
            )
        )
        register_provider(
            _CALENDAR_PROVIDER_NAME,
            _AlternativeCalendarProvider(),
            replace=True,
        )
        alternative_provider_calendar = Calendar(
            start_date="2026-05-01",
            end_date="2026-06-01",
            calendar_id="RESEARCH_CALENDAR",
            provider_name=_CALENDAR_PROVIDER_NAME,
        )
        second_calendar_provider = workspace.run(
            ResearchSpec(
                definition=definition,
                calendar=alternative_provider_calendar,
                analytics=None,
            )
        )
    finally:
        registry.unregister(_CALENDAR_PROVIDER_NAME)

    assert first_calendar_provider.manifest.definition_fingerprint != (
        second_calendar_provider.manifest.definition_fingerprint
    )


def test_runtime_revision_invalidates_review_and_recipe_stage_identity(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = ResearchWorkspace.open("recipe_runtime_identity")
    _StaticFeatureStage.calls = 0
    spec = ResearchSpec(
        definition=IndexDefinition(
            "RECIPE_RUNTIME_INDEX",
            _feature_recipe(_StaticFeatureStage(), tilt=1.0),
        ),
        calendar=_calendar(),
        analytics=None,
        cache=_reuse(),
    )

    monkeypatch.setattr(
        workflow_module,
        "automatic_runtime_identity",
        lambda: ({"name": "test_runtime", "version": "1"},),
    )
    first = workspace.run(spec)
    monkeypatch.setattr(
        workflow_module,
        "automatic_runtime_identity",
        lambda: ({"name": "test_runtime", "version": "2"},),
    )
    second = workspace.run(spec)

    assert first.manifest.definition_fingerprint != (
        second.manifest.definition_fingerprint
    )
    assert _StaticFeatureStage.calls == 4


def test_schedule_policy_changes_request_but_not_definition_identity(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = ResearchWorkspace.open("schedule_policy_identity")
    definition = IndexDefinition(
        "SCHEDULE_POLICY_INDEX",
        _feature_recipe(_StaticFeatureStage(), tilt=1.0),
        rebalance_frequency=RebalanceFrequency.MONTHLY,
    )
    default_policy = workspace.run(
        ResearchSpec(
            definition=definition,
            calendar=_calendar(),
            analytics=None,
        )
    )
    permissive_policy = workspace.run(
        ResearchSpec(
            definition=definition,
            calendar=_calendar(),
            analytics=None,
            allow_additional_reviews=True,
            allow_frequency_gaps=False,
            allow_empty_recipe_initial_state=True,
        )
    )

    assert default_policy.manifest.definition_fingerprint == (
        permissive_policy.manifest.definition_fingerprint
    )
    assert default_policy.manifest.request_fingerprint != (
        permissive_policy.manifest.request_fingerprint
    )


def test_recipe_seed_is_definition_derived_and_can_be_overridden(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = ResearchWorkspace.open("recipe_random_seed")
    definition = IndexDefinition(
        "RECIPE_RANDOM_INDEX",
        _recipe(_RandomWeightStage()),
    )
    automatic_run = workspace.run(
        ResearchSpec(
            definition=definition,
            calendar=_calendar(),
            analytics=None,
        )
    )
    first_context = next(iter(automatic_run.backtest.reviews.values()))
    stage_record = first_context.diagnostics["index_recipe"]["stages"][0]
    expected = int(
        automatic_digest(
            {
                "definition_fingerprint": (
                    automatic_run.manifest.definition_fingerprint
                )
            }
        )[:16],
        16,
    ) % (2**32)
    assert stage_record["random_seed"] == expected

    explicit_run = workspace.run(
        ResearchSpec(
            definition=definition,
            calendar=_calendar(),
            analytics=None,
            random_seed=17,
        )
    )
    explicit_context = next(iter(explicit_run.backtest.reviews.values()))
    assert (
        explicit_context.diagnostics["index_recipe"]["stages"][0][
            "random_seed"
        ]
        == 17
    )
    assert automatic_run.manifest.definition_fingerprint == (
        explicit_run.manifest.definition_fingerprint
    )
    assert automatic_run.manifest.request_fingerprint != (
        explicit_run.manifest.request_fingerprint
    )


def test_explicit_recipe_seed_invalidates_reusable_review_results(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = ResearchWorkspace.open("recipe_random_seed_reuse")
    _RandomWeightStage.calls = 0
    definition = IndexDefinition(
        "RECIPE_RANDOM_REUSE_INDEX",
        _recipe(_RandomWeightStage()),
    )
    first = workspace.run(
        ResearchSpec(
            definition=definition,
            calendar=_calendar(),
            analytics=None,
            cache=_reuse(),
            random_seed=17,
        )
    )
    second = workspace.run(
        ResearchSpec(
            definition=definition,
            calendar=_calendar(),
            analytics=None,
            cache=_reuse(),
            random_seed=23,
        )
    )

    assert _RandomWeightStage.calls == 4
    assert first.manifest.definition_fingerprint == (
        second.manifest.definition_fingerprint
    )
    assert first.manifest.request_fingerprint != (
        second.manifest.request_fingerprint
    )
    assert not first.backtest.weights.equals(second.backtest.weights)


def test_actual_off_disables_native_recipe_stage_cache(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    _OpaqueWeightStage.calls = 0
    spec = ResearchSpec(
        definition=IndexDefinition(
            "OPAQUE_RECIPE_INDEX",
            _recipe(_OpaqueWeightStage()),
        ),
        calendar=_calendar(),
        analytics=None,
        cache=_reuse(),
    )
    workspace = ResearchWorkspace.open("opaque_recipe")

    first = workspace.run(spec)
    second = workspace.run(spec)

    assert _OpaqueWeightStage.calls == 4
    for run in (first, second):
        decision = next(
            item
            for item in run.manifest.cache_decisions
            if item["stage"] == "reviews"
        )
        assert decision["requested_mode"] == "reuse"
        assert decision["mode"] == "off"


def test_unfingerprintable_callable_recipe_runs_only_with_cache_off(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    opaque = object()

    def calculate(inputs, runtime):
        del inputs, runtime
        if opaque is None:
            raise AssertionError("captured state must remain reachable")
        return {
            CORE_TARGET_WEIGHTS: pd.Series(
                [1.0],
                index=pd.Index(["A"], name="instrument_id"),
                name="index_weight",
            )
        }

    recipe = IndexRecipe(
        nodes=(
            StageNode(
                "weights",
                CallableStage(
                    function=calculate,
                    output_specs=(ArtifactOutput(CORE_TARGET_WEIGHTS),),
                    cache_scope=StageCacheScope.CONTENT,
                ),
            ),
        )
    )
    workspace = ResearchWorkspace.open("opaque_callable_recipe")
    off_run = workspace.run(
        ResearchSpec(
            definition=IndexDefinition("OPAQUE_CALLABLE_OFF", recipe),
            calendar=_calendar(),
            analytics=None,
        )
    )
    assert off_run.backtest.weights["index_weight"].eq(1.0).all()

    with pytest.raises(UnsafeCacheReuseError, match="stable executable"):
        workspace.run(
            ResearchSpec(
                definition=IndexDefinition(
                    "OPAQUE_CALLABLE_REUSE",
                    recipe,
                ),
                calendar=_calendar(),
                analytics=None,
                cache=_reuse(),
            )
        )


def test_stateful_recipe_chains_reviews_and_reuses_stage_state(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    _StatefulWeightStage.calls = 0
    spec = ResearchSpec(
        definition=IndexDefinition(
            "STATEFUL_RECIPE_INDEX",
            _recipe(_StatefulWeightStage()),
            rebalance_frequency=RebalanceFrequency.MONTHLY,
        ),
        calendar=_calendar(),
        analytics=None,
        cache=_reuse(),
        allow_empty_recipe_initial_state=True,
    )
    workspace = ResearchWorkspace.open("stateful_recipe")

    first = workspace.run(spec)
    calls_after_first = _StatefulWeightStage.calls
    second = workspace.run(spec)

    assert calls_after_first == 2
    assert _StatefulWeightStage.calls == calls_after_first
    first_weights = first.backtest.weights["index_weight"]
    assert first_weights.loc[(pd.Timestamp("2026-05-01"), "A")] == 0.5
    assert first_weights.loc[(pd.Timestamp("2026-06-01"), "A")] == 0.6
    pd.testing.assert_frame_equal(
        first.backtest.weights,
        second.backtest.weights,
    )


def test_stateful_recipe_without_an_explicit_initial_seed_fails(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    spec = ResearchSpec(
        definition=IndexDefinition(
            "STATEFUL_RECIPE_MISSING_SEED",
            _recipe(_StatefulWeightStage()),
        ),
        calendar=_calendar(),
        analytics=None,
    )
    workspace = ResearchWorkspace.open("stateful_missing_seed")

    with pytest.raises(PriorReviewStateError):
        workspace.run(spec)
    assert workspace.latest().status == "failed"


def test_recipe_provider_requirements_must_be_bound_explicitly():
    with pytest.raises(
        ValueError,
        match="missing capabilities: load_universe",
    ):
        ResearchSpec(
            definition=IndexDefinition(
                "MISSING_RECIPE_PROVIDER",
                _recipe(_ProviderWeightStage()),
            ),
            calendar=_calendar(),
            analytics=None,
        )


@pytest.mark.parametrize(
    "frequency",
    (
        RebalanceFrequency.MONTHLY,
        RebalanceFrequency.QUARTERLY,
        RebalanceFrequency.SEMI_ANNUAL,
        RebalanceFrequency.ANNUAL,
    ),
)
def test_research_spec_accepts_supported_periodic_schedules(frequency):
    calendar = Calendar.from_frequency(
        start_date="2025-01-01",
        end_date="2026-12-31",
        frequency=frequency,
    )
    original = calendar.dates.copy(deep=True)

    ResearchSpec(
        definition=IndexDefinition(
            f"{frequency.value.upper()}_INDEX",
            _recipe(_OpaqueWeightStage()),
            rebalance_frequency=frequency,
        ),
        calendar=calendar,
        analytics=None,
    )

    pd.testing.assert_frame_equal(calendar.dates, original)


def test_research_spec_custom_and_special_review_controls():
    custom = Calendar.from_dates(
        [
            {
                "reference_date": "2026-01-02",
                "effective_date": "2026-01-05",
            },
            {
                "reference_date": "2026-01-16",
                "effective_date": "2026-01-20",
            },
        ]
    )
    definition = IndexDefinition(
        "SPECIAL_REVIEW_INDEX",
        _recipe(_OpaqueWeightStage()),
        rebalance_frequency=RebalanceFrequency.MONTHLY,
    )
    with pytest.raises(ValueError, match="multiple effective dates"):
        ResearchSpec(
            definition=definition,
            calendar=custom,
            analytics=None,
        )

    accepted = ResearchSpec(
        definition=definition,
        calendar=custom,
        analytics=None,
        allow_additional_reviews=True,
    )
    assert len(accepted.calendar.dates) == 2

    ResearchSpec(
        definition=IndexDefinition(
            "CUSTOM_REVIEW_INDEX",
            _recipe(_OpaqueWeightStage()),
            rebalance_frequency=RebalanceFrequency.CUSTOM,
        ),
        calendar=custom,
        analytics=None,
    )

    gap = Calendar.from_dates(
        [
            {
                "reference_date": "2026-01-23",
                "effective_date": "2026-01-30",
            },
            {
                "reference_date": "2026-03-24",
                "effective_date": "2026-03-31",
            },
        ]
    )
    with pytest.raises(ValueError, match="missing monthly periods"):
        ResearchSpec(
            definition=definition,
            calendar=gap,
            analytics=None,
            allow_frequency_gaps=False,
        )
