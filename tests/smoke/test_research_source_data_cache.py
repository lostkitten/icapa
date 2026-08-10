"""High-level source-data cache behavior and provider snapshot contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import pandas as pd
import pytest

from icapa.backtesting import (
    Calendar,
    RebalanceFrequency,
    RelativeCapitalizationDrift,
    SimulationMaterialization,
    SimulationParams,
    WeightSnapshotMode,
)
from icapa.data_sources import register_provider, registry
from icapa.research import (
    IndexDefinition,
    ResearchSimulationSpec,
    ResearchSpec,
    ResearchWorkspace,
    UnsafeCacheReuseError,
)
from icapa.workspace.caches.source import (
    BusinessDayCacheLoader,
    SourceDataCacheLoader,
)
from icapa.workspace import (
    ArtifactIntegrityError,
    CacheMode,
    CacheOptions,
    CacheStage,
    ManifestIntegrityError,
    WorkspaceRepository as ManifestWorkspace,
)


_PROVIDER_NAME = "research_source_cache_provider"


class _SnapshotProvider:
    def __init__(self) -> None:
        self._universe_calls = 0
        self._market_calls = 0
        self._snapshot_calls = 0
        self._legacy_identity_calls = 0

    def describe_snapshot(self, *, capability, request):
        self._snapshot_calls += 1
        return {
            "dataset": "source-cache-fixture",
            "revision": 3,
            "capability": capability,
            # These values deliberately test that snapshot payloads are hashed
            # rather than persisted.
            "password": request.get("password"),
            "sql": request.get("sql"),
            "connection_string": request.get("connection_string"),
        }

    def research_data_identity(self, *, capability, request):
        self._legacy_identity_calls += 1
        return {"revision": "legacy-hook-must-not-win"}

    def load_universe(
        self,
        universe_id,
        reference_date,
        effective_date,
        **kwargs,
    ):
        self._universe_calls += 1
        return pd.DataFrame(
            {
                "instrument_id": ["A", "B"],
                "name": ["Alpha", "Beta"],
                "country": ["US", "US"],
                "industry": ["Technology", "Industrials"],
                "shares": [100.0, 120.0],
                "free_float": [0.9, 0.8],
                "price": [20.0, 30.0],
                "currency": ["USD", "USD"],
                "base_currency": ["USD", "USD"],
                "fx_rate": [1.0, 1.0],
                "market_cap": [1_800.0, 2_880.0],
                "benchmark_weight": [0.4, 0.6],
                "reference_date": [pd.Timestamp(reference_date)] * 2,
                "effective_date": [pd.Timestamp(effective_date)] * 2,
            }
        )

    def load_daily_market_data(
        self,
        instrument_ids,
        start_date,
        end_date,
        **kwargs,
    ):
        self._market_calls += 1
        rows = []
        origin = pd.Timestamp("2026-01-01")
        for business_date in pd.bdate_range(start_date, end_date):
            date_offset = int((business_date - origin).days)
            for instrument_offset, instrument_id in enumerate(instrument_ids):
                rows.append(
                    {
                        "instrument_id": instrument_id,
                        "business_date": business_date,
                        "price_return": (
                            0.0002
                            + 0.00001 * (date_offset % 7)
                            + 0.00003 * instrument_offset
                        ),
                        "gross_dividend": 0.00001,
                        "net_dividend": 0.000008,
                        "market_cap": (
                            1_000_000.0
                            * (instrument_offset + 1)
                            * (1.0 + 0.0001 * date_offset)
                        ),
                    }
                )
        return pd.DataFrame.from_records(rows)


class _LegacySnapshotProvider(_SnapshotProvider):
    describe_snapshot = None


class _NoSnapshotProvider(_SnapshotProvider):
    describe_snapshot = None
    research_data_identity = None


class _CalendarSnapshotProvider(_SnapshotProvider):
    def __init__(self) -> None:
        super().__init__()
        self._business_day_calls = 0
        self._business_day_requests = []

    def load_business_days(
        self,
        calendar_id,
        start_date,
        end_date,
        **kwargs,
    ):
        self._business_day_calls += 1
        self._business_day_requests.append(
            (
                pd.Timestamp(start_date).normalize(),
                pd.Timestamp(end_date).normalize(),
            )
        )
        return pd.DataFrame(
            {
                "business_date": pd.bdate_range(
                    start_date,
                    end_date,
                )
            }
        )


class _CredentialScopedProvider(_SnapshotProvider):
    """Return one public token while private connection scope changes."""

    def describe_snapshot(self, *, capability, request):
        self._snapshot_calls += 1
        return {
            "dataset": "credential-scope-fixture",
            "revision": 1,
            "capability": capability,
        }


class _MutableSnapshotProvider(_SnapshotProvider):
    def __init__(self) -> None:
        super().__init__()
        self.revision = 1

    def describe_snapshot(self, *, capability, request):
        self._snapshot_calls += 1
        return {
            "dataset": "mutable-source-cache-fixture",
            "revision": self.revision,
            "capability": capability,
        }


class _PerReviewSnapshotProvider(_SnapshotProvider):
    def __init__(self) -> None:
        super().__init__()
        self._review_snapshot_requests = []

    @property
    def review_snapshot_requests(self):
        return tuple(self._review_snapshot_requests)

    def describe_snapshot(self, *, capability, request):
        if capability != "load_universe":
            return super().describe_snapshot(
                capability=capability,
                request=request,
            )
        required = {
            "universe_id",
            "reference_date",
            "effective_date",
        }
        if not required.issubset(request):
            raise ValueError("exact universe snapshot request is required")
        if "methodology" in request:
            raise ValueError("generic methodology payload is not a provider request")
        self._snapshot_calls += 1
        normalized = {
            "universe_id": str(request["universe_id"]),
            "reference_date": str(pd.Timestamp(request["reference_date"]).date()),
            "effective_date": str(pd.Timestamp(request["effective_date"]).date()),
        }
        self._review_snapshot_requests.append(normalized)
        return {
            "dataset": "review-specific-snapshot",
            **normalized,
        }


class _FailingExactSnapshotProvider(_SnapshotProvider):
    def describe_snapshot(self, *, capability, request):
        if capability == "load_universe":
            raise RuntimeError("sensitive provider snapshot failure")
        return super().describe_snapshot(
            capability=capability,
            request=request,
        )


@dataclass(frozen=True)
class _WeightMethodology:
    provider_name: str = _PROVIDER_NAME
    universe_id: str = "CACHE_TEST"
    tilt: float = 0.0

    @property
    def universe_provider_name(self):
        return self.provider_name

    def execute(self, context):
        provider = registry.resolve("load_universe", self.provider_name)
        frame = provider.load_universe(
            universe_id=self.universe_id,
            reference_date=context.reference_date,
            effective_date=context.effective_date,
        )
        weights = frame["benchmark_weight"].copy()
        weights.iloc[0] += self.tilt
        weights.iloc[1] -= self.tilt
        frame["index_weight"] = weights
        context.universe_id = self.universe_id
        context.set_dataframe(frame)
        return context


def _spec(
    *,
    index_id: str,
    cache_mode: CacheMode,
    tilt: float = 0.0,
    simulation_cache_mode: CacheMode = CacheMode.OFF,
    review_cache_mode: CacheMode = CacheMode.OFF,
):
    return ResearchSpec(
        definition=IndexDefinition(
            index_id=index_id,
            methodology=_WeightMethodology(tilt=tilt),
            rebalance_frequency=RebalanceFrequency.MONTHLY,
        ),
        calendar=Calendar.from_dates(
            [
                {
                    "reference_date": "2026-05-22",
                    "effective_date": "2026-06-01",
                },
                {
                    "reference_date": "2026-06-19",
                    "effective_date": "2026-07-01",
                },
            ]
        ),
        simulation=ResearchSimulationSpec(
            market_data_provider_name=_PROVIDER_NAME,
            start_date="2026-06-01",
            end_date="2026-07-10",
            provider_parameters={
                "password": "do-not-persist-password",
                "sql": "select secret_column from private_table",
                "connection_string": "server=private;token=hidden",
                "dataset": "daily",
            },
        ),
        analytics=None,
        cache=CacheOptions(
            stage_modes={
                CacheStage.SOURCE_DATA: cache_mode,
                CacheStage.SIMULATION: simulation_cache_mode,
                CacheStage.REVIEWS: review_cache_mode,
            }
        ),
    )


@pytest.fixture
def _registered_provider():
    providers = []

    def register(provider):
        providers.append(provider)
        register_provider(_PROVIDER_NAME, provider, replace=True)
        return provider

    yield register
    registry.unregister(_PROVIDER_NAME)


def test_source_cache_reuses_streaming_partitions_across_scenarios(
    tmp_path,
    monkeypatch,
    _registered_provider,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _registered_provider(_SnapshotProvider())
    workspace = ResearchWorkspace.open("source_partition_reuse")

    first = workspace.run(
        _spec(index_id="BASELINE_INDEX", cache_mode=CacheMode.REUSE)
    )
    calls_after_first = provider._market_calls
    second = workspace.run(
        _spec(
            index_id="CANDIDATE_INDEX",
            cache_mode=CacheMode.REUSE,
            tilt=0.05,
        )
    )

    assert calls_after_first == 2
    assert provider._market_calls == calls_after_first
    assert provider._legacy_identity_calls == 0
    assert first.simulation is not None
    assert second.simulation is not None
    source_records = [
        item
        for item in second.manifest.input_digests
        if item["input_type"] == "source_daily_market_data"
    ]
    assert len(source_records) == 2
    assert {item["cache_source"] for item in source_records} == {"workspace"}
    assert {item["snapshot_protocol"] for item in source_records} == {
        "describe_snapshot"
    }
    decision = next(
        item
        for item in second.manifest.cache_decisions
        if item["stage"] == "source_data"
    )
    assert decision["mode"] == "reuse"
    assert len(tuple(workspace.workspace_path.glob(
        "bindings/source_data/*/daily-market-data.json"
    ))) == 2

    forbidden = (
        b"do-not-persist-password",
        b"select secret_column from private_table",
        b"server=private;token=hidden",
    )
    persisted = b"".join(
        path.read_bytes()
        for path in workspace.workspace_path.rglob("*")
        if path.is_file()
    )
    assert all(value not in persisted for value in forbidden)


def test_source_cache_refresh_and_read_only_have_explicit_call_semantics(
    tmp_path,
    monkeypatch,
    _registered_provider,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _registered_provider(_SnapshotProvider())
    workspace = ResearchWorkspace.open("source_modes")

    populated = workspace.run(
        _spec(index_id="SOURCE_MODES", cache_mode=CacheMode.REUSE)
    )
    assert populated.simulation is not None
    assert provider._market_calls == 2
    snapshot_calls_after_populated = provider._snapshot_calls

    read_only = workspace.run(
        _spec(index_id="SOURCE_MODES", cache_mode=CacheMode.READ_ONLY)
    )
    assert read_only.simulation is not None
    assert provider._market_calls == 2
    assert provider._snapshot_calls == snapshot_calls_after_populated
    assert {
        item["cache_source"]
        for item in read_only.manifest.input_digests
        if item["input_type"] == "source_daily_market_data"
    } == {"workspace"}
    assert (
        read_only.manifest.result_fingerprint
        == populated.manifest.result_fingerprint
    )

    refreshed = workspace.run(
        _spec(index_id="SOURCE_MODES", cache_mode=CacheMode.REFRESH)
    )
    assert refreshed.simulation is not None
    assert provider._market_calls == 4
    assert {
        item["cache_source"]
        for item in refreshed.manifest.input_digests
        if item["input_type"] == "source_daily_market_data"
    } == {"provider"}
    assert (
        refreshed.manifest.result_fingerprint
        == populated.manifest.result_fingerprint
    )


def test_cache_provenance_does_not_change_numerical_result_identity(
    tmp_path,
    monkeypatch,
    _registered_provider,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    _registered_provider(_SnapshotProvider())
    workspace = ResearchWorkspace.open("source_identity_modes")

    uncached = workspace.run(
        _spec(index_id="SOURCE_IDENTITY", cache_mode=CacheMode.OFF)
    )
    reusable = workspace.run(
        _spec(index_id="SOURCE_IDENTITY", cache_mode=CacheMode.REUSE)
    )

    assert uncached.manifest.request_fingerprint == (
        reusable.manifest.request_fingerprint
    )
    assert uncached.manifest.result_fingerprint == (
        reusable.manifest.result_fingerprint
    )


def test_private_provider_scope_prevents_cross_credential_read_only_reuse(
    tmp_path,
    monkeypatch,
    _registered_provider,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _registered_provider(_CredentialScopedProvider())
    workspace = ManifestWorkspace.open("private_provider_scope")
    first_parameters = {
        "password": "credential-alpha",
        "endpoint": "server-alpha.example",
    }
    request = {
        "instrument_ids": ["A", "B"],
        "start_date": "2026-06-01",
        "end_date": "2026-06-05",
    }

    first = SourceDataCacheLoader(
        workspace,
        provider_name=_PROVIDER_NAME,
        provider_parameters=first_parameters,
        mode=CacheMode.REUSE,
    )
    expected = first.load(**request)
    market_calls = provider._market_calls
    snapshot_calls = provider._snapshot_calls

    same_scope = SourceDataCacheLoader(
        workspace,
        provider_name=_PROVIDER_NAME,
        provider_parameters=first_parameters,
        mode=CacheMode.READ_ONLY,
    )
    pd.testing.assert_frame_equal(
        same_scope.load(**request),
        expected,
    )
    assert provider._market_calls == market_calls
    assert provider._snapshot_calls == snapshot_calls

    different_scope = SourceDataCacheLoader(
        workspace,
        provider_name=_PROVIDER_NAME,
        provider_parameters={
            "password": "credential-beta",
            "endpoint": "server-beta.example",
        },
        mode=CacheMode.READ_ONLY,
    )
    with pytest.raises(UnsafeCacheReuseError, match="snapshot identity"):
        different_scope.load(**request)
    assert provider._market_calls == market_calls
    assert provider._snapshot_calls == snapshot_calls

    persisted = b"".join(
        path.read_bytes()
        for path in workspace.workspace_path.rglob("*")
        if path.is_file()
    )
    for secret in (
        b"credential-alpha",
        b"server-alpha.example",
        b"credential-beta",
        b"server-beta.example",
    ):
        assert secret not in persisted


def test_shorter_month_partition_reuses_containing_parquet_source(
    tmp_path,
    monkeypatch,
    _registered_provider,
):
    """A fresh loader can slice a verified longer monthly source artifact."""

    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _registered_provider(_SnapshotProvider())
    workspace = ManifestWorkspace.open("containing_source_partition")
    common = {
        "instrument_ids": ["A", "B"],
        "start_date": "2026-06-01",
    }

    initial_loader = SourceDataCacheLoader(
        workspace,
        provider_name=_PROVIDER_NAME,
        provider_parameters={},
        mode=CacheMode.REUSE,
    )
    full_month = initial_loader.load(
        **common,
        end_date="2026-06-30",
    )
    calls_after_full_month = provider._market_calls

    fresh_loader = SourceDataCacheLoader(
        workspace,
        provider_name=_PROVIDER_NAME,
        provider_parameters={},
        mode=CacheMode.REUSE,
    )
    shorter = fresh_loader.load(
        **common,
        end_date="2026-06-10",
    )

    assert provider._market_calls == calls_after_full_month
    pd.testing.assert_frame_equal(
        shorter,
        full_month.loc[
            full_month["business_date"] <= pd.Timestamp("2026-06-10")
        ].reset_index(drop=True),
    )
    assert fresh_loader.records[-1]["cache_source"] == (
        "workspace_containing_range"
    )


def test_changed_snapshot_rejects_same_loader_containing_source(
    tmp_path,
    monkeypatch,
    _registered_provider,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _registered_provider(_MutableSnapshotProvider())
    workspace = ManifestWorkspace.open("changed_containing_snapshot")
    loader = SourceDataCacheLoader(
        workspace,
        provider_name=_PROVIDER_NAME,
        provider_parameters={},
        mode=CacheMode.REUSE,
    )

    loader.load(
        instrument_ids=["A", "B"],
        start_date="2026-06-01",
        end_date="2026-06-30",
    )
    calls_before_change = provider._market_calls
    provider.revision = 2
    loader.load(
        instrument_ids=["A", "B"],
        start_date="2026-06-01",
        end_date="2026-06-10",
    )

    assert provider._market_calls == calls_before_change + 1
    assert loader.records[-1]["cache_source"] == "provider"


def test_legacy_snapshot_hook_remains_compatible(
    tmp_path,
    monkeypatch,
    _registered_provider,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _registered_provider(_LegacySnapshotProvider())
    workspace = ResearchWorkspace.open("legacy_snapshot")

    workspace.run(
        _spec(index_id="LEGACY_SNAPSHOT", cache_mode=CacheMode.REUSE)
    )
    calls_after_first = provider._market_calls
    second = workspace.run(
        _spec(index_id="LEGACY_SNAPSHOT", cache_mode=CacheMode.REUSE)
    )

    assert calls_after_first == 2
    assert provider._market_calls == calls_after_first
    assert provider._legacy_identity_calls > 0
    assert {
        item["snapshot_protocol"]
        for item in second.manifest.input_digests
        if item["input_type"] == "source_daily_market_data"
    } == {"research_data_identity"}


def test_standard_provider_snapshot_enables_review_weight_reuse(
    tmp_path,
    monkeypatch,
    _registered_provider,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _registered_provider(_SnapshotProvider())
    workspace = ResearchWorkspace.open("review_provider_snapshot")
    reusable = _spec(
        index_id="REVIEW_PROVIDER_SNAPSHOT",
        cache_mode=CacheMode.OFF,
        review_cache_mode=CacheMode.REUSE,
    )

    first = workspace.run(reusable)
    universe_calls = provider._universe_calls
    snapshot_calls = provider._snapshot_calls
    second = workspace.run(reusable)

    assert universe_calls == 2
    assert provider._universe_calls == universe_calls
    assert first.manifest.result_fingerprint == (
        second.manifest.result_fingerprint
    )
    snapshot_calls_before_read_only = provider._snapshot_calls
    read_only = workspace.run(
        _spec(
            index_id="REVIEW_PROVIDER_SNAPSHOT",
            cache_mode=CacheMode.OFF,
            review_cache_mode=CacheMode.READ_ONLY,
        )
    )
    assert read_only.backtest.metadata.workspace_name == (
        workspace.workspace_name
    )
    assert provider._universe_calls == universe_calls
    # REUSE checks the provider for a newer token; READ_ONLY uses only the
    # locally persisted descriptor.
    assert snapshot_calls_before_read_only > snapshot_calls
    assert provider._snapshot_calls == snapshot_calls_before_read_only


def test_per_review_snapshots_preserve_subset_and_extension_reuse(
    tmp_path,
    monkeypatch,
    _registered_provider,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _registered_provider(_PerReviewSnapshotProvider())
    workspace = ResearchWorkspace.open("per_review_snapshot_reuse")
    first_schedule = [
        {
            "reference_date": "2026-04-24",
            "effective_date": "2026-05-01",
        },
        {
            "reference_date": "2026-05-22",
            "effective_date": "2026-06-01",
        },
    ]
    third_review = {
        "reference_date": "2026-06-19",
        "effective_date": "2026-07-01",
    }
    base = replace(
        _spec(
            index_id="PER_REVIEW_SNAPSHOT",
            cache_mode=CacheMode.OFF,
            review_cache_mode=CacheMode.REUSE,
        ),
        calendar=Calendar.from_dates(first_schedule),
        simulation=None,
        analytics=None,
    )

    first = workspace.run(base)
    calls_after_first = provider._universe_calls
    extended = workspace.run(
        replace(
            base,
            calendar=Calendar.from_dates([*first_schedule, third_review]),
        )
    )
    calls_after_extension = provider._universe_calls
    subset = workspace.run(
        replace(
            base,
            calendar=Calendar.from_dates(first_schedule[:1]),
        )
    )

    assert calls_after_first == 2
    assert calls_after_extension == 3
    assert provider._universe_calls == calls_after_extension
    assert (
        first.manifest.definition_fingerprint
        == extended.manifest.definition_fingerprint
        == subset.manifest.definition_fingerprint
    )
    assert {
        request["effective_date"]
        for request in provider.review_snapshot_requests
    } == {"2026-05-01", "2026-06-01", "2026-07-01"}
    assert all(
        set(request)
        == {"universe_id", "reference_date", "effective_date"}
        for request in provider.review_snapshot_requests
    )


def test_exact_review_snapshot_contract_failure_is_not_downgraded(
    tmp_path,
    monkeypatch,
    _registered_provider,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _registered_provider(_FailingExactSnapshotProvider())
    workspace = ResearchWorkspace.open("failing_exact_review_snapshot")

    with pytest.raises(
        UnsafeCacheReuseError,
        match="exact review-provider snapshot identity failed",
    ):
        workspace.run(
            _spec(
                index_id="FAILING_EXACT_REVIEW_SNAPSHOT",
                cache_mode=CacheMode.OFF,
                review_cache_mode=CacheMode.REUSE,
            )
        )

    assert provider._universe_calls == 0
    latest = workspace.latest()
    assert latest.status == "failed"
    assert "sensitive provider snapshot failure" not in str(latest.failure)


def test_missing_snapshot_never_reuses_unverified_source_data(
    tmp_path,
    monkeypatch,
    _registered_provider,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _registered_provider(_NoSnapshotProvider())
    workspace = ResearchWorkspace.open("content_verified_source")

    first = workspace.run(
        _spec(index_id="NO_SNAPSHOT", cache_mode=CacheMode.REUSE)
    )
    second = workspace.run(
        _spec(index_id="NO_SNAPSHOT", cache_mode=CacheMode.REUSE)
    )

    assert first.simulation is not None
    assert second.simulation is not None
    assert provider._market_calls == 4
    assert {
        item["cache_source"]
        for item in second.manifest.input_digests
        if item["input_type"] == "source_daily_market_data"
    } == {"provider"}

    calls_before_failure = provider._market_calls
    with pytest.raises(UnsafeCacheReuseError):
        workspace.run(
            _spec(index_id="NO_SNAPSHOT", cache_mode=CacheMode.READ_ONLY)
        )
    assert provider._market_calls == calls_before_failure
    assert workspace.latest().status == "failed"


def test_missing_snapshot_content_hash_reuses_downstream_simulation(
    tmp_path,
    monkeypatch,
    _registered_provider,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _registered_provider(_NoSnapshotProvider())
    workspace = ResearchWorkspace.open("content_verified_simulation")
    spec = _spec(
        index_id="CONTENT_VERIFIED_SIMULATION",
        cache_mode=CacheMode.REUSE,
        simulation_cache_mode=CacheMode.REUSE,
    )

    first = workspace.run(spec)
    second = workspace.run(spec)

    assert provider._market_calls == 4
    assert first.simulation is not None
    assert second.simulation is not None
    assert second.simulation.metadata["cache_source"] == "workspace"
    assert first.manifest.result_fingerprint == (
        second.manifest.result_fingerprint
    )
    simulation_decision = next(
        item
        for item in second.manifest.cache_decisions
        if item["stage"] == "simulation"
    )
    assert simulation_decision["mode"] == "reuse"


def test_source_data_off_preflight_is_reused_within_the_same_run(
    tmp_path,
    monkeypatch,
    _registered_provider,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _registered_provider(_NoSnapshotProvider())
    workspace = ResearchWorkspace.open("off_preflight_same_run")

    run = workspace.run(
        _spec(
            index_id="OFF_PREFLIGHT_SAME_RUN",
            cache_mode=CacheMode.OFF,
            simulation_cache_mode=CacheMode.REUSE,
        )
    )

    assert run.simulation is not None
    assert provider._market_calls == 2
    assert run.simulation.metadata["immutable_segments_computed"] == 2
    assert {
        item["cache_source"]
        for item in run.manifest.input_digests
        if item["input_type"] == "source_daily_market_data"
    } == {"provider"}


def test_no_snapshot_extension_reuses_verified_closed_segments(
    tmp_path,
    monkeypatch,
    _registered_provider,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _registered_provider(_NoSnapshotProvider())
    workspace = ResearchWorkspace.open("verified_segment_extension")
    extended_spec = _spec(
        index_id="VERIFIED_SEGMENT_EXTENSION",
        cache_mode=CacheMode.OFF,
        simulation_cache_mode=CacheMode.REUSE,
    )
    assert extended_spec.simulation is not None
    short_spec = replace(
        extended_spec,
        simulation=replace(
            extended_spec.simulation,
            end_date=pd.Timestamp("2026-06-30"),
        ),
    )

    short = workspace.run(short_spec)
    extended = workspace.run(extended_spec)

    assert provider._market_calls == 3
    assert short.manifest.definition_fingerprint == (
        extended.manifest.definition_fingerprint
    )
    assert extended.simulation is not None
    assert extended.simulation.metadata["immutable_segments_reused"] == 1
    assert extended.simulation.metadata["immutable_segments_computed"] == 1


def test_business_day_cache_supports_provider_free_read_only_reuse(
    tmp_path,
    monkeypatch,
    _registered_provider,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _registered_provider(_CalendarSnapshotProvider())
    workspace = ManifestWorkspace.open("business_day_read_only")
    request = {
        "calendar_id": "PRIMARY",
        "start_date": "2026-06-01",
        "end_date": "2026-06-12",
    }

    reusable = BusinessDayCacheLoader(
        workspace,
        provider_name=_PROVIDER_NAME,
        provider_parameters={"password": "calendar-secret"},
        mode=CacheMode.REUSE,
    )
    expected = reusable.load(**request)
    snapshot_calls = provider._snapshot_calls

    read_only = BusinessDayCacheLoader(
        workspace,
        provider_name=_PROVIDER_NAME,
        provider_parameters={"password": "calendar-secret"},
        mode=CacheMode.READ_ONLY,
    )
    actual = read_only.load(**request)

    assert actual.equals(expected)
    assert provider._business_day_calls == 1
    assert provider._snapshot_calls == snapshot_calls
    assert {
        item["cache_source"] for item in read_only.records
    } == {"workspace"}
    persisted = b"".join(
        path.read_bytes()
        for path in workspace.workspace_path.rglob("*")
        if path.is_file()
    )
    assert b"calendar-secret" not in persisted


def test_calendar_provider_uses_authoritative_midrange_replay_lookback(
    tmp_path,
    monkeypatch,
    _registered_provider,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _registered_provider(_CalendarSnapshotProvider())
    workspace = ResearchWorkspace.open("midrange_calendar_replay")
    calendar = Calendar.from_dates(
        [
            {
                "reference_date": "2026-05-22",
                "effective_date": "2026-06-01",
            },
            {
                "reference_date": "2026-06-19",
                "effective_date": "2026-07-01",
            },
        ]
    )
    calendar.provider_name = _PROVIDER_NAME
    calendar.calendar_id = "PRIMARY"
    calendar.provider_parameters = {
        "password": "calendar-replay-secret"
    }
    base = _spec(
        index_id="MIDRANGE_CALENDAR_REPLAY",
        cache_mode=CacheMode.OFF,
        simulation_cache_mode=CacheMode.OFF,
    )
    assert base.simulation is not None
    params = SimulationParams(
        index_drift=RelativeCapitalizationDrift(
            lookback_calendar_days=14
        ),
        benchmark_drift=RelativeCapitalizationDrift(
            lookback_calendar_days=14
        ),
        materialization=SimulationMaterialization(
            weight_snapshots=WeightSnapshotMode.NONE,
            include_asset_returns=False,
        ),
    )
    spec = replace(
        base,
        calendar=calendar,
        simulation=replace(
            base.simulation,
            start_date=pd.Timestamp("2026-07-02"),
            params=params,
        ),
    )

    run = workspace.run(spec)

    assert run.simulation is not None
    assert provider._business_day_calls == 1
    assert provider._business_day_requests == [
        (
            pd.Timestamp("2026-05-18"),
            pd.Timestamp("2026-07-10"),
        )
    ]
    assert (
        run.simulation.metadata["state_replayed_from"]
        == "2026-06-01"
    )


def test_missing_source_partition_is_rebuilt_but_corruption_is_not_trusted(
    tmp_path,
    monkeypatch,
    _registered_provider,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _registered_provider(_SnapshotProvider())
    workspace = ResearchWorkspace.open("source_integrity")
    spec = _spec(index_id="SOURCE_INTEGRITY", cache_mode=CacheMode.REUSE)

    workspace.run(spec)
    source_paths = _source_artifact_paths(workspace)
    assert len(source_paths) == 2
    source_paths[0].unlink()

    workspace.run(spec)
    assert provider._market_calls == 3
    repaired_paths = _source_artifact_paths(workspace)
    assert all(path.is_file() for path in repaired_paths)

    calls_before_corruption = provider._market_calls
    for path in repaired_paths:
        raw = bytearray(path.read_bytes())
        raw[len(raw) // 2] ^= 0x01
        path.write_bytes(bytes(raw))
    with pytest.raises(ArtifactIntegrityError):
        workspace.run(spec)
    # A preceding missing partition may be rebuilt before the corrupt binding
    # is reached, but the corrupt partition itself must never be recalculated
    # and silently rebound during the failing run.
    assert provider._market_calls <= calls_before_corruption + 1


def test_read_only_verifies_source_artifacts_before_simulation_cache_hit(
    tmp_path,
    monkeypatch,
    _registered_provider,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = _registered_provider(_SnapshotProvider())
    workspace = ResearchWorkspace.open("read_only_source_integrity")
    workspace.run(
        _spec(
            index_id="READ_ONLY_SOURCE_INTEGRITY",
            cache_mode=CacheMode.REUSE,
            simulation_cache_mode=CacheMode.REUSE,
        )
    )
    source_paths = _source_artifact_paths(workspace)
    assert source_paths
    source_paths[0].unlink()
    market_calls = provider._market_calls
    snapshot_calls = provider._snapshot_calls

    with pytest.raises(ManifestIntegrityError):
        workspace.run(
            _spec(
                index_id="READ_ONLY_SOURCE_INTEGRITY",
                cache_mode=CacheMode.READ_ONLY,
                simulation_cache_mode=CacheMode.READ_ONLY,
            )
        )

    assert provider._market_calls == market_calls
    assert provider._snapshot_calls == snapshot_calls
    assert workspace.latest().status == "failed"


def _source_artifact_paths(workspace: ResearchWorkspace):
    paths = []
    for binding_path in workspace.workspace_path.glob(
        "bindings/source_data/*/daily-market-data.json"
    ):
        envelope = json.loads(binding_path.read_text(encoding="utf-8"))
        relative = envelope["payload"]["artifact"]["relative_path"]
        paths.append(workspace.workspace_path.joinpath(relative))
    return sorted(set(paths), key=str)
