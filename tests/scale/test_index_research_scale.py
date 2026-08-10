"""Independent, opt-in scale benchmarks for the public research core.

The default test run only collects and skips this module. The ``full`` profile
executes the approved scale matrix. The ``quick`` profile validates the same
code paths with a deliberately small data set.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import pytest

try:
    import resource
except ImportError:  # pragma: no cover - resource is available on CI targets.
    resource = None

from icapa.backtesting import (
    BacktestResult,
    IndexSimulator,
    PriceReturnDrift,
    SimulationMaterialization,
    SimulationParams,
    WeightSnapshotMode,
)
from icapa.data_sources import register_provider, registry
from icapa.portfolio_construction import (
    Artifact,
    ArtifactKey,
    ArtifactOutput,
    ArtifactRequirement,
    CORE_TARGET_WEIGHTS,
    IndexRecipe,
    RecipeRunner,
    ReviewIdentity,
    StageCacheScope,
    StageCacheSource,
    StageDescriptor,
    StageNode,
    StageRequirements,
    StageResult,
    StageRuntime,
)
from icapa.workspace.caches.recipe import WorkspaceStageCache
from icapa.workspace.caches.source import SourceDataCacheLoader
from icapa.portfolio_construction.context import DataContext
from icapa.workspace import CacheMode
from icapa.workspace import WorkspaceRepository as ArtifactWorkspace
from icapa.workspace import WorkspaceStore


pytestmark = pytest.mark.scale

_SCALE_FEATURES = ArtifactKey("icapa.scale", "shared_features")


def _peak_rss_bytes() -> int:
    if resource is None:
        return 0
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if platform.system() == "Darwin" else raw * 1024


def _directory_bytes(path) -> int:
    if not path.exists():
        return 0
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _profile_dimensions(
    profile: str,
    *,
    instruments: int,
    years: int,
) -> tuple[int, int]:
    if profile == "full":
        return instruments, years
    return min(instruments, 250), 1


def _effective_dates(years: int, frequency: str) -> pd.DatetimeIndex:
    periods_per_year = {
        "monthly": 12,
        "quarterly": 4,
        "semi_annual": 2,
    }
    pandas_frequency = {
        "monthly": "BMS",
        "quarterly": "BQS-JAN",
        "semi_annual": "2BQS-JAN",
    }
    return pd.date_range(
        "2005-01-01",
        periods=years * periods_per_year[frequency],
        freq=pandas_frequency[frequency],
    ).normalize()


def _backtest_result(
    instrument_count: int,
    years: int,
    frequency: str,
) -> BacktestResult:
    instrument_ids = pd.Index(
        [f"I{offset:05d}" for offset in range(instrument_count)],
        name="instrument_id",
    )
    effective_dates = _effective_dates(years, frequency)
    target = np.full(instrument_count, 1.0 / instrument_count, dtype=np.float64)
    reviews: dict[pd.Timestamp, DataContext] = {}
    for effective_date in effective_dates:
        context = DataContext(
            reference_date=effective_date - pd.offsets.BDay(5),
            effective_date=effective_date,
            index_id="SCALE_RESEARCH_INDEX",
        )
        context.set_dataframe(
            pd.DataFrame(
                {
                    "instrument_id": instrument_ids,
                    "index_weight": target,
                    "benchmark_weight": target,
                }
            )
        )
        reviews[effective_date] = context

    index = pd.MultiIndex.from_product(
        (effective_dates, instrument_ids),
        names=("effective_date", "instrument_id"),
    )
    weights = pd.DataFrame(
        {"index_weight": np.tile(target, len(effective_dates))},
        index=index,
    )
    return BacktestResult(weights=weights, reviews=reviews)


class _PartitionedMarketDataProvider:
    """Generate deterministic market data one requested partition at a time."""

    def __init__(self) -> None:
        self._calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        self._rows_loaded = 0
        self._maximum_partition_rows = 0

    @property
    def calls(self):
        return tuple(self._calls)

    @property
    def rows_loaded(self):
        return self._rows_loaded

    @property
    def maximum_partition_rows(self):
        return self._maximum_partition_rows

    def describe_snapshot(self, capability, request):
        return {
            "capability": str(capability),
            "dataset": "deterministic_scale_market_data",
            "revision": 1,
        }

    def load_daily_market_data(
        self,
        instrument_ids,
        start_date,
        end_date,
        **kwargs,
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        self._calls.append((start, end))
        days = pd.bdate_range(start, end)
        ids = np.asarray(list(instrument_ids), dtype=object)
        instrument_count = len(ids)
        row_count = len(days) * instrument_count
        self._rows_loaded += row_count
        self._maximum_partition_rows = max(
            self._maximum_partition_rows,
            row_count,
        )
        if row_count == 0:
            return pd.DataFrame(
                columns=(
                    "instrument_id",
                    "business_date",
                    "price_return",
                    "gross_dividend",
                    "net_dividend",
                    "market_cap",
                )
            )

        instrument_number = np.tile(
            np.arange(instrument_count, dtype=np.float64),
            len(days),
        )
        day_number = np.repeat(
            days.to_numpy(dtype="datetime64[D]").astype(np.int64),
            instrument_count,
        ).astype(np.float64)
        price_return = (
            0.0001
            + ((instrument_number % 31.0) - 15.0) * 0.000001
            + (day_number % 7.0) * 0.0000001
        )
        market_cap = (
            1_000_000.0
            * (instrument_number + 1.0)
            * (1.0 + (day_number % 1000.0) * 0.000001)
        )
        return pd.DataFrame(
            {
                "instrument_id": np.tile(ids, len(days)),
                "business_date": np.repeat(days.to_numpy(), instrument_count),
                "price_return": price_return,
                "gross_dividend": np.zeros(row_count, dtype=np.float64),
                "net_dividend": np.zeros(row_count, dtype=np.float64),
                "market_cap": market_cap,
            }
        )


def _workspace_store(name: str) -> WorkspaceStore:
    return WorkspaceStore(
        workspace_name=name,
        fingerprint=_digest(f"{name}:fingerprint"),
        index_id="SCALE_RESEARCH_INDEX",
        methodology_name="public_scale_weight_producer",
        configuration_digest=_digest(f"{name}:configuration"),
        data_revision="scale-data-v1",
    )


def _simulation_params() -> SimulationParams:
    return SimulationParams(
        index_drift=PriceReturnDrift(),
        benchmark_drift=PriceReturnDrift(),
        materialization=SimulationMaterialization(
            weight_snapshots=WeightSnapshotMode.NONE,
            include_asset_returns=False,
        ),
    )


def _simulate(
    *,
    backtest: BacktestResult,
    provider_name: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    workspace: WorkspaceStore,
):
    source_loader = SourceDataCacheLoader(
        ArtifactWorkspace.open(workspace.workspace_name),
        provider_name=provider_name,
        provider_parameters={},
        mode=CacheMode.REUSE,
    )
    return IndexSimulator(
        backtest_result=backtest,
        market_data_provider_name=provider_name,
        start_date=start_date,
        end_date=end_date,
        params=_simulation_params(),
        data_revision="scale-data-v1",
        workspace=workspace,
        segmented_cache=True,
        streaming=True,
        market_data_loader=source_loader.load,
    ).run()


@pytest.mark.parametrize("instrument_count", (5_000, 10_000))
@pytest.mark.parametrize("years", (10, 20))
@pytest.mark.parametrize(
    "frequency",
    ("monthly", "quarterly", "semi_annual"),
)
def test_streaming_simulation_scale_matrix(
    instrument_count,
    years,
    frequency,
    scale_profile,
    record_scale_metric,
    tmp_path,
    monkeypatch,
):
    """Measure the approved instrument/history/rebalance matrix."""

    if scale_profile == "quick" and (
        instrument_count,
        years,
        frequency,
    ) != (5_000, 10, "monthly"):
        pytest.skip("the quick profile runs one representative matrix cell")
    actual_instruments, actual_years = _profile_dimensions(
        scale_profile,
        instruments=instrument_count,
        years=years,
    )
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    backtest = _backtest_result(
        actual_instruments,
        actual_years,
        frequency,
    )
    effective_dates = pd.DatetimeIndex(sorted(backtest.reviews))
    start_date = effective_dates[0]
    end_date = pd.Timestamp(start_date.year + actual_years - 1, 12, 31)
    provider = _PartitionedMarketDataProvider()
    provider_name = (
        f"scale_matrix_{instrument_count}_{years}_{frequency}"
    )
    register_provider(provider_name, provider, replace=True)
    store = _workspace_store(provider_name)
    workspace_path = store.workspace_path.parents[1]
    disk_before = _directory_bytes(workspace_path)
    started = perf_counter()
    try:
        result = _simulate(
            backtest=backtest,
            provider_name=provider_name,
            start_date=start_date,
            end_date=end_date,
            workspace=store,
        )
    finally:
        registry.unregister(provider_name)
    elapsed = perf_counter() - started
    disk_delta = _directory_bytes(workspace_path) - disk_before
    business_dates = len(result.daily)
    instrument_days = actual_instruments * business_dates

    assert business_dates > 0
    assert result.holdings.empty
    assert result.asset_returns.empty
    assert provider.maximum_partition_rows <= actual_instruments * 25
    assert np.isfinite(
        result.daily.select_dtypes(include=[np.number]).to_numpy()
    ).all()
    record_scale_metric(
        "streaming_simulation_matrix",
        requested_instruments=instrument_count,
        requested_years=years,
        actual_instruments=actual_instruments,
        actual_years=actual_years,
        rebalance_frequency=frequency,
        scenarios=1,
        business_dates=business_dates,
        provider_calls=len(provider.calls),
        provider_rows=provider.rows_loaded,
        maximum_partition_rows=provider.maximum_partition_rows,
        cache_hit_ratio=0.0,
        wall_seconds=elapsed,
        peak_rss_bytes=_peak_rss_bytes(),
        disk_delta_bytes=disk_delta,
        persisted_bytes_per_instrument_day=(
            disk_delta / instrument_days if instrument_days else 0.0
        ),
    )


def test_cold_warm_shorter_and_extended_segment_reuse(
    scale_profile,
    record_scale_metric,
    tmp_path,
    monkeypatch,
):
    """Measure exact, containing-range, and prefix-extension reuse."""

    instrument_count, years = (
        (5_000, 10) if scale_profile == "full" else (250, 2)
    )
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    backtest = _backtest_result(instrument_count, years, "monthly")
    effective_dates = pd.DatetimeIndex(sorted(backtest.reviews))
    start_date = effective_dates[0]
    full_end = pd.Timestamp(start_date.year + years - 1, 12, 31)
    business_dates = pd.bdate_range(start_date, full_end)
    shorter_end = business_dates[int(len(business_dates) * 0.55)]
    base_end = business_dates[int(len(business_dates) * 0.75)]
    provider = _PartitionedMarketDataProvider()
    provider_name = "scale_segment_reuse"
    register_provider(provider_name, provider, replace=True)
    store = _workspace_store(provider_name)
    workspace_path = store.workspace_path.parents[1]
    disk_before = _directory_bytes(workspace_path)
    timings: dict[str, float] = {}
    calls: dict[str, int] = {}

    def measured(label: str, end_date: pd.Timestamp):
        before = len(provider.calls)
        started = perf_counter()
        value = _simulate(
            backtest=backtest,
            provider_name=provider_name,
            start_date=start_date,
            end_date=end_date,
            workspace=store,
        )
        timings[label] = perf_counter() - started
        calls[label] = len(provider.calls) - before
        return value

    try:
        cold = measured("cold", base_end)
        warm = measured("warm", base_end)
        shorter = measured("shorter", shorter_end)
        extended = measured("extended", full_end)
    finally:
        registry.unregister(provider_name)

    assert calls["cold"] > 0
    assert calls["warm"] == 0
    assert calls["shorter"] == 0
    assert 0 < calls["extended"] < calls["cold"]
    pd.testing.assert_frame_equal(cold.daily, warm.daily)
    assert len(shorter.daily) < len(cold.daily) < len(extended.daily)
    assert extended.metadata.get("segment_reuse") == "extended_prefix"
    record_scale_metric(
        "segmented_cache_reuse",
        actual_instruments=instrument_count,
        actual_years=years,
        rebalance_frequency="monthly",
        scenarios=1,
        provider_calls=sum(calls.values()),
        provider_calls_by_run=calls,
        exact_or_slice_cache_hit_ratio=2.0 / 3.0,
        extension_reused_prefix=True,
        wall_seconds_by_run=timings,
        warm_speedup=(
            timings["cold"] / timings["warm"]
            if timings["warm"] > 0
            else 0.0
        ),
        peak_rss_bytes=_peak_rss_bytes(),
        disk_delta_bytes=(
            _directory_bytes(workspace_path) - disk_before
        ),
    )


@dataclass
class _SharedFeatureStage:
    instrument_count: int
    calls = 0

    @property
    def descriptor(self):
        return StageDescriptor(
            "scale.shared_feature_stage",
            cache_scope=StageCacheScope.CONTENT,
        )

    @property
    def requirements(self):
        return StageRequirements(review_dimensions=frozenset())

    @property
    def outputs(self):
        return (ArtifactOutput(_SCALE_FEATURES),)

    def canonical_configuration(self):
        return {"instrument_count": self.instrument_count}

    def run(self, inputs, runtime):
        type(self).calls += 1
        ids = pd.Index(
            [f"I{offset:05d}" for offset in range(self.instrument_count)],
            name="instrument_id",
        )
        frame = pd.DataFrame(
            {
                "benchmark_weight": np.full(
                    self.instrument_count,
                    1.0 / self.instrument_count,
                    dtype=np.float64,
                ),
                "signal": np.linspace(
                    -1.0,
                    1.0,
                    self.instrument_count,
                    dtype=np.float64,
                ),
            },
            index=ids,
        )
        return StageResult(
            {_SCALE_FEATURES: Artifact.from_value(_SCALE_FEATURES, frame)}
        )


@dataclass
class _ScenarioWeightStage:
    tilt: float
    calls = 0

    @property
    def descriptor(self):
        return StageDescriptor(
            "scale.scenario_weight_stage",
            cache_scope=StageCacheScope.CONTENT,
        )

    @property
    def requirements(self):
        return StageRequirements(
            artifacts=(ArtifactRequirement(_SCALE_FEATURES),),
            review_dimensions=frozenset(),
        )

    @property
    def outputs(self):
        return (ArtifactOutput(CORE_TARGET_WEIGHTS),)

    def canonical_configuration(self):
        return {"tilt": float(self.tilt)}

    def run(self, inputs, runtime):
        type(self).calls += 1
        frame = inputs.value(_SCALE_FEATURES)
        values = frame["benchmark_weight"] * np.exp(
            frame["signal"] * self.tilt
        )
        weights = (values / float(values.sum())).rename("index_weight")
        return StageResult(
            {
                CORE_TARGET_WEIGHTS: Artifact.from_value(
                    CORE_TARGET_WEIGHTS,
                    weights,
                )
            }
        )


def _scenario_recipe(instrument_count: int, tilt: float) -> IndexRecipe:
    return IndexRecipe(
        nodes=(
            StageNode(
                "shared_features",
                _SharedFeatureStage(instrument_count),
            ),
            StageNode(
                "scenario_weights",
                _ScenarioWeightStage(tilt),
            ),
        )
    )


def test_twenty_parameter_scenarios_reuse_shared_upstream_artifacts(
    scale_profile,
    record_scale_metric,
    tmp_path,
    monkeypatch,
):
    """Verify parameter changes do not duplicate unchanged feature stages."""

    instrument_count = 10_000 if scale_profile == "full" else 500
    scenario_count = 20 if scale_profile == "full" else 4
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = ArtifactWorkspace.open("scale_parameter_scenarios")
    cache = WorkspaceStageCache(workspace)
    runtime = StageRuntime(
        data_revision="scale-data-v1",
        code_revision="scale-suite-code-v1",
    )
    review = ReviewIdentity(
        index_id="SCALE_RESEARCH_INDEX",
        reference_date="2025-12-19",
        effective_date="2026-01-02",
    )
    tilts = np.linspace(-1.0, 1.0, scenario_count)
    _SharedFeatureStage.calls = 0
    _ScenarioWeightStage.calls = 0
    provider = _PartitionedMarketDataProvider()
    provider_name = "scale_scenario_source"
    register_provider(provider_name, provider, replace=True)
    cache_hits = 0
    stage_count = 0
    disk_before = _directory_bytes(workspace.workspace_path)
    started = perf_counter()

    try:
        instrument_ids = [
            f"I{offset:05d}" for offset in range(instrument_count)
        ]
        source_loader = SourceDataCacheLoader(
            workspace,
            provider_name=provider_name,
            provider_parameters={},
            mode=CacheMode.REUSE,
        )
        for _ in range(scenario_count):
            source = source_loader.load(
                instrument_ids=instrument_ids,
                start_date="2026-01-01",
                end_date="2026-01-31",
            )
            assert len(source) > 0

        first_recipe = None
        for tilt in tilts:
            recipe = _scenario_recipe(instrument_count, float(tilt))
            first_recipe = recipe if first_recipe is None else first_recipe
            result = RecipeRunner(
                cache=cache,
                runtime=runtime,
            ).run_review(recipe, review)
            assert np.isclose(
                float(result.target_weights.sum()),
                1.0,
                atol=1e-12,
            )
            stage_count += len(result.stages)
            cache_hits += sum(
                item.cache_source is StageCacheSource.CACHE
                for item in result.stages
            )

        repeated = RecipeRunner(
            cache=cache,
            runtime=runtime,
        ).run_review(first_recipe, review)
    finally:
        registry.unregister(provider_name)
    stage_count += len(repeated.stages)
    cache_hits += sum(
        item.cache_source is StageCacheSource.CACHE
        for item in repeated.stages
    )
    elapsed = perf_counter() - started

    assert _SharedFeatureStage.calls == 1
    assert _ScenarioWeightStage.calls == scenario_count
    assert len(provider.calls) == 1
    assert all(
        item.cache_source is StageCacheSource.CACHE
        for item in repeated.stages
    )
    record_scale_metric(
        "parameter_scenario_stage_reuse",
        actual_instruments=instrument_count,
        actual_years=0,
        rebalance_frequency="single_review",
        scenarios=scenario_count,
        provider_calls=len(provider.calls),
        source_cache_hit_ratio=(scenario_count - 1) / scenario_count,
        shared_feature_stage_calls=_SharedFeatureStage.calls,
        scenario_weight_stage_calls=_ScenarioWeightStage.calls,
        cache_hit_ratio=cache_hits / stage_count,
        wall_seconds=elapsed,
        peak_rss_bytes=_peak_rss_bytes(),
        disk_delta_bytes=(
            _directory_bytes(workspace.workspace_path) - disk_before
        ),
    )
