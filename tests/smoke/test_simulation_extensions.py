"""Smoke tests for explicit simulation strategies and additive segment reuse."""

from __future__ import annotations

import multiprocessing
import os

import numpy as np
import pandas as pd
import pytest

from icapa.backtesting import (
    BacktestResult,
    Calendar,
    CapitalizationDrift,
    DividendTreatment,
    IndexSimulator,
    PriceReturnDrift,
    RebalanceFrequency,
    RebalancePhase,
    RebalanceTiming,
    RelativeCapitalizationDrift,
    SimulationMaterialization,
    SimulationParams,
    WeightDrift,
    WeightSnapshotMode,
)
from icapa.backtesting.simulation import (
    ImmutableSimulationSegment,
    LegacyAbsoluteMarketCapDrift,
)
from icapa.data_sources import register_provider, registry
from icapa.portfolio_construction.context import DataContext
from icapa.workspace import (
    WorkspaceStore,
    automatic_digest,
    clear_memory_cache,
    dataframe_content_digest,
)


class StrategyMarketProvider:
    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def load_daily_market_data(
        self,
        instrument_ids,
        start_date,
        end_date,
        **kwargs,
    ):
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        type(self).calls.append((start, end))
        dates = pd.bdate_range(start, end)
        origin = pd.Timestamp("2026-01-01")
        rows = []
        for business_date in dates:
            elapsed = len(pd.bdate_range(origin, business_date)) - 1
            for instrument_id in instrument_ids:
                is_first = instrument_id == "A"
                rows.append(
                    {
                        "instrument_id": instrument_id,
                        "business_date": business_date,
                        "price_return": 0.10 if is_first else 0.0,
                        "gross_dividend": 0.0,
                        "net_dividend": 0.0,
                        "market_cap": (
                            100.0 * (1.10**elapsed)
                            if is_first
                            else 100.0
                        ),
                    }
                )
        return pd.DataFrame.from_records(rows)


class MutableSegmentMarketProvider:
    def __init__(self) -> None:
        self.early_return = 0.10
        self.calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def frame(self, instrument_ids, start_date, end_date):
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        rows = []
        for business_date in pd.bdate_range(start, end):
            for instrument_id in instrument_ids:
                rows.append(
                    {
                        "instrument_id": instrument_id,
                        "business_date": business_date,
                        "price_return": (
                            self.early_return
                            if (
                                business_date
                                == pd.Timestamp("2026-01-02")
                                and instrument_id == "A"
                            )
                            else 0.0
                        ),
                        "gross_dividend": 0.0,
                        "net_dividend": 0.0,
                        "market_cap": 100.0,
                    }
                )
        return pd.DataFrame.from_records(rows)

    def load_daily_market_data(
        self,
        instrument_ids,
        start_date,
        end_date,
        **kwargs,
    ):
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        self.calls.append((start, end))
        return self.frame(instrument_ids, start, end)


@pytest.fixture
def strategy_provider():
    StrategyMarketProvider.calls = []
    register_provider(
        "simulation_strategy_provider",
        StrategyMarketProvider(),
        replace=True,
    )
    yield
    registry.unregister("simulation_strategy_provider")


@pytest.fixture
def mutable_segment_provider():
    provider = MutableSegmentMarketProvider()
    register_provider(
        "mutable_segment_provider",
        provider,
        replace=True,
    )
    yield provider
    registry.unregister("mutable_segment_provider")


def _context(effective_date: str, index_weights) -> DataContext:
    context = DataContext(
        reference_date=pd.Timestamp(effective_date) - pd.Timedelta(days=1),
        effective_date=effective_date,
        index_id="STRATEGY_INDEX",
    )
    context.set_dataframe(
        pd.DataFrame(
            {
                "instrument_id": ["A", "B"],
                "index_weight": index_weights,
                "benchmark_weight": [0.5, 0.5],
            }
        )
    )
    return context


def _backtest_result() -> BacktestResult:
    reviews = {
        pd.Timestamp("2026-01-02"): _context("2026-01-02", [0.8, 0.2]),
        pd.Timestamp("2026-01-05"): _context("2026-01-05", [0.2, 0.8]),
    }
    weights = pd.concat(
        [
            context.cons[["index_weight"]]
            .assign(effective_date=effective_date)
            .reset_index()
            for effective_date, context in reviews.items()
        ],
        ignore_index=True,
    ).set_index(["effective_date", "instrument_id"])
    return BacktestResult(weights=weights, reviews=reviews)


def _single_review_result(index_weights) -> BacktestResult:
    effective_date = pd.Timestamp("2026-01-02")
    context = _context("2026-01-02", index_weights)
    weights = (
        context.cons[["index_weight"]]
        .assign(effective_date=effective_date)
        .reset_index()
        .set_index(["effective_date", "instrument_id"])
    )
    return BacktestResult(weights=weights, reviews={effective_date: context})


def _concurrent_segment_coverage_writer(
    workspace_root: str,
    workspace_name: str,
    variant: int,
    barrier,
    outcomes,
) -> None:
    try:
        os.environ["ICAPA_WORKSPACE_ROOT"] = workspace_root
        store = WorkspaceStore(
            workspace_name=workspace_name,
            fingerprint="7" * 64,
            index_id="STRATEGY_INDEX",
            methodology_name="strategy_test",
            configuration_digest="8" * 64,
            data_revision="market-v1",
        )
        simulator = IndexSimulator(
            backtest_result=_single_review_result([0.8, 0.2]),
            market_data_provider_name="simulation_strategy_provider",
            start_date="2026-01-02",
            end_date="2026-01-31",
            workspace=store,
            data_revision="market-v1",
            segmented_cache=True,
        )
        segment_date = pd.Timestamp("2026-01-02") + pd.Timedelta(
            days=variant
        )
        segment = ImmutableSimulationSegment(
            start_date=segment_date,
            end_date=segment_date,
            effective_date="2026-01-02",
            next_effective_date=None,
            kind="open_tail_checkpoint",
            target_checksum="9" * 64,
            previous_target_checksum=None,
        )
        cache_key = automatic_digest({"variant": variant})
        barrier.wait(timeout=30)
        simulator._record_immutable_segment(
            segment,
            cache_key=cache_key,
            partition_digest=None,
            opening_checkpoint=None,
        )
        outcomes.put(("saved", variant, cache_key))
    except BaseException as exc:  # pragma: no cover - failure diagnostics.
        outcomes.put(("error", variant, repr(exc)))
        raise


def _simulator(**kwargs) -> IndexSimulator:
    start_date = kwargs.pop("start_date", "2026-01-02")
    end_date = kwargs.pop("end_date", "2026-01-06")
    return IndexSimulator(
        backtest_result=_backtest_result(),
        market_data_provider_name="simulation_strategy_provider",
        start_date=start_date,
        end_date=end_date,
        **kwargs,
    )


def _segment_lineage(
    provider: MutableSegmentMarketProvider,
) -> tuple[dict[str, object], ...]:
    instruments = ("A", "B")
    result = []
    for start_date, end_date in (
        ("2026-01-02", "2026-01-04"),
        ("2026-01-05", "2026-01-06"),
    ):
        frame = provider.frame(instruments, start_date, end_date)
        result.append(
            {
                "input_type": "source_daily_market_data",
                "provider_name": "mutable_segment_provider",
                "capability": "load_daily_market_data",
                "request_digest": automatic_digest(
                    {
                        "instrument_ids": instruments,
                        "start_date": start_date,
                        "end_date": end_date,
                    }
                ),
                "content_digest": dataframe_content_digest(
                    frame,
                    sort_by=["business_date", "instrument_id"],
                ),
                "rows": len(frame),
                "start_date": start_date,
                "end_date": end_date,
                "instrument_set_digest": automatic_digest(
                    list(instruments)
                ),
            }
        )
    return tuple(result)


def test_separate_index_and_benchmark_drift_strategies(strategy_provider):
    result = _simulator(
        end_date="2026-01-02",
        params=SimulationParams(
            index_drift=PriceReturnDrift(),
            benchmark_drift=CapitalizationDrift(),
        ),
    ).run()

    closing = result.holdings.xs(pd.Timestamp("2026-01-02"))
    assert closing.loc["A", "index_closing_weight"] == pytest.approx(
        0.88 / 1.08
    )
    assert closing.loc["A", "benchmark_closing_weight"] == pytest.approx(
        110.0 / 210.0
    )

    legacy = SimulationParams(weight_drift=WeightDrift.MARKET_CAP)
    assert isinstance(
        legacy.resolved_index_drift,
        LegacyAbsoluteMarketCapDrift,
    )
    assert isinstance(
        legacy.resolved_benchmark_drift,
        LegacyAbsoluteMarketCapDrift,
    )
    with pytest.raises(ValueError, match="exact v1 cache replay"):
        _simulator(
            end_date="2026-01-02",
            params=legacy,
        ).run()


def test_capitalization_drift_validates_target_at_phase_appropriate_cap(
    strategy_provider,
):
    current_cap_weights = [110.0 / 210.0, 100.0 / 210.0]
    arguments = {
        "backtest_result": _single_review_result(current_cap_weights),
        "market_data_provider_name": "simulation_strategy_provider",
        "start_date": "2026-01-02",
        "end_date": "2026-01-02",
    }

    with pytest.raises(ValueError, match="not capitalization weighted"):
        IndexSimulator(
            **arguments,
            params=SimulationParams(
                rebalance_phase=RebalancePhase.OPEN,
                index_drift=CapitalizationDrift(),
            ),
        ).run()

    close_result = IndexSimulator(
        **arguments,
        params=SimulationParams(
            rebalance_phase=RebalancePhase.CLOSE,
            index_drift=CapitalizationDrift(),
        ),
    ).run()
    close_weights = close_result.holdings.xs(pd.Timestamp("2026-01-02"))
    assert close_weights.loc["A", "index_closing_weight"] == pytest.approx(
        current_cap_weights[0]
    )


def test_relative_capitalization_uses_explicit_prior_observation(
    strategy_provider,
):
    result = _simulator(
        end_date="2026-01-02",
        params=SimulationParams(
            index_drift=RelativeCapitalizationDrift(),
            benchmark_drift=RelativeCapitalizationDrift(),
        ),
    ).run()

    assert StrategyMarketProvider.calls[0][0] < pd.Timestamp("2026-01-02")
    closing = result.holdings.xs(pd.Timestamp("2026-01-02"))
    assert closing.loc["A", "index_closing_weight"] == pytest.approx(
        0.88 / 1.08
    )


def test_mid_period_start_replays_from_prior_effective_date(
    strategy_provider,
):
    result = IndexSimulator(
        backtest_result=_single_review_result([0.5, 0.5]),
        market_data_provider_name="simulation_strategy_provider",
        start_date="2026-01-05",
        end_date="2026-01-05",
    ).run()

    opening_weight_a = 0.55 / 1.05
    expected_return = opening_weight_a * 0.10
    assert StrategyMarketProvider.calls == [
        (pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05"))
    ]
    assert result.daily.index.tolist() == [pd.Timestamp("2026-01-05")]
    assert result.daily.iloc[0]["index_price_return"] == pytest.approx(
        expected_return
    )
    assert result.daily.iloc[0]["index_price_level"] == pytest.approx(
        100.0 * (1.0 + expected_return)
    )
    assert result.rebalances.empty
    assert result.metadata["state_replayed_from"] == "2026-01-02"


def test_rebalance_phase_changes_which_weights_receive_daily_return(
    strategy_provider,
):
    before = _simulator(
        params=SimulationParams(
            rebalance_phase=RebalancePhase.OPEN,
        )
    ).run()
    after = _simulator(
        params=SimulationParams(
            rebalance_phase=RebalancePhase.CLOSE,
        )
    ).run()

    date = pd.Timestamp("2026-01-05")
    assert before.daily.loc[date, "index_price_return"] == pytest.approx(0.02)
    assert after.daily.loc[date, "index_price_return"] > 0.08
    after_close = after.holdings.xs(date)
    assert after_close.loc["A", "index_closing_weight"] == pytest.approx(0.2)
    assert after.rebalances.iloc[0]["index_turnover"] != after.rebalances.iloc[0][
        "index_turnover"
    ]


@pytest.mark.parametrize(
    ("mode", "expected_dates"),
    (
        (WeightSnapshotMode.NONE, 0),
        (WeightSnapshotMode.REBALANCE, 2),
        (WeightSnapshotMode.DAILY, 3),
    ),
)
def test_weight_snapshot_materialization_modes(
    strategy_provider,
    mode,
    expected_dates,
):
    result = _simulator(
        params=SimulationParams(
            materialization=SimulationMaterialization(
                weight_snapshots=mode,
                include_asset_returns=False,
            )
        )
    ).run()

    observed = (
        0
        if result.holdings.empty
        else result.holdings.index.get_level_values("business_date").nunique()
    )
    assert observed == expected_dates
    assert result.asset_returns.empty
    assert result.checkpoint is not None
    assert result.weight_snapshots.index.names == [
        "applied_business_date",
        "snapshot",
        "instrument_id",
    ]
    assert result.rebalance_weight_snapshots is result.weight_snapshots
    if mode is WeightSnapshotMode.NONE:
        assert result.weight_snapshots.empty
    else:
        assert set(
            result.weight_snapshots.index.get_level_values("snapshot")
        ) == {"pre_rebalance", "target", "end_of_day"}
        assert (
            result.weight_snapshots.index.get_level_values(
                "applied_business_date"
            ).nunique()
            == 2
        )


def test_rebalance_weight_snapshots_have_explicit_lifecycle_semantics(
    strategy_provider,
):
    result = _simulator(
        params=SimulationParams(
            materialization=SimulationMaterialization(
                weight_snapshots=WeightSnapshotMode.REBALANCE,
                include_asset_returns=False,
            )
        )
    ).run()

    applied_date = pd.Timestamp("2026-01-05")
    snapshots = result.weight_snapshots.xs(
        applied_date,
        level="applied_business_date",
    )
    pre_rebalance = snapshots.xs("pre_rebalance", level="snapshot")
    target = snapshots.xs("target", level="snapshot")
    end_of_day = snapshots.xs("end_of_day", level="snapshot")

    assert pre_rebalance["index_weight"].sum() == pytest.approx(1.0)
    assert target.loc["A", "index_weight"] == pytest.approx(0.2)
    assert target.loc["B", "index_weight"] == pytest.approx(0.8)
    assert end_of_day.loc["A", "index_weight"] == pytest.approx(0.22 / 1.02)
    assert pre_rebalance.loc["A", "index_weight"] > target.loc[
        "A", "index_weight"
    ]
    assert (
        target["scheduled_effective_date"].unique().tolist()
        == [applied_date]
    )
    assert set(target["rebalance_phase"]) == {"before_return"}


def test_close_phase_end_of_day_snapshot_is_post_rebalance_target(
    strategy_provider,
):
    result = _simulator(
        params=SimulationParams(
            rebalance_phase=RebalancePhase.CLOSE,
            materialization=SimulationMaterialization(
                weight_snapshots=WeightSnapshotMode.REBALANCE,
                include_asset_returns=False,
            ),
        )
    ).run()

    snapshots = result.weight_snapshots.xs(
        pd.Timestamp("2026-01-05"),
        level="applied_business_date",
    )
    target = snapshots.xs("target", level="snapshot")
    end_of_day = snapshots.xs("end_of_day", level="snapshot")
    pd.testing.assert_series_equal(
        end_of_day["index_weight"],
        target["index_weight"],
        check_names=False,
    )
    assert set(target["rebalance_phase"]) == {"after_return"}


def test_frequency_generation_is_explicit_and_schedule_remains_authoritative():
    business_days = pd.bdate_range("2025-12-15", "2026-03-31")
    calendar = Calendar.from_frequency(
        start_date="2026-01-01",
        end_date="2026-03-31",
        frequency=RebalanceFrequency.MONTHLY,
        reference_lag_business_days=2,
        business_days=business_days,
    )

    assert calendar.dates["effective_date"].tolist() == [
        pd.Timestamp("2026-01-30"),
        pd.Timestamp("2026-02-27"),
        pd.Timestamp("2026-03-31"),
    ]
    assert (
        calendar.dates["effective_date"]
        - calendar.dates["reference_date"]
    ).dt.days.ge(2).all()
    calendar.validate_frequency(RebalanceFrequency.MONTHLY)

    manual = Calendar.from_dates(
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
    with pytest.raises(ValueError, match="multiple effective dates"):
        manual.validate_frequency(RebalanceFrequency.MONTHLY)
    manual.validate_frequency(RebalanceFrequency.CUSTOM)
    with pytest.raises(ValueError, match="Calendar.from_dates"):
        Calendar.from_frequency(
            start_date="2026-01-01",
            end_date="2026-03-31",
            frequency=RebalanceFrequency.CUSTOM,
        )


def test_frequency_diagnostics_report_gaps_without_rewriting_schedule():
    calendar = Calendar.from_dates(
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

    diagnostics = calendar.frequency_diagnostics(
        RebalanceFrequency.MONTHLY
    )

    assert diagnostics == (
        {
            "code": "missing_rebalance_periods",
            "severity": "warning",
            "frequency": "monthly",
            "previous_effective_date": pd.Timestamp("2026-01-30"),
            "next_effective_date": pd.Timestamp("2026-03-31"),
            "missing_period_count": 1,
        },
    )
    calendar.validate_frequency(RebalanceFrequency.MONTHLY)
    with pytest.raises(ValueError, match="missing monthly periods"):
        calendar.validate_frequency(
            RebalanceFrequency.MONTHLY,
            allow_gaps=False,
        )


def test_compatibility_aliases_and_positional_simulation_params():
    assert RebalancePhase.OPEN is RebalancePhase.BEFORE_RETURN
    assert RebalancePhase.CLOSE is RebalancePhase.AFTER_RETURN
    assert (
        WeightSnapshotMode.REBALANCE
        is WeightSnapshotMode.REBALANCE_DATES
    )
    params = SimulationParams(
        DividendTreatment.STANDARD,
        WeightDrift.PRICE_RETURN,
        RebalanceTiming.EXACT_DATE,
        1_000.0,
    )
    assert params.base_value == 1_000.0
    assert params.rebalance_phase is RebalancePhase.OPEN


def test_default_configuration_uses_standard_cache_and_metadata_values(
    strategy_provider,
):
    simulator = _simulator(end_date="2026-01-02")
    expected_params = {
        "dividend_treatment": "standard",
        "weight_drift": "price_return",
        "rebalance_timing": "next_business_day",
        "base_value": 100.0,
    }

    assert simulator._cache_key_payload()["schema"] == 1
    assert simulator._cache_key_payload()["params"] == expected_params
    assert simulator.run().metadata["simulation_params"] == expected_params


def test_legacy_market_cap_exact_v1_cache_can_still_be_replayed(
    strategy_provider,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    store = WorkspaceStore(
        workspace_name="legacy_market_cap_replay",
        fingerprint="a" * 64,
        index_id="STRATEGY_INDEX",
        methodology_name="strategy_test",
        configuration_digest="b" * 64,
        data_revision="market-v1",
    )
    arguments = {
        "backtest_result": _backtest_result(),
        "market_data_provider_name": "simulation_strategy_provider",
        "start_date": "2026-01-02",
        "end_date": "2026-01-02",
        "workspace": store,
        "data_revision": "market-v1",
        "params": SimulationParams(weight_drift=WeightDrift.MARKET_CAP),
    }
    writer = IndexSimulator(**arguments)
    cache_key = writer._cache_key()
    market_data = writer._load_market_data(
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-02"),
    )
    writer._save_cached(
        cache_key,
        writer._simulate(market_data, cache_key),
    )
    calls_before_replay = len(StrategyMarketProvider.calls)

    replayed = IndexSimulator(**arguments).run()

    assert replayed.metadata["cache_source"] == "workspace"
    assert len(StrategyMarketProvider.calls) == calls_before_replay


def test_segment_cache_extends_and_slices_same_start_range(
    strategy_provider,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    store = WorkspaceStore(
        workspace_name="segmented_simulation",
        fingerprint="c" * 64,
        index_id="STRATEGY_INDEX",
        methodology_name="strategy_test",
        configuration_digest="d" * 64,
        data_revision="market-v1",
    )
    common = {
        "backtest_result": _backtest_result(),
        "market_data_provider_name": "simulation_strategy_provider",
        "start_date": "2026-01-02",
        "workspace": store,
        "data_revision": "market-v1",
        "segmented_cache": True,
    }

    short = IndexSimulator(end_date="2026-01-05", **common).run()
    extended = IndexSimulator(end_date="2026-01-06", **common).run()
    assert len(StrategyMarketProvider.calls) == 2
    assert StrategyMarketProvider.calls[1][0] == pd.Timestamp("2026-01-06")
    assert extended.metadata["segment_reuse"] == "extended_prefix"

    full = _simulator().run()
    pd.testing.assert_frame_equal(extended.daily, full.daily)
    pd.testing.assert_frame_equal(extended.holdings, full.holdings)
    pd.testing.assert_frame_equal(
        extended.weight_snapshots,
        full.weight_snapshots,
    )
    pd.testing.assert_frame_equal(
        extended.rebalances.reset_index(drop=True),
        full.rebalances.reset_index(drop=True),
    )

    calls_before_slice = len(StrategyMarketProvider.calls)
    one_day = IndexSimulator(end_date="2026-01-02", **common).run()
    assert len(StrategyMarketProvider.calls) == calls_before_slice
    assert len(one_day.daily) == 1
    assert one_day.metadata["segment_reuse"] == "containing_prefix"
    assert not short.daily.empty


def test_segment_coverage_merges_concurrent_process_updates(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace_name = "concurrent_segment_coverage"
    store = WorkspaceStore(
        workspace_name=workspace_name,
        fingerprint="7" * 64,
        index_id="STRATEGY_INDEX",
        methodology_name="strategy_test",
        configuration_digest="8" * 64,
        data_revision="market-v1",
    )
    simulator = IndexSimulator(
        backtest_result=_single_review_result([0.8, 0.2]),
        market_data_provider_name="simulation_strategy_provider",
        start_date="2026-01-02",
        end_date="2026-01-31",
        workspace=store,
        data_revision="market-v1",
        segmented_cache=True,
    )
    namespace = simulator._immutable_segment_namespace()
    store.save_json(
        "simulation_segments",
        namespace,
        "coverage",
        {"schema_version": 2, "segments": []},
    )

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(4)
    outcomes = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_segment_coverage_writer,
            args=(
                str(tmp_path),
                workspace_name,
                variant,
                barrier,
                outcomes,
            ),
        )
        for variant in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=45)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)

    observed = [outcomes.get(timeout=10) for _ in processes]
    outcomes.close()
    outcomes.join_thread()
    assert all(process.exitcode == 0 for process in processes), observed
    assert all(item[0] == "saved" for item in observed), observed

    clear_memory_cache()
    coverage = store.load_json(
        "simulation_segments",
        namespace,
        "coverage",
    )
    assert {
        item["cache_key"] for item in coverage["segments"]
    } == {item[2] for item in observed}


def test_base_value_is_applied_after_reusable_simulation_calculation(
    strategy_provider,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    store = WorkspaceStore(
        workspace_name="base_value_reuse",
        fingerprint="e" * 64,
        index_id="STRATEGY_INDEX",
        methodology_name="strategy_test",
        configuration_digest="f" * 64,
        data_revision="market-v1",
    )
    materialization = SimulationMaterialization(
        weight_snapshots=WeightSnapshotMode.NONE,
        include_asset_returns=False,
    )
    common = {
        "backtest_result": _backtest_result(),
        "market_data_provider_name": "simulation_strategy_provider",
        "start_date": "2026-01-02",
        "end_date": "2026-01-06",
        "workspace": store,
        "data_revision": "market-v1",
        "segmented_cache": True,
    }

    base_100 = IndexSimulator(
        **common,
        params=SimulationParams(
            base_value=100.0,
            materialization=materialization,
        ),
    ).run()
    first_simulator = IndexSimulator(
        **common,
        params=SimulationParams(
            base_value=100.0,
            materialization=materialization,
        ),
    )
    coverage = store.load_json(
        "simulation_segments",
        first_simulator._immutable_segment_namespace(),
        "coverage",
    )
    segment_key = coverage["segments"][0]["cache_key"]
    cached_daily = store.load_frame(
        "simulation",
        segment_key,
        "daily",
    )
    cached_metadata = store.load_json(
        "simulation",
        segment_key,
        "metadata",
    )
    calls_after_first = len(StrategyMarketProvider.calls)
    base_1_000 = IndexSimulator(
        **common,
        params=SimulationParams(
            base_value=1_000.0,
            materialization=materialization,
        ),
    ).run()

    assert len(StrategyMarketProvider.calls) == calls_after_first
    pd.testing.assert_frame_equal(
        base_100.daily.filter(like="_return"),
        base_1_000.daily.filter(like="_return"),
    )
    np.testing.assert_allclose(
        base_1_000.daily.filter(like="_level"),
        base_100.daily.filter(like="_level") * 10.0,
        atol=1e-10,
    )
    for level_column in cached_daily.filter(like="_level"):
        return_column = (
            level_column.removesuffix("_level") + "_return"
        )
        np.testing.assert_allclose(
            cached_daily[level_column],
            (1.0 + cached_daily[return_column]).cumprod(),
            atol=1e-12,
        )
    assert cached_metadata["cached_level_representation"] == "return_factor"
    assert cached_metadata["simulation_params"]["base_value"] == 1.0


def test_midrange_start_reuses_prior_effective_state_and_turnover(
    strategy_provider,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    store = WorkspaceStore(
        workspace_name="midrange_effective_state",
        fingerprint="7" * 64,
        index_id="STRATEGY_INDEX",
        methodology_name="strategy_test",
        configuration_digest="8" * 64,
        data_revision="market-v1",
    )
    params = SimulationParams(
        materialization=SimulationMaterialization(
            weight_snapshots=WeightSnapshotMode.NONE,
            include_asset_returns=False,
        )
    )
    common = {
        "backtest_result": _backtest_result(),
        "market_data_provider_name": "simulation_strategy_provider",
        "end_date": "2026-01-06",
        "workspace": store,
        "data_revision": "market-v1",
        "segmented_cache": True,
        "params": params,
    }

    full = IndexSimulator(
        **common,
        start_date="2026-01-02",
    ).run()
    calls_after_full = len(StrategyMarketProvider.calls)
    midrange = IndexSimulator(
        **common,
        start_date="2026-01-05",
    ).run()

    assert len(StrategyMarketProvider.calls) == calls_after_full
    assert midrange.metadata["immutable_segments_reused"] == 2
    assert midrange.metadata["immutable_segments_computed"] == 0
    pd.testing.assert_frame_equal(
        midrange.daily.filter(like="_return"),
        full.daily.loc[
            pd.Timestamp("2026-01-05") :
        ].filter(like="_return"),
    )
    expected_rebalances = full.rebalances.loc[
        pd.to_datetime(full.rebalances["applied_business_date"])
        >= pd.Timestamp("2026-01-05")
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        midrange.rebalances.reset_index(drop=True),
        expected_rebalances,
    )
    assert (
        midrange.rebalances.iloc[0]["index_turnover"]
        == pytest.approx(
            expected_rebalances.iloc[0]["index_turnover"],
            abs=1e-12,
        )
    )


def test_changed_earlier_partition_invalidates_downstream_state_chain(
    mutable_segment_provider,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    store = WorkspaceStore(
        workspace_name="state_chain_invalidation",
        fingerprint="9" * 64,
        index_id="STRATEGY_INDEX",
        methodology_name="strategy_test",
        configuration_digest="a" * 64,
        data_revision="content-verified",
    )
    params = SimulationParams(
        materialization=SimulationMaterialization(
            weight_snapshots=WeightSnapshotMode.NONE,
            include_asset_returns=False,
        )
    )
    common = {
        "backtest_result": _backtest_result(),
        "market_data_provider_name": "mutable_segment_provider",
        "start_date": "2026-01-02",
        "end_date": "2026-01-06",
        "workspace": store,
        "data_revision": "content-verified",
        "segmented_cache": True,
        "streaming": True,
        "params": params,
    }

    first = IndexSimulator(
        **common,
        market_data_lineage=_segment_lineage(
            mutable_segment_provider
        ),
    ).run()
    first_turnover = float(
        first.rebalances.loc[
            first.rebalances["scheduled_effective_date"]
            == pd.Timestamp("2026-01-05"),
            "index_turnover",
        ].iloc[0]
    )
    mutable_segment_provider.early_return = 0.50
    second = IndexSimulator(
        **common,
        market_data_lineage=_segment_lineage(
            mutable_segment_provider
        ),
    ).run()
    second_turnover = float(
        second.rebalances.loc[
            second.rebalances["scheduled_effective_date"]
            == pd.Timestamp("2026-01-05"),
            "index_turnover",
        ].iloc[0]
    )

    assert len(mutable_segment_provider.calls) == 4
    assert second.metadata["immutable_segments_reused"] == 0
    assert second.metadata["immutable_segments_computed"] == 2
    assert second_turnover != pytest.approx(first_turnover, abs=1e-12)


def test_preloaded_shared_market_data_and_optional_calendar_check(
    strategy_provider,
):
    provider = StrategyMarketProvider()
    market_data = provider.load_daily_market_data(
        instrument_ids=["A", "B"],
        start_date="2026-01-02",
        end_date="2026-01-06",
    )
    StrategyMarketProvider.calls = []
    business_days = pd.bdate_range("2026-01-02", "2026-01-06")

    result = _simulator(
        market_data=market_data,
        business_days=business_days,
    ).run()

    assert not result.daily.empty
    assert StrategyMarketProvider.calls == []

    incomplete = market_data.loc[
        market_data["business_date"] != pd.Timestamp("2026-01-06")
    ]
    with pytest.raises(ValueError, match="business-date coverage"):
        _simulator(
            market_data=incomplete,
            business_days=business_days,
        ).run()
