"""Parity tests for opt-in calendar-month simulation streaming."""

from __future__ import annotations

import pandas as pd
import pytest

from icapa.backtesting import (
    BacktestResult,
    CapitalizationDrift,
    IndexSimulator,
    PriceReturnDrift,
    RebalancePhase,
    RelativeCapitalizationDrift,
    SimulationParams,
    WeightDrift,
)
from icapa.data_sources import register_provider, registry
from icapa.portfolio_construction.context import DataContext
from icapa.workspace import WorkspaceStore


_DAILY_COLUMNS = [
    "instrument_id",
    "business_date",
    "price_return",
    "gross_dividend",
    "net_dividend",
    "market_cap",
]


def _review_context(effective_date: str) -> DataContext:
    context = DataContext(
        reference_date=pd.Timestamp(effective_date) - pd.Timedelta(days=1),
        effective_date=effective_date,
        index_id="STREAMING_DEMO",
    )
    context.set_dataframe(
        pd.DataFrame(
            {
                "instrument_id": ["A", "B"],
                "index_weight": [2.0 / 3.0, 1.0 / 3.0],
                "benchmark_weight": [2.0 / 3.0, 1.0 / 3.0],
            }
        )
    )
    return context


def _backtest_result() -> BacktestResult:
    reviews = {
        pd.Timestamp(effective_date): _review_context(effective_date)
        for effective_date in ("2026-01-27", "2026-01-31", "2026-02-28")
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


def _market_data(start_date, end_date) -> pd.DataFrame:
    rows = []
    origin = pd.Timestamp("2026-01-01")
    for business_date in pd.bdate_range(start_date, end_date):
        position = len(pd.bdate_range(origin, business_date)) - 1
        rows.extend(
            [
                {
                    "instrument_id": "A",
                    "business_date": business_date,
                    "price_return": 0.001 + position * 0.000001,
                    "gross_dividend": 0.0001,
                    "net_dividend": 0.00008,
                    "market_cap": 200.0,
                },
                {
                    "instrument_id": "B",
                    "business_date": business_date,
                    "price_return": -0.0004 + position * 0.0000005,
                    "gross_dividend": 0.00005,
                    "net_dividend": 0.00004,
                    "market_cap": 100.0,
                },
            ]
        )
    return pd.DataFrame.from_records(rows, columns=_DAILY_COLUMNS)


def _params(strategy_name: str, phase: RebalancePhase) -> SimulationParams:
    if strategy_name == "legacy_capitalization":
        return SimulationParams(
            weight_drift=WeightDrift.MARKET_CAP,
            rebalance_phase=phase,
        )
    strategies = {
        "price_return": PriceReturnDrift,
        "capitalization": CapitalizationDrift,
        "relative_capitalization": RelativeCapitalizationDrift,
    }
    strategy_type = strategies[strategy_name]
    return SimulationParams(
        index_drift=strategy_type(),
        benchmark_drift=strategy_type(),
        rebalance_phase=phase,
    )


def _assert_simulation_equal(
    left,
    right,
) -> None:
    for left_frame, right_frame in (
        (left.daily, right.daily),
        (left.holdings, right.holdings),
        (left.asset_returns, right.asset_returns),
        (left.weight_snapshots, right.weight_snapshots),
    ):
        pd.testing.assert_frame_equal(
            left_frame,
            right_frame,
            check_exact=False,
            rtol=0.0,
            atol=1e-10,
        )
    pd.testing.assert_frame_equal(
        left.rebalances.reset_index(drop=True),
        right.rebalances.reset_index(drop=True),
        check_exact=False,
        rtol=0.0,
        atol=1e-10,
    )
    assert left.checkpoint is not None
    assert right.checkpoint is not None
    pd.testing.assert_series_equal(
        left.checkpoint.index_weights,
        right.checkpoint.index_weights,
        check_exact=False,
        rtol=0.0,
        atol=1e-10,
    )
    pd.testing.assert_series_equal(
        left.checkpoint.benchmark_weights,
        right.checkpoint.benchmark_weights,
        check_exact=False,
        rtol=0.0,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "strategy_name",
    (
        "price_return",
        "capitalization",
        "relative_capitalization",
    ),
)
@pytest.mark.parametrize(
    "phase",
    (RebalancePhase.OPEN, RebalancePhase.CLOSE),
)
def test_streaming_matches_full_calculation_for_every_drift_and_phase(
    strategy_name,
    phase,
):
    market_data = _market_data("2026-01-01", "2026-03-06")
    common = {
        "backtest_result": _backtest_result(),
        "market_data_provider_name": "preloaded",
        "start_date": "2026-01-27",
        "end_date": "2026-03-06",
        "market_data": market_data,
        "business_days": pd.bdate_range("2026-01-27", "2026-03-06"),
        "params": _params(strategy_name, phase),
    }

    full_simulator = IndexSimulator(**common)
    streaming_simulator = IndexSimulator(**common, streaming=True)
    full = full_simulator.run()
    streaming = streaming_simulator.run()

    _assert_simulation_equal(streaming, full)
    assert streaming_simulator._cache_key() == full_simulator._cache_key()
    assert streaming.metadata["calculation_mode"] == "calendar_month_partitions"
    assert streaming.metadata["market_data_partitions_calculated"] == 3
    applied = pd.to_datetime(streaming.rebalances["applied_business_date"])
    assert applied.is_unique
    assert pd.Timestamp("2026-02-02") in set(applied)
    assert pd.Timestamp("2026-03-02") in set(applied)


def test_legacy_market_cap_is_replay_only():
    simulator = IndexSimulator(
        backtest_result=_backtest_result(),
        market_data_provider_name="preloaded",
        start_date="2026-01-27",
        end_date="2026-03-06",
        market_data=_market_data("2026-01-01", "2026-03-06"),
        params=SimulationParams(weight_drift=WeightDrift.MARKET_CAP),
    )

    with pytest.raises(ValueError, match="exact v1 cache replay"):
        simulator.run()


class PartitionRecordingProvider:
    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def load_daily_market_data(
        self,
        instrument_ids,
        start_date,
        end_date,
        **kwargs,
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        type(self).calls.append((start, end))
        return _market_data(start, end).loc[
            lambda frame: frame["instrument_id"].isin(instrument_ids)
        ]


@pytest.fixture
def partition_provider():
    PartitionRecordingProvider.calls = []
    register_provider(
        "partition_recording_provider",
        PartitionRecordingProvider(),
        replace=True,
    )
    yield
    registry.unregister("partition_recording_provider")


def test_streaming_segment_extension_loads_only_new_month_partitions(
    partition_provider,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    store = WorkspaceStore(
        workspace_name="streaming_extension",
        fingerprint="1" * 64,
        index_id="STREAMING_DEMO",
        methodology_name="streaming_test",
        configuration_digest="2" * 64,
        data_revision="market-v1",
    )
    common = {
        "backtest_result": _backtest_result(),
        "market_data_provider_name": "partition_recording_provider",
        "start_date": "2026-01-27",
        "workspace": store,
        "data_revision": "market-v1",
        "segmented_cache": True,
        "streaming": True,
    }

    first = IndexSimulator(end_date="2026-02-27", **common).run()
    first_call_count = len(PartitionRecordingProvider.calls)
    checkpoint_date = first.checkpoint.business_date
    extended = IndexSimulator(end_date="2026-03-06", **common).run()
    extension_calls = PartitionRecordingProvider.calls[first_call_count:]

    assert extension_calls
    assert all(start > checkpoint_date for start, _ in extension_calls)
    assert extension_calls == [
        (pd.Timestamp("2026-02-28"), pd.Timestamp("2026-02-28")),
        (pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-06")),
    ]
    assert extended.metadata["segment_reuse"] == "extended_prefix"

    full = IndexSimulator(
        backtest_result=_backtest_result(),
        market_data_provider_name="preloaded",
        start_date="2026-01-27",
        end_date="2026-03-06",
        market_data=_market_data("2026-01-01", "2026-03-06"),
    ).run()
    _assert_simulation_equal(extended, full)
