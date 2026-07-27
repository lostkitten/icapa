"""Smoke tests for stateful daily index simulation."""

import numpy as np
import pandas as pd
import pytest

from icapa.backtesting import (
    BacktestResult,
    IndexSimulator,
    SimulationParams,
)
from icapa.data_sources import register_provider
from icapa.tools.container import DataContext
from icapa.workspace import WorkspaceStore


class SyntheticMarketProvider:
    """Deterministic daily market data used only by this test module."""

    instrument_ids = tuple(range(1001, 1007))

    def load_daily_market_data(self, instrument_ids, start_date, end_date, **kwargs):
        dates = pd.bdate_range(start=start_date, end=end_date)
        if len(dates) < 2:
            raise ValueError("synthetic requests must span at least two business days")
        records = []
        time = np.arange(len(dates), dtype=float)
        for position, instrument_id in enumerate(instrument_ids, start=1):
            price_returns = (
                0.0001 * position
                + 0.0007 * position * np.sin(time / (2.5 + position / 3.0))
                + 0.0003 * np.cos(time * position / 5.0)
            )
            for business_date, price_return in zip(dates, price_returns):
                records.append(
                    {
                        "instrument_id": instrument_id,
                        "business_date": business_date,
                        "price_return": float(price_return),
                        "gross_dividend": 0.00002 * position,
                        "net_dividend": 0.000015 * position,
                        "market_cap": 1_000_000.0 * position,
                    }
                )
        return pd.DataFrame.from_records(records)


def _context(effective_date, index_weights, benchmark_weights):
    context = DataContext(
        reference_date=pd.Timestamp(effective_date) - pd.Timedelta(days=7),
        effective_date=effective_date,
        index_id="SIMULATION_DEMO",
    )
    context.set_dataframe(
        pd.DataFrame(
            {
                "instrument_id": SyntheticMarketProvider.instrument_ids,
                "name": [
                    f"Instrument {item}"
                    for item in SyntheticMarketProvider.instrument_ids
                ],
                "country": ["US", "US", "CA", "CA", "GB", "GB"],
                "industry": ["Technology", "Industrials"] * 3,
                "benchmark_weight": benchmark_weights,
                "index_weight": index_weights,
            }
        )
    )
    return context


def _backtest_result():
    first_date = pd.Timestamp("2026-06-22")
    second_date = pd.Timestamp("2026-06-25")
    benchmark = np.full(6, 1.0 / 6.0)
    reviews = {
        first_date: _context(
            first_date,
            np.array([0.30, 0.24, 0.18, 0.12, 0.10, 0.06]),
            benchmark,
        ),
        second_date: _context(
            second_date,
            np.array([0.08, 0.12, 0.16, 0.20, 0.20, 0.24]),
            benchmark,
        ),
    }
    rows = []
    for effective_date, context in reviews.items():
        weights = context.cons[["index_weight"]].reset_index()
        weights.insert(0, "effective_date", effective_date)
        rows.append(weights)
    weight_frame = pd.concat(rows, ignore_index=True).set_index(
        ["effective_date", "instrument_id"]
    )
    return BacktestResult(weights=weight_frame, reviews=reviews)


def test_daily_simulation_rebalances_drifts_and_builds_index_levels():
    register_provider(
        "daily_simulation_demo",
        SyntheticMarketProvider(),
        replace=True,
    )
    result = IndexSimulator(
        backtest_result=_backtest_result(),
        market_data_provider_name="daily_simulation_demo",
        start_date="2026-06-22",
        end_date="2026-06-30",
        params=SimulationParams(base_value=1_000.0),
    ).run()

    assert len(result.daily) == len(pd.bdate_range("2026-06-22", "2026-06-30"))
    assert np.isfinite(result.daily.to_numpy(dtype=float)).all()
    assert (result.daily.filter(like="_level") > 0).all().all()
    assert result.rebalances["index_turnover"].iloc[1] > 0
    assert result.rebalances["benchmark_turnover"].iloc[1] > 0

    opening_totals = result.holdings.groupby(level="business_date")[
        ["index_opening_weight", "benchmark_opening_weight"]
    ].sum()
    closing_totals = result.holdings.groupby(level="business_date")[
        ["index_closing_weight", "benchmark_closing_weight"]
    ].sum()
    np.testing.assert_allclose(opening_totals, 1.0, atol=1e-12)
    np.testing.assert_allclose(closing_totals, 1.0, atol=1e-12)


def test_simulation_requires_a_review_active_at_the_start():
    register_provider(
        "daily_simulation_demo",
        SyntheticMarketProvider(),
        replace=True,
    )
    result = _backtest_result()
    later_reviews = {
        date: context
        for date, context in result.reviews.items()
        if date > pd.Timestamp("2026-06-22")
    }
    later_weights = result.weights.loc[
        result.weights.index.get_level_values("effective_date")
        > pd.Timestamp("2026-06-22")
    ]
    with pytest.raises(ValueError, match="before the first available review"):
        IndexSimulator(
            backtest_result=BacktestResult(
                weights=later_weights,
                reviews=later_reviews,
            ),
            market_data_provider_name="daily_simulation_demo",
            start_date="2026-06-22",
            end_date="2026-06-30",
        ).run()


def test_daily_simulation_reuses_workspace_stage_cache(tmp_path, monkeypatch):
    class CountingProvider(SyntheticMarketProvider):
        calls = 0

        def load_daily_market_data(self, *args, **kwargs):
            type(self).calls += 1
            return super().load_daily_market_data(*args, **kwargs)

    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    provider = CountingProvider()
    register_provider("cached_daily_demo", provider, replace=True)
    store = WorkspaceStore(
        workspace_name="simulation_cache_demo",
        fingerprint="a" * 64,
        index_id="SIMULATION_DEMO",
        methodology_name="demo",
        configuration_digest="b" * 64,
        data_revision="market-data-1",
    )
    arguments = {
        "backtest_result": _backtest_result(),
        "market_data_provider_name": "cached_daily_demo",
        "start_date": "2026-06-22",
        "end_date": "2026-06-30",
        "data_revision": "market-data-1",
        "workspace": store,
    }

    first = IndexSimulator(**arguments).run()
    second = IndexSimulator(**arguments).run()

    assert CountingProvider.calls == 1
    assert first.metadata["cache_source"] == "computed"
    assert second.metadata["cache_source"] == "workspace"
    pd.testing.assert_frame_equal(second.daily, first.daily)
