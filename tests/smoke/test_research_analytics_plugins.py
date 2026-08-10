"""Smoke tests for the extensible research analytics profile."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from icapa.analytics import (
    AnalyticsPluginSpec,
    AnalyticsSpec,
    AnalyticsValidationError,
    ReturnSeries,
    run_analytics_plugins,
)
from icapa.backtesting.reviews import BacktestResult
from icapa.portfolio_construction.context import DataContext


def _context(effective_date, weights):
    context = DataContext(
        reference_date=pd.Timestamp(effective_date) - pd.Timedelta(days=10),
        effective_date=effective_date,
        index_id="RESEARCH_INDEX",
    )
    context.set_dataframe(
        pd.DataFrame(
            [
                {
                    "instrument_id": instrument_id,
                    "index_weight": index_weight,
                    "benchmark_weight": benchmark_weight,
                    "country": country,
                    "industry": industry,
                }
                for (
                    instrument_id,
                    index_weight,
                    benchmark_weight,
                    country,
                    industry,
                ) in weights
            ]
        )
    )
    return context


def _backtest():
    first_date = pd.Timestamp("2026-03-23")
    second_date = pd.Timestamp("2026-06-22")
    first = _context(
        first_date,
        (
            ("A", 0.6, 0.5, "US", "Technology"),
            ("B", 0.4, 0.5, "GB", "Industrials"),
        ),
    )
    second = _context(
        second_date,
        (
            ("A", 0.4, 0.5, "US", "Technology"),
            ("C", 0.6, 0.5, "JP", "Health Care"),
        ),
    )
    reviews = {first_date: first, second_date: second}
    rows = []
    for effective_date, context in reviews.items():
        frame = context.cons[["index_weight"]].reset_index()
        frame.insert(0, "effective_date", effective_date)
        rows.append(frame)
    weights = pd.concat(rows, ignore_index=True).set_index(
        ["effective_date", "instrument_id"]
    )
    return BacktestResult(weights=weights, reviews=reviews)


def _simulation():
    dates = pd.date_range("2026-06-22", periods=30, freq="B")
    daily = pd.DataFrame(
        {
            "index_price_return": [0.001] * len(dates),
            "benchmark_price_return": [0.0008] * len(dates),
            "index_gross_total_return": [0.0011] * len(dates),
            "benchmark_gross_total_return": [0.0009] * len(dates),
            "index_net_total_return": [0.00105] * len(dates),
            "benchmark_net_total_return": [0.00085] * len(dates),
        },
        index=pd.DatetimeIndex(dates, name="business_date"),
    )
    return SimpleNamespace(
        daily=daily,
        rebalances=pd.DataFrame(
            {
                "scheduled_effective_date": [
                    pd.Timestamp("2026-03-23"),
                    pd.Timestamp("2026-06-22"),
                ],
                "index_turnover": [float("nan"), 0.3],
            }
        ),
    )


def _calendar_period_spec():
    return AnalyticsSpec(
        profile="calendar_period_test",
        plugins=(AnalyticsPluginSpec("calendar_period_performance"),),
    )


def test_standard_profile_adds_research_tables_without_changing_legacy_result():
    result = run_analytics_plugins(_backtest(), _simulation())

    assert result.legacy_result is not None
    tables = result.tables()
    changes = tables["constituent_change.detail"]
    assert set(changes["status"]) >= {"entrant", "exit", "weight_decrease"}
    stability = tables["constituent_change.membership_stability"].iloc[0]
    assert stability["entrants"] == 1
    assert stability["exits"] == 1
    assert stability["membership_jaccard"] == 1 / 3
    assert not tables["turnover_decomposition.detail"].empty
    assert not tables["rolling_risk.metrics"].empty
    annual = tables["calendar_period_performance.annual"].iloc[0]
    assert annual["observations"] == 30
    assert annual["start_date"] == pd.Timestamp("2026-06-22")
    assert annual["end_date"] == pd.Timestamp("2026-07-31")
    assert result.spec.return_series is ReturnSeries.NET_TOTAL


def test_optional_simulation_plugins_skip_with_structured_warnings():
    result = run_analytics_plugins(_backtest(), spec=AnalyticsSpec.standard_research())

    assert result.legacy_result is not None
    skipped = {item.code for item in result.diagnostics if item.level == "warning"}
    assert "calendar_period_performance_skipped" in skipped
    assert "rolling_risk_skipped" in skipped
    assert "drawdown_episodes_skipped" in skipped


def test_return_series_is_explicit_in_research_profile():
    spec = AnalyticsSpec(
        profile="price_only",
        plugins=AnalyticsSpec.legacy_parity().plugins,
        return_series=ReturnSeries.PRICE,
    )

    result = run_analytics_plugins(_backtest(), _simulation(), spec=spec)

    selected = [
        item
        for item in result.legacy_result.diagnostics
        if item.code == "return_columns_selected"
    ]
    assert len(selected) == 1
    assert "index_price_return" in selected[0].message


def test_calendar_period_omits_disjoint_return_dates():
    dates = pd.to_datetime(
        ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
    )
    simulation = SimpleNamespace(
        daily=pd.DataFrame(
            {
                "index_net_total_return": [0.01, None, 0.02, None],
                "benchmark_net_total_return": [None, 0.01, None, 0.02],
            },
            index=pd.DatetimeIndex(dates, name="business_date"),
        )
    )

    result = run_analytics_plugins(
        _backtest(),
        simulation,
        spec=_calendar_period_spec(),
    )

    annual = result.tables()["calendar_period_performance.annual"]
    assert annual.empty
    assert list(annual.columns) == [
        "period",
        "start_date",
        "end_date",
        "observations",
        "index_return",
        "benchmark_return",
        "active_return",
    ]
    assert annual.dtypes.astype(str).to_dict() == {
        "period": "object",
        "start_date": "datetime64[ns]",
        "end_date": "datetime64[ns]",
        "observations": "int64",
        "index_return": "float64",
        "benchmark_return": "float64",
        "active_return": "float64",
    }
    assert {item.code for item in result.diagnostics} == {
        "calendar_period_incomplete_returns_dropped",
        "calendar_period_returns_unavailable",
    }


def test_calendar_period_compounds_only_complete_return_pairs():
    dates = pd.to_datetime(
        [
            "2025-12-30",
            "2025-12-31",
            "2026-01-02",
            "2026-01-05",
            "2026-01-06",
        ]
    )
    simulation = SimpleNamespace(
        daily=pd.DataFrame(
            {
                "index_net_total_return": [0.01, None, 0.02, 0.03, None],
                "benchmark_net_total_return": [None, 0.01, 0.01, 0.02, 0.04],
            },
            index=pd.DatetimeIndex(dates, name="business_date"),
        )
    )

    daily_before = simulation.daily.copy(deep=True)
    result = run_analytics_plugins(
        _backtest(),
        simulation,
        spec=_calendar_period_spec(),
    )

    annual = result.tables()["calendar_period_performance.annual"]
    assert annual["period"].tolist() == ["2026"]
    row = annual.iloc[0]
    assert row["start_date"] == pd.Timestamp("2026-01-02")
    assert row["end_date"] == pd.Timestamp("2026-01-05")
    assert row["observations"] == 2
    assert abs(row["index_return"] - 0.0506) < 1e-12
    assert abs(row["benchmark_return"] - 0.0302) < 1e-12
    assert abs(row["active_return"] - 0.0204) < 1e-12
    assert [item.code for item in result.diagnostics] == [
        "calendar_period_incomplete_returns_dropped"
    ]
    warning = result.diagnostics[0]
    assert warning.level == "warning"
    assert "3 daily rows" in warning.message
    pd.testing.assert_frame_equal(simulation.daily, daily_before)


@pytest.mark.parametrize(
    ("return_series", "index_column", "benchmark_column"),
    (
        (
            ReturnSeries.GROSS_TOTAL,
            "index_gross_total_return",
            "benchmark_gross_total_return",
        ),
        (
            ReturnSeries.PRICE,
            "index_price_return",
            "benchmark_price_return",
        ),
    ),
)
def test_calendar_period_uses_explicit_return_series(
    return_series,
    index_column,
    benchmark_column,
):
    simulation = _simulation()
    spec = AnalyticsSpec(
        profile="calendar_period_explicit_return_series",
        plugins=(AnalyticsPluginSpec("calendar_period_performance"),),
        return_series=return_series,
    )

    result = run_analytics_plugins(_backtest(), simulation, spec=spec)

    annual = result.tables()["calendar_period_performance.annual"].iloc[0]
    expected_index = float(
        (1.0 + simulation.daily[index_column]).prod() - 1.0
    )
    expected_benchmark = float(
        (1.0 + simulation.daily[benchmark_column]).prod() - 1.0
    )
    assert annual["index_return"] == pytest.approx(expected_index)
    assert annual["benchmark_return"] == pytest.approx(expected_benchmark)


@pytest.mark.parametrize(
    ("invalid_value", "message"),
    (
        ("invalid", "contains non-numeric values"),
        (np.inf, "contains non-finite values"),
        (-1.1, "daily returns cannot be less than -100%"),
    ),
)
def test_calendar_period_rejects_invalid_returns(invalid_value, message):
    simulation = _simulation()
    if isinstance(invalid_value, str):
        simulation.daily["index_net_total_return"] = simulation.daily[
            "index_net_total_return"
        ].astype(object)
    simulation.daily.loc[
        simulation.daily.index[0], "index_net_total_return"
    ] = invalid_value

    with pytest.raises(AnalyticsValidationError, match=message):
        run_analytics_plugins(
            _backtest(),
            simulation,
            spec=_calendar_period_spec(),
        )


def test_calendar_period_rejects_invalid_business_dates():
    simulation = SimpleNamespace(
        daily=pd.DataFrame(
            {
                "index_net_total_return": [0.01],
                "benchmark_net_total_return": [0.01],
            },
            index=pd.DatetimeIndex([pd.NaT], name="business_date"),
        )
    )

    with pytest.raises(
        AnalyticsValidationError,
        match="invalid business dates",
    ):
        run_analytics_plugins(
            _backtest(),
            simulation,
            spec=_calendar_period_spec(),
        )
