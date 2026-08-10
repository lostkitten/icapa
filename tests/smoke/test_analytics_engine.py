"""Focused smoke tests for provider-neutral, side-effect-free analytics."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from icapa.analytics import (
    AnalyticsValidationError,
    BrinsonInput,
    analyze_backtest,
)
from icapa.backtesting.reviews import BacktestResult
from icapa.portfolio_construction.context import DataContext


FIRST_REVIEW = pd.Timestamp("2026-03-23")
SECOND_REVIEW = pd.Timestamp("2026-06-22")


def _context(
    effective_date: pd.Timestamp,
    rows: list[dict[str, object]],
) -> DataContext:
    context = DataContext(
        reference_date=effective_date - pd.Timedelta(days=14),
        effective_date=effective_date,
        index_id="GENERIC_DEMO",
    )
    context.set_dataframe(pd.DataFrame(rows))
    return context


def _backtest_result() -> BacktestResult:
    first = _context(
        FIRST_REVIEW,
        [
            {
                "instrument_id": "A",
                "index_weight": 0.50,
                "benchmark_weight": 0.40,
                "country": "US",
                "industry": "Technology",
            },
            {
                "instrument_id": "B",
                "index_weight": 0.30,
                "benchmark_weight": 0.35,
                "country": "US",
                "industry": "Health Care",
            },
            {
                "instrument_id": "C",
                "index_weight": 0.20,
                "benchmark_weight": 0.25,
                "country": "GB",
                "industry": "Technology",
            },
        ],
    )
    second = _context(
        SECOND_REVIEW,
        [
            {
                "instrument_id": "A",
                "index_weight": 0.40,
                "benchmark_weight": 0.35,
                "country": "US",
                "industry": "Technology",
            },
            {
                "instrument_id": "B",
                "index_weight": 0.20,
                "benchmark_weight": 0.25,
                "country": "US",
                "industry": "Health Care",
            },
            {
                "instrument_id": "D",
                "index_weight": 0.40,
                "benchmark_weight": 0.40,
                "country": "JP",
                "industry": "Industrials",
            },
        ],
    )
    reviews = {FIRST_REVIEW: first, SECOND_REVIEW: second}
    rows = []
    for effective_date, context in reviews.items():
        weights = context.cons[["index_weight"]].reset_index()
        weights.insert(0, "effective_date", effective_date)
        rows.append(weights)
    combined = pd.concat(rows, ignore_index=True).set_index(
        ["effective_date", "instrument_id"]
    )
    return BacktestResult(weights=combined, reviews=reviews)


def _simulation_result() -> SimpleNamespace:
    business_dates = pd.date_range("2026-06-22", periods=5, freq="B")
    daily = pd.DataFrame(
        {
            "index_price_return": [0.009, -0.018, 0.014, 0.000, 0.004],
            "benchmark_price_return": [0.007, -0.009, 0.011, 0.001, 0.003],
            "index_gross_total_return": [0.011, -0.017, 0.016, 0.001, 0.006],
            "benchmark_gross_total_return": [0.009, -0.008, 0.013, 0.002, 0.005],
            "index_net_total_return": [0.010, -0.020, 0.015, 0.000, 0.005],
            "benchmark_net_total_return": [0.008, -0.010, 0.012, 0.001, 0.004],
        },
        index=pd.DatetimeIndex(business_dates, name="business_date"),
    )
    rebalances = pd.DataFrame(
        {
            "scheduled_effective_date": [FIRST_REVIEW, SECOND_REVIEW],
            "applied_business_date": [FIRST_REVIEW, SECOND_REVIEW],
            "index_turnover": [np.nan, 0.37],
            "benchmark_turnover": [np.nan, 0.15],
            "review_source": ["computed", "computed"],
        }
    )
    return SimpleNamespace(
        daily=daily,
        holdings=pd.DataFrame(),
        rebalances=rebalances,
        asset_returns=pd.DataFrame(),
        metadata={"calculation": "synthetic"},
    )


def _brinson_input() -> BrinsonInput:
    return BrinsonInput(
        pd.DataFrame(
            {
                "period": ["Q2", "Q2", "Q2", "Q2"],
                "instrument_id": ["A", "B", "C", "D"],
                "industry": [
                    "Technology",
                    "Technology",
                    "Health Care",
                    "Health Care",
                ],
                "index_weight": [0.4, 0.1, 0.2, 0.3],
                "benchmark_weight": [0.3, 0.2, 0.3, 0.2],
                "asset_return": [0.10, 0.00, 0.02, 0.04],
            }
        )
    )


def test_review_metrics_exposures_and_weight_change_are_generic():
    result = analyze_backtest(_backtest_result())

    assert result.review_validation["is_valid"].all()
    first = result.review_metrics.loc[FIRST_REVIEW]
    assert first["constituent_count"] == 3
    assert first["max_weight"] == pytest.approx(0.5)
    assert first["top_10_weight"] == pytest.approx(1.0)
    assert first["hhi"] == pytest.approx(0.38)
    assert first["effective_n"] == pytest.approx(1.0 / 0.38)
    assert first["active_share"] == pytest.approx(0.10)

    us = result.country_exposures.loc[(FIRST_REVIEW, "US")]
    assert us["portfolio_weight"] == pytest.approx(0.80)
    assert us["benchmark_weight"] == pytest.approx(0.75)
    assert us["active_weight"] == pytest.approx(0.05)
    technology = result.industry_exposures.loc[
        (FIRST_REVIEW, "Technology")
    ]
    assert technology["active_weight"] == pytest.approx(0.05)

    change = result.target_review_weight_change.loc[SECOND_REVIEW]
    assert change["gross_target_weight_change"] == pytest.approx(0.80)
    assert change["one_way_target_weight_change"] == pytest.approx(0.40)


def test_daily_performance_and_formal_turnover_use_simulation_contract():
    backtest = _backtest_result()
    simulation = _simulation_result()
    daily_before = simulation.daily.copy(deep=True)
    rebalances_before = simulation.rebalances.copy(deep=True)

    result = analyze_backtest(backtest, simulation)

    assert result.performance["observations"] == 5
    assert result.performance["tracking_error"] > 0
    assert np.isfinite(result.performance["information_ratio"])
    assert result.performance["maximum_drawdown"] < 0
    assert result.drawdowns.index.name == "business_date"
    assert result.drawdowns["index_drawdown"].le(0).all()
    assert "formal_one_way_turnover" in result.formal_turnover
    assert result.formal_turnover.iloc[1]["formal_one_way_turnover"] == pytest.approx(
        0.37
    )
    assert result.formal_turnover.iloc[1]["review_source"] == "computed"
    assert (
        result.target_review_weight_change.loc[
            SECOND_REVIEW, "one_way_target_weight_change"
        ]
        != result.formal_turnover.iloc[1]["formal_one_way_turnover"]
    )
    selected = [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code == "return_columns_selected"
    ]
    assert len(selected) == 1
    assert "index_net_total_return" in selected[0].message
    pd.testing.assert_frame_equal(simulation.daily, daily_before)
    pd.testing.assert_frame_equal(simulation.rebalances, rebalances_before)


def test_automatic_return_columns_support_gross_total_only():
    daily = _simulation_result().daily[
        ["index_gross_total_return", "benchmark_gross_total_return"]
    ]

    result = analyze_backtest(_backtest_result(), daily_returns=daily)

    expected_total_return = (
        (1.0 + daily["index_gross_total_return"]).prod() - 1.0
    )
    assert result.performance["observations"] == len(daily)
    assert result.performance["total_return"] == pytest.approx(
        expected_total_return
    )
    selected = next(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code == "return_columns_selected"
    )
    assert "index_gross_total_return" in selected.message


def test_automatic_return_columns_support_price_only():
    daily = _simulation_result().daily[
        ["index_price_return", "benchmark_price_return"]
    ]

    result = analyze_backtest(_backtest_result(), daily_returns=daily)

    expected_total_return = (1.0 + daily["index_price_return"]).prod() - 1.0
    assert result.performance["observations"] == len(daily)
    assert result.performance["total_return"] == pytest.approx(
        expected_total_return
    )
    selected = next(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code == "return_columns_selected"
    )
    assert "index_price_return" in selected.message


def test_automatic_return_column_priority_follows_documented_order():
    daily = _simulation_result().daily.assign(
        index_return=0.25,
        benchmark_return=-0.25,
    )
    pairs = (
        ("index_net_total_return", "benchmark_net_total_return"),
        ("index_gross_total_return", "benchmark_gross_total_return"),
        ("index_price_return", "benchmark_price_return"),
        ("index_return", "benchmark_return"),
    )

    for offset, expected_pair in enumerate(pairs):
        available_columns = [
            column for pair in pairs[offset:] for column in pair
        ]
        result = analyze_backtest(
            _backtest_result(),
            daily_returns=daily[available_columns],
        )
        selected = next(
            diagnostic
            for diagnostic in result.diagnostics
            if diagnostic.code == "return_columns_selected"
        )
        assert expected_pair[0] in selected.message
        assert expected_pair[1] in selected.message


def test_automatic_return_columns_skip_incomplete_higher_priority_pair():
    daily = _simulation_result().daily[
        [
            "index_net_total_return",
            "index_gross_total_return",
            "benchmark_gross_total_return",
        ]
    ]

    result = analyze_backtest(_backtest_result(), daily_returns=daily)

    selected = next(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code == "return_columns_selected"
    )
    assert "index_gross_total_return" in selected.message
    assert "benchmark_gross_total_return" in selected.message


def test_automatic_return_columns_do_not_fallback_from_empty_selected_pair():
    daily = _simulation_result().daily[
        [
            "index_net_total_return",
            "benchmark_net_total_return",
            "index_gross_total_return",
            "benchmark_gross_total_return",
        ]
    ].copy()
    daily[["index_net_total_return", "benchmark_net_total_return"]] = np.nan

    result = analyze_backtest(_backtest_result(), daily_returns=daily)

    assert result.performance.empty
    assert "paired_returns_unavailable" in {
        diagnostic.code for diagnostic in result.diagnostics
    }


def test_explicit_return_columns_override_priority_and_remain_strict():
    daily = _simulation_result().daily
    selected_columns = (
        "index_price_return",
        "benchmark_price_return",
    )

    result = analyze_backtest(
        _backtest_result(),
        daily_returns=daily,
        return_columns=selected_columns,
    )

    expected_total_return = (1.0 + daily["index_price_return"]).prod() - 1.0
    assert result.performance["total_return"] == pytest.approx(
        expected_total_return
    )
    selected = next(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code == "return_columns_selected"
    )
    assert "index_price_return" in selected.message

    with pytest.raises(
        AnalyticsValidationError,
        match="missing selected return columns",
    ):
        analyze_backtest(
            _backtest_result(),
            daily_returns=daily,
            return_columns=("missing_index_return", "missing_benchmark_return"),
        )


def test_explicit_brinson_attribution_reconciles_to_active_return():
    result = analyze_backtest(
        _backtest_result(),
        brinson_input=_brinson_input(),
    )

    assert result.brinson is not None
    totals = result.brinson.totals.loc["Q2"]
    assert totals["portfolio_return"] == pytest.approx(0.056)
    assert totals["benchmark_return"] == pytest.approx(0.044)
    assert totals["active_return"] == pytest.approx(0.012)
    assert totals["total_attribution"] == pytest.approx(0.012)


def test_invalid_review_weights_fail_loudly():
    context = _context(
        FIRST_REVIEW,
        [
            {
                "instrument_id": "A",
                "index_weight": 0.60,
                "benchmark_weight": 0.50,
            },
            {
                "instrument_id": "B",
                "index_weight": 0.50,
                "benchmark_weight": 0.50,
            },
        ],
    )
    weights = context.cons[["index_weight"]].reset_index()
    weights.insert(0, "effective_date", FIRST_REVIEW)
    backtest = BacktestResult(
        weights=weights.set_index(["effective_date", "instrument_id"]),
        reviews={FIRST_REVIEW: context},
    )

    with pytest.raises(
        AnalyticsValidationError, match="index_weight sums to"
    ):
        analyze_backtest(backtest)


def test_missing_optional_inputs_are_reported_without_side_effects():
    backtest = _backtest_result()
    weights_before = backtest.weights.copy(deep=True)
    contexts_before = {
        date: context.cons.copy(deep=True)
        for date, context in backtest.reviews.items()
    }

    result = analyze_backtest(backtest)

    assert result.performance.empty
    assert result.drawdowns.empty
    assert result.formal_turnover.empty
    codes = {diagnostic.code for diagnostic in result.diagnostics}
    assert "daily_returns_unavailable" in codes
    assert "formal_turnover_unavailable" in codes
    pd.testing.assert_frame_equal(backtest.weights, weights_before)
    for date, context in backtest.reviews.items():
        pd.testing.assert_frame_equal(context.cons, contexts_before[date])

    exported = result.tables()
    exported["review_metrics"].iloc[0, 0] = -1
    assert result.review_metrics.iloc[0, 0] >= 0
