"""Focused contracts for optional index-research analytics."""

import pandas as pd
import pytest

from icapa.analytics import (
    AnalyticsPluginSpec,
    AnalyticsSpec,
    BrinsonInput,
    ResearchAnalyticsInputs,
    run_analytics_plugins,
)
from icapa.backtesting.reviews import BacktestResult
from icapa.portfolio_construction.context import DataContext


def _review(effective_date, rows, diagnostics):
    context = DataContext(
        reference_date=pd.Timestamp(effective_date) - pd.Timedelta(days=10),
        effective_date=effective_date,
        index_id="RESEARCH_INDEX",
        diagnostics=diagnostics,
    )
    context.set_dataframe(pd.DataFrame(rows))
    return context


def _backtest():
    first_date = pd.Timestamp("2026-03-23")
    second_date = pd.Timestamp("2026-06-22")
    first = _review(
        first_date,
        [
            {
                "instrument_id": "A",
                "index_weight": 0.6,
                "benchmark_weight": 0.5,
                "country": "US",
                "industry": "Technology",
                "selection_reason": ["highest score"],
                "exclusion_reason": [],
                "quality_signal": 1.0,
                "value_zscore": 0.5,
                "average_daily_value_traded": 100.0,
                "capacity_weight_limit": 0.5,
                "signal_observation_date": "2026-03-10",
            },
            {
                "instrument_id": "B",
                "index_weight": 0.4,
                "benchmark_weight": 0.3,
                "country": "GB",
                "industry": "Industrials",
                "selection_reason": ["eligible"],
                "exclusion_reason": [],
                "quality_signal": 0.0,
                "value_zscore": 0.2,
                "average_daily_value_traded": 80.0,
                "capacity_weight_limit": 0.5,
                "signal_observation_date": "2026-03-11",
            },
            {
                "instrument_id": "C",
                "index_weight": 0.0,
                "benchmark_weight": 0.2,
                "country": "JP",
                "industry": "Health Care",
                "selection_reason": [],
                "exclusion_reason": ["below selection threshold"],
                "quality_signal": -1.0,
                "value_zscore": -0.4,
                "average_daily_value_traded": 40.0,
                "capacity_weight_limit": 0.4,
                "signal_observation_date": "2026-03-09",
            },
        ],
        {
            "target_diagnostics": [
                {
                    "name": "quality_signal",
                    "requested": 0.55,
                    "achieved": 0.56,
                    "tolerance": 0.02,
                }
            ],
            "optimizer": {
                "constraint_diagnostics": [
                    {
                        "name": "maximum_weight",
                        "value": 0.6,
                        "lower": 0.0,
                        "upper": 0.6,
                    }
                ]
            },
        },
    )
    second = _review(
        second_date,
        [
            {
                "instrument_id": "A",
                "index_weight": 0.4,
                "benchmark_weight": 0.5,
                "country": "US",
                "industry": "Technology",
                "selection_reason": ["eligible"],
                "exclusion_reason": [],
                "quality_signal": 0.8,
                "value_zscore": 0.4,
                "average_daily_value_traded": 105.0,
                "capacity_weight_limit": 0.5,
                "signal_observation_date": "2026-06-10",
            },
            {
                "instrument_id": "B",
                "index_weight": 0.0,
                "benchmark_weight": 0.3,
                "country": "GB",
                "industry": "Industrials",
                "selection_reason": [],
                "exclusion_reason": ["below selection threshold"],
                "quality_signal": -0.2,
                "value_zscore": 0.1,
                "average_daily_value_traded": 75.0,
                "capacity_weight_limit": 0.5,
                "signal_observation_date": "2026-06-11",
            },
            {
                "instrument_id": "C",
                "index_weight": 0.6,
                "benchmark_weight": 0.2,
                "country": "JP",
                "industry": "Health Care",
                "selection_reason": ["highest score"],
                "exclusion_reason": [],
                "quality_signal": 1.2,
                "value_zscore": 0.7,
                "average_daily_value_traded": 50.0,
                "capacity_weight_limit": 0.55,
                "signal_observation_date": "2026-06-15",
            },
        ],
        {
            "target_attainment": [
                {
                    "field": "quality_signal",
                    "target": 1.0,
                    "value": 1.04,
                    "tolerance": 0.05,
                }
            ],
            "constraints": [
                {
                    "name": "maximum_weight",
                    "value": 0.6,
                    "lower": 0.0,
                    "upper": 0.6,
                }
            ],
        },
    )
    reviews = {first_date: first, second_date: second}
    weights = pd.concat(
        [
            context.cons[["index_weight"]]
            .reset_index()
            .assign(effective_date=effective_date)
            for effective_date, context in reviews.items()
        ],
        ignore_index=True,
    ).set_index(["effective_date", "instrument_id"])
    return BacktestResult(weights=weights, reviews=reviews)


def _attribution_input():
    return BrinsonInput(
        pd.DataFrame(
            [
                ("P1", "A", "Technology", 0.6, 0.5, 0.01),
                ("P1", "B", "Industrials", 0.4, 0.3, 0.02),
                ("P1", "C", "Health Care", 0.0, 0.2, 0.03),
                ("P2", "A", "Technology", 0.4, 0.5, 0.00),
                ("P2", "B", "Industrials", 0.0, 0.3, 0.01),
                ("P2", "C", "Health Care", 0.6, 0.2, 0.02),
            ],
            columns=[
                "period",
                "instrument_id",
                "industry",
                "index_weight",
                "benchmark_weight",
                "asset_return",
            ],
        ),
        classification_column="industry",
    )


def test_standard_profile_materialises_available_research_diagnostics():
    inputs = ResearchAnalyticsInputs(
        attribution_input=_attribution_input()
    )

    result = run_analytics_plugins(_backtest(), inputs=inputs)
    tables = result.tables()

    assert result.legacy_result is not None
    reasons = tables["selection_reasons.detail"]
    assert {
        "highest score",
        "below selection threshold",
    }.issubset(set(reasons["reason"]))

    contributors = tables["weight_change_contributors.detail"]
    shares = contributors.groupby("effective_date")[
        "share_of_one_way_turnover"
    ].sum()
    assert shares.iloc[0] == pytest.approx(1.0)

    targets = tables["target_attainment.detail"]
    assert targets["within_bounds"].all()
    constraints = tables["constraint_diagnostics.detail"]
    assert constraints["binding"].all()
    assert not constraints["violated"].any()

    exposures = tables["factor_signal_exposure.exposures"]
    assert set(exposures["field"]) == {
        "quality_signal",
        "value_zscore",
    }
    assert exposures["index_weight_coverage"].eq(1.0).all()

    liquidity = tables["liquidity_capacity_coverage.coverage"]
    assert liquidity["available_count"].eq(3).all()
    capacity = tables["liquidity_capacity_coverage.capacity"]
    assert capacity["capacity_breach"].any()

    freshness = tables["data_freshness.sources"]
    assert set(freshness["source"]) == {"signal_observation_date"}
    assert freshness["future_observation_count"].sum() == 1
    assert "future_dated_research_input" in {
        item.code for item in result.diagnostics
    }

    attribution = tables["multi_period_attribution.totals"]
    assert set(attribution["period"]) == {"P1", "P2"}
    linked = tables["multi_period_attribution.linked_totals"].set_index(
        "component"
    )
    assert linked.loc[
        "total_attribution",
        "linked_contribution",
    ] == pytest.approx(
        linked.loc[
            "multi_period_active_return",
            "linked_contribution",
        ]
    )


def test_explicit_reason_inputs_are_copied_and_do_not_require_review_columns():
    backtest = _backtest()
    supplied = pd.DataFrame(
        [
            {
                "effective_date": "2026-06-22",
                "instrument_id": "C",
                "decision": "selected",
                "reason": "explicit research note",
            }
        ]
    )
    inputs = ResearchAnalyticsInputs(selection_reasons=supplied)
    supplied.loc[0, "reason"] = "mutated"
    spec = AnalyticsSpec(
        profile="selection_only",
        plugins=(AnalyticsPluginSpec("selection_reasons"),),
    )

    result = run_analytics_plugins(backtest, spec=spec, inputs=inputs)

    detail = result.tables()["selection_reasons.detail"]
    assert detail.loc[0, "reason"] == "explicit research note"


def test_missing_optional_research_inputs_produce_warnings_not_fake_tables():
    backtest = _backtest()
    for context in backtest.reviews.values():
        context.cons.drop(
            columns=[
                "selection_reason",
                "exclusion_reason",
                "quality_signal",
                "value_zscore",
                "average_daily_value_traded",
                "capacity_weight_limit",
                "signal_observation_date",
            ],
            inplace=True,
        )
        context.diagnostics.clear()

    result = run_analytics_plugins(backtest)

    warning_codes = {
        item.code for item in result.diagnostics if item.level == "warning"
    }
    assert {
        "selection_reasons_skipped",
        "target_attainment_skipped",
        "constraint_diagnostics_skipped",
        "factor_signal_exposure_skipped",
        "liquidity_capacity_coverage_skipped",
        "data_freshness_skipped",
        "multi_period_attribution_skipped",
    }.issubset(warning_codes)
    assert result.plugin_results["target_attainment"].tables == {}


def test_factor_exposures_accept_canonical_review_date_columns():
    backtest = _backtest()
    for effective_date, context in backtest.reviews.items():
        context.cons["effective_date"] = effective_date
        context.cons["reference_date"] = context.reference_date
    spec = AnalyticsSpec(
        profile="factor_exposure_only",
        plugins=(AnalyticsPluginSpec("factor_signal_exposure"),),
    )

    result = run_analytics_plugins(backtest, spec=spec)

    exposures = result.tables()["factor_signal_exposure.exposures"]
    assert set(exposures["effective_date"]) == set(backtest.reviews)
    for effective_date, context in backtest.reviews.items():
        assert context.cons["effective_date"].eq(effective_date).all()
        assert context.cons["reference_date"].eq(
            context.reference_date
        ).all()
