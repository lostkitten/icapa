"""Pseudodata coverage for domain-organized research analytics."""

import numpy as np
import pandas as pd
import pytest

from icapa.analytics import AnalyticsFeatureEngine, default_analytics_registry
from icapa.analytics.attribution import calculate_factor_attribution
from icapa.analytics.constituents import (
    explain_weight_change,
    explain_weight_construction,
)
from icapa.analytics.events import run_event_study
from icapa.analytics.exposures import calculate_weighted_exposures
from icapa.analytics.performance import calculate_performance_metrics
from icapa.analytics.quality import (
    calculate_data_coverage,
    calculate_data_freshness,
)
from icapa.analytics.reconciliation import compare_data_stages
from icapa.analytics.regimes import analyze_regimes
from icapa.analytics.risk import (
    LiquidityCapacitySpec,
    calculate_concentration,
    calculate_liquidity_capacity,
    calculate_risk_contributions,
)


def _business_dates() -> pd.DatetimeIndex:
    return pd.date_range("2026-01-02", periods=30, freq="B")


def test_performance_events_factor_attribution_and_regimes_use_pseudodata():
    dates = _business_dates()
    factor_returns = pd.DataFrame(
        {
            "quality": np.linspace(-0.01, 0.012, len(dates)),
            "value": np.sin(np.arange(len(dates))) / 100.0,
        },
        index=dates,
    )
    benchmark = pd.Series(
        np.linspace(-0.001, 0.0015, len(dates)),
        index=dates,
        name="benchmark",
    )
    index = (
        benchmark
        + 0.4 * factor_returns["quality"]
        - 0.2 * factor_returns["value"]
        + 0.0001
    )

    performance = calculate_performance_metrics(index, benchmark)
    assert performance.summary["tracking_error"] > 0
    assert performance.daily.index.name == "business_date"

    event_study = run_event_study(
        index,
        [dates[10], dates[20]],
        benchmark_returns=benchmark,
    )
    assert set(event_study.summary["relative_day"]) == set(range(-5, 6))
    assert event_study.summary["event_count"].eq(2).all()

    attribution = calculate_factor_attribution(
        index,
        factor_returns,
        benchmark_returns=benchmark,
    )
    assert attribution.coefficients["quality"] == pytest.approx(0.4)
    assert attribution.coefficients["value"] == pytest.approx(-0.2)
    assert attribution.coefficients["intercept"] == pytest.approx(0.0001)
    assert np.allclose(
        attribution.daily["attributed_return"],
        attribution.daily["reconciled_return"],
        atol=1e-12,
    )

    regimes = pd.Series(
        ["expansion"] * 15 + ["contraction"] * 15,
        index=dates,
    )
    regime_result = analyze_regimes(
        pd.DataFrame({"index": index, "benchmark": benchmark}),
        regimes,
        include_distribution_tests=False,
    )
    assert set(regime_result.summary["regime"]) == {
        "expansion",
        "contraction",
    }
    assert (
        regime_result.transition_counts.loc["expansion", "contraction"] == 1
    )
    assert regime_result.transition_probabilities.sum(axis=1).eq(1.0).all()


def test_risk_liquidity_exposure_quality_and_concentration_reconcile():
    instruments = pd.Index(["A", "B", "C"], name="instrument_id")
    weights = pd.Series([0.50, 0.30, 0.20], index=instruments)
    benchmark = pd.Series([0.40, 0.35, 0.25], index=instruments)
    covariance = pd.DataFrame(
        [
            [0.040, 0.010, 0.000],
            [0.010, 0.090, 0.010],
            [0.000, 0.010, 0.160],
        ],
        index=instruments,
        columns=instruments,
    )

    risk = calculate_risk_contributions(
        weights,
        covariance,
        benchmark_weights=benchmark,
    )
    assert risk.portfolio["risk_contribution"].sum() == pytest.approx(
        risk.summary["portfolio_volatility"]
    )
    assert risk.active["tracking_error_contribution"].sum() == pytest.approx(
        risk.summary["tracking_error"]
    )

    concentration = calculate_concentration(weights)
    assert concentration["hhi"] == pytest.approx(0.38)
    assert concentration["top_1_weight"] == pytest.approx(0.5)

    liquidity = calculate_liquidity_capacity(
        weights,
        pd.Series([5_000_000.0, 2_000_000.0, 250_000.0], index=instruments),
        spec=LiquidityCapacitySpec(
            assets_under_management=10_000_000.0,
            participation_rate=0.2,
            trading_days=2,
        ),
    )
    assert liquidity.detail.loc["C", "capacity_breach"]
    assert liquidity.summary["capacity_breach_count"] >= 1

    frame = pd.DataFrame(
        {
            "index_weight": weights,
            "benchmark_weight": benchmark,
            "country": ["US", "GB", "JP"],
            "quality_signal": [1.0, 0.2, -0.5],
            "average_daily_value_traded": [5_000_000.0, None, 250_000.0],
        },
        index=instruments,
    )
    exposures = calculate_weighted_exposures(
        frame,
        ["quality_signal"],
    )
    assert exposures.loc[0, "active_exposure"] == pytest.approx(
        exposures.loc[0, "portfolio_exposure"]
        - exposures.loc[0, "benchmark_exposure"]
    )
    coverage = calculate_data_coverage(
        frame,
        ["average_daily_value_traded"],
        weight_column="index_weight",
    )
    assert coverage.loc[0, "available_count"] == 2
    assert coverage.loc[0, "weight_coverage"] == pytest.approx(0.7)
    freshness = calculate_data_freshness(
        pd.Series(["2026-03-01", "2026-03-10", "2026-03-20"]),
        "2026-03-15",
    )
    assert freshness["future_observation_count"] == 1


def test_weight_explanation_waterfall_and_registry_are_discoverable():
    instruments = pd.Index(["A", "B", "C"], name="instrument_id")
    construction = pd.DataFrame(
        {
            "benchmark_weight": [0.50, 0.30, 0.20],
            "quality_tilt": [1.20, 0.80, 1.00],
            "size_tilt": [0.90, 1.10, 1.00],
        },
        index=instruments,
    )
    first = (
        construction["benchmark_weight"] * construction["quality_tilt"]
    )
    first /= first.sum()
    target = first * construction["size_tilt"]
    construction["index_weight"] = target / target.sum()

    explanation = explain_weight_construction(
        construction,
        tilt_columns=("quality_tilt", "size_tilt"),
    )
    assert explanation.reconciled
    assert np.allclose(
        explanation.summary["resulting_weight_sum"],
        1.0,
        atol=1e-12,
    )
    assert np.allclose(
        explanation.final_weights,
        construction["index_weight"],
        atol=1e-12,
    )

    changes = explain_weight_change(
        construction["benchmark_weight"],
        construction["index_weight"],
    )
    assert changes["one_way_turnover_contribution"].sum() == pytest.approx(
        0.5
        * (
            construction["index_weight"]
            - construction["benchmark_weight"]
        ).abs().sum()
    )

    stages = {
        "loaded": pd.DataFrame(
            {"instrument_id": ["A", "B"], "signal": [0.2, 0.4]}
        ),
        "processed": pd.DataFrame(
            {"instrument_id": ["A", "B"], "signal": [0.2, 0.5]}
        ),
        "validated": pd.DataFrame(
            {"instrument_id": ["A", "B"], "signal": [0.2, 0.5]}
        ),
    }
    waterfall = compare_data_stages(
        stages,
        key_columns=("instrument_id",),
    )
    assert not waterfall.reconciled
    assert waterfall.detail["status"].eq("changed").sum() == 1
    assert waterfall.summary["changed"].sum() == 1

    registry = default_analytics_registry()
    identifiers = {feature.feature_id for feature in registry.list()}
    assert {
        "events.study",
        "attribution.factor",
        "regimes.analysis",
        "risk.contributions",
        "constituents.weight_explanation",
        "reconciliation.data_waterfall",
    }.issubset(identifiers)
    engine = AnalyticsFeatureEngine(registry)
    registry_concentration = engine.run(
        "risk.concentration",
        construction["benchmark_weight"],
    )
    assert registry_concentration["hhi"] == pytest.approx(0.38)
