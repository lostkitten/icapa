"""Smoke tests for baseline-to-candidate index research comparisons."""

from dataclasses import replace

import pandas as pd
import pytest

from icapa.analytics import (
    ComparisonInput,
    ComparisonSpec,
    CompatibilityPolicy,
    compare_research_results,
    run_analytics_plugins,
)
from icapa.backtesting.reviews import BacktestResult
from icapa.portfolio_construction.context import DataContext
from icapa.workspace import RunManifest


def _result(second_weights):
    effective_dates = (
        pd.Timestamp("2026-03-23"),
        pd.Timestamp("2026-06-22"),
    )
    review_rows = (
        (("A", 0.6), ("B", 0.4)),
        second_weights,
    )
    reviews = {}
    rows = []
    for effective_date, weights in zip(effective_dates, review_rows):
        context = DataContext(
            reference_date=effective_date - pd.Timedelta(days=10),
            effective_date=effective_date,
            index_id="RESEARCH_INDEX",
        )
        context.set_dataframe(
            pd.DataFrame(
                [
                    {
                        "instrument_id": instrument_id,
                        "index_weight": weight,
                        "benchmark_weight": 0.5,
                        "country": "US" if instrument_id == "A" else "GB",
                        "industry": "Technology",
                    }
                    for instrument_id, weight in weights
                ]
            )
        )
        reviews[effective_date] = context
        frame = context.cons[["index_weight"]].reset_index()
        frame.insert(0, "effective_date", effective_date)
        rows.append(frame)
    return BacktestResult(
        weights=pd.concat(rows, ignore_index=True).set_index(
            ["effective_date", "instrument_id"]
        ),
        reviews=reviews,
    )


def _run_manifest(
    *,
    base_currency="USD",
    calendar_id="US_RESEARCH",
    parameters=None,
):
    return RunManifest(
        schema_version=1,
        execution_id="comparison-test",
        status="complete",
        definition_fingerprint="a" * 64,
        request_fingerprint="b" * 64,
        result_fingerprint="c" * 64,
        workspace_name="comparison",
        index_id="RESEARCH_INDEX",
        created_at="2026-07-01T00:00:00+00:00",
        completed_at="2026-07-01T00:00:01+00:00",
        calendar={
            "type": "icapa.backtesting.Calendar",
            "calendar_id": calendar_id,
            "provider_name": None,
            "provider_parameters": {},
            "command": None,
        },
        request={
            "definition": {
                "base_currency": base_currency,
                "methodology_parameters": dict(parameters or {}),
            }
        },
    )


def test_comparison_aligns_union_and_reports_constituent_changes():
    baseline_result = _result((("A", 0.5), ("B", 0.5)))
    candidate_result = _result((("A", 0.4), ("C", 0.6)))
    baseline = ComparisonInput(
        name="baseline",
        backtest=baseline_result,
        analytics=run_analytics_plugins(baseline_result),
        manifest={
            "base_currency": "USD",
            "calendar_semantics": "US",
            "parameters": {"tilt": 1.0},
        },
    )
    candidate = ComparisonInput(
        name="candidate",
        backtest=candidate_result,
        analytics=run_analytics_plugins(candidate_result),
        manifest={
            "base_currency": "USD",
            "calendar_semantics": "US",
            "parameters": {"tilt": 1.5},
        },
    )

    result = compare_research_results(baseline, [candidate])

    second = result.weight_differences[
        result.weight_differences["effective_date"]
        == pd.Timestamp("2026-06-22")
    ]
    assert set(second["instrument_id"]) == {"A", "B", "C"}
    statuses = dict(zip(second["instrument_id"], second["status"]))
    assert statuses["B"] == "exit"
    assert statuses["C"] == "entrant"
    assert result.overview.iloc[-1]["one_way_weight_difference"] == pytest.approx(0.6)
    assert result.parameter_differences.iloc[0]["parameter"] == "tilt"


def test_incompatible_base_currency_fails_before_numeric_comparison():
    baseline = ComparisonInput(
        name="baseline",
        backtest=_result((("A", 0.5), ("B", 0.5))),
        manifest={"base_currency": "USD"},
    )
    candidate = ComparisonInput(
        name="candidate",
        backtest=_result((("A", 0.5), ("B", 0.5))),
        manifest={"base_currency": "EUR"},
    )

    with pytest.raises(Exception, match="base_currency"):
        compare_research_results(baseline, [candidate])


def test_current_run_manifest_schema_rejects_incompatible_currency():
    baseline = ComparisonInput(
        name="baseline",
        backtest=_result((("A", 0.5), ("B", 0.5))),
        manifest=_run_manifest(base_currency="USD"),
    )
    candidate = ComparisonInput(
        name="candidate",
        backtest=_result((("A", 0.5), ("B", 0.5))),
        manifest=_run_manifest(base_currency="EUR").as_dict(),
    )

    with pytest.raises(Exception, match="base_currency"):
        compare_research_results(baseline, [candidate])


def test_current_run_manifest_schema_rejects_incompatible_calendar():
    baseline = ComparisonInput(
        name="baseline",
        backtest=_result((("A", 0.5), ("B", 0.5))),
        manifest=_run_manifest(calendar_id="US_RESEARCH"),
    )
    candidate = ComparisonInput(
        name="candidate",
        backtest=_result((("A", 0.5), ("B", 0.5))),
        manifest=_run_manifest(calendar_id="GLOBAL_RESEARCH"),
    )

    with pytest.raises(Exception, match="calendar_semantics"):
        compare_research_results(baseline, [candidate])


def test_current_run_manifest_schema_reports_methodology_parameter_differences():
    baseline = ComparisonInput(
        name="baseline",
        backtest=_result((("A", 0.5), ("B", 0.5))),
        manifest=_run_manifest(parameters={"tilt": 1.0, "maximum_weight": 0.1}),
    )
    candidate = ComparisonInput(
        name="candidate",
        backtest=_result((("A", 0.5), ("B", 0.5))),
        manifest=_run_manifest(parameters={"tilt": 1.5, "maximum_weight": 0.1}),
    )

    comparison = compare_research_results(baseline, [candidate])

    assert comparison.parameter_differences.to_dict(orient="records") == [
        {
            "candidate": "candidate",
            "parameter": "tilt",
            "baseline_value": 1.0,
            "candidate_value": 1.5,
        }
    ]


def test_strict_policy_rejects_reportable_lineage_differences():
    baseline = ComparisonInput(
        name="baseline",
        backtest=_result((("A", 0.5), ("B", 0.5))),
        manifest={"data_snapshot": "one"},
    )
    candidate = ComparisonInput(
        name="candidate",
        backtest=_result((("A", 0.5), ("B", 0.5))),
        manifest={"data_snapshot": "two"},
    )
    spec = replace(
        ComparisonSpec(),
        compatibility_policy=CompatibilityPolicy.REQUIRE_EQUAL,
    )

    with pytest.raises(Exception, match="data_snapshot"):
        compare_research_results(baseline, [candidate], spec=spec)


def test_comparison_includes_factor_target_and_constraint_diagnostics():
    baseline_result = _result((("A", 0.5), ("B", 0.5)))
    candidate_result = _result((("A", 0.4), ("C", 0.6)))
    for result, offset, achieved in (
        (baseline_result, 0.0, 0.50),
        (candidate_result, 0.5, 0.60),
    ):
        for context in result.reviews.values():
            context.cons["quality_signal"] = [
                float(position) + offset
                for position in range(len(context.cons))
            ]
            context.diagnostics["target_diagnostics"] = [
                {
                    "name": "quality_signal",
                    "requested": 0.55,
                    "achieved": achieved,
                    "tolerance": 0.10,
                }
            ]
            context.diagnostics["constraint_diagnostics"] = [
                {
                    "name": "maximum_weight",
                    "value": max(context.cons["index_weight"]),
                    "lower": 0.0,
                    "upper": 0.8,
                }
            ]

    baseline = ComparisonInput(
        name="baseline",
        backtest=baseline_result,
        analytics=run_analytics_plugins(baseline_result),
    )
    candidate = ComparisonInput(
        name="candidate",
        backtest=candidate_result,
        analytics=run_analytics_plugins(candidate_result),
    )

    comparison = compare_research_results(baseline, [candidate])

    factor_rows = comparison.exposure_differences[
        comparison.exposure_differences["classification"]
        == "factor_signal"
    ]
    assert not factor_rows.empty
    assert factor_rows["exposure_difference"].abs().gt(0).any()
    assert {"target_attainment", "constraint"}.issubset(
        set(comparison.validation_differences["diagnostic_type"])
    )
