"""Smoke tests for the in-memory context and calculation variants."""

import pandas as pd
import pytest

from icapa.backtesting.simulation import (
    DividendTreatment,
    calculate_index_returns,
)
from icapa.portfolio_construction.context import DataContext


def test_context_keeps_constituent_and_daily_dimensions_separate():
    context = DataContext(
        reference_date="2026-06-05",
        effective_date="2026-06-22",
        index_id="SYNTHETIC_DEMO",
    )
    universe = pd.DataFrame(
        {"instrument_id": [1, 2], "benchmark_weight": [0.4, 0.6]}
    )
    context.set_dataframe(universe)

    daily = pd.DataFrame(
        {
            "instrument_id": [1, 2],
            "business_date": pd.to_datetime(["2026-06-05", "2026-06-05"]),
            "price_return": [0.01, -0.01],
        }
    ).set_index(["instrument_id", "business_date"])
    context.set_dataframe(daily)

    assert context.cons.index.name == "instrument_id"
    assert context.daily is not None
    assert context.daily.index.names == ["instrument_id", "business_date"]


def test_dividend_treatments_are_explicit_and_deterministic():
    daily = pd.DataFrame(
        {
            "instrument_id": [1, 2],
            "business_date": pd.to_datetime(["2026-06-05", "2026-06-05"]),
            "price_return": [0.02, -0.01],
            "gross_dividend": [0.01, 0.00],
            "net_dividend": [0.008, 0.00],
        }
    )
    weights = pd.Series([0.4, 0.6], index=pd.Index([1, 2], name="instrument_id"))

    standard = calculate_index_returns(daily, weights, DividendTreatment.STANDARD)
    alternative = calculate_index_returns(
        daily, weights, DividendTreatment.ALTERNATIVE
    )

    expected_price = 0.4 * 0.02 + 0.6 * -0.01
    assert standard.iloc[0]["price_return"] == pytest.approx(expected_price)
    assert standard.iloc[0]["gross_total_return"] == pytest.approx(
        (1.0 + expected_price) / (1.0 - 0.004) - 1.0
    )
    assert alternative.iloc[0]["gross_total_return"] == pytest.approx(
        expected_price + 0.004
    )
    assert standard.iloc[0]["gross_total_return"] != pytest.approx(
        alternative.iloc[0]["gross_total_return"]
    )
