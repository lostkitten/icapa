"""Public integration tests for Entropy Exposure construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icapa.data_sources import register_provider, registry
from icapa.portfolio_construction import DataContext
from icapa.portfolio_construction.engines import (
    EntropyExposureEngine,
    EntropyExposureMode,
    ExposureTarget,
    TargetDirection,
)
from icapa.portfolio_construction.methodologies import (
    EntropyExposureMethodology,
)


class _ExposureProvider:
    def load_universe(
        self,
        universe_id,
        reference_date,
        effective_date,
        **parameters,
    ):
        del universe_id, parameters
        return pd.DataFrame(
            {
                "instrument_id": ["A", "B", "C"],
                "name": ["A", "B", "C"],
                "country": ["US", "US", "GB"],
                "industry": ["One", "One", "Two"],
                "shares": [1.0, 1.0, 1.0],
                "free_float": [1.0, 1.0, 1.0],
                "price": [1.0, 1.0, 1.0],
                "currency": ["USD", "USD", "GBP"],
                "base_currency": ["USD", "USD", "USD"],
                "fx_rate": [1.0, 1.0, 1.0],
                "market_cap": [20.0, 30.0, 50.0],
                "benchmark_weight": [0.2, 0.3, 0.5],
                "reference_date": pd.Timestamp(reference_date),
                "effective_date": pd.Timestamp(effective_date),
            }
        )

    def load_third_party_data(
        self,
        data_type,
        instrument_ids,
        fields,
        reference_date,
        parameters=None,
    ):
        del data_type, fields, parameters
        return pd.DataFrame(
            {
                "instrument_id": list(instrument_ids),
                "quality": [0.0, 0.5, 1.0],
                "reference_date": pd.Timestamp(reference_date),
            }
        )


def test_entropy_exposure_engine_preserves_caps_during_elastic_fallback():
    context = DataContext()
    context.set_dataframe(
        pd.DataFrame(
            {
                "instrument_id": ["A", "B"],
                "benchmark_weight": [0.5, 0.5],
                "quality": [0.0, 1.0],
            }
        )
    )

    EntropyExposureEngine(
        targets=[ExposureTarget("quality", 2.0)],
        mode=EntropyExposureMode.HARD_THEN_ELASTIC,
        elastic_penalty=100.0,
        maximum_weight=0.6,
    ).execute(context)

    diagnostics = context.diagnostics["entropy_exposure_optimization"]
    assert diagnostics["requested_mode"] == "hard_then_elastic"
    assert diagnostics["mode"] == "elastic"
    assert diagnostics["maximum_constraint_violation"] <= 1.0e-7
    assert context.cons["index_weight"].max() <= 0.6 + 1.0e-7
    assert context.cons["index_weight"].sum() == pytest.approx(1.0)


def test_entropy_exposure_methodology_executes_and_declares_recipe_inputs():
    provider_name = "entropy_exposure_test"
    provider = _ExposureProvider()
    register_provider(provider_name, provider, replace=True)
    methodology = EntropyExposureMethodology(
        targets=[
            ExposureTarget(
                "quality",
                0.65,
                direction=TargetDirection.EQUAL,
            )
        ],
        universe_id="TEST_UNIVERSE",
        universe_provider_name=provider_name,
        target_provider_name=provider_name,
    )
    context = DataContext(
        reference_date=pd.Timestamp("2026-06-05"),
        effective_date=pd.Timestamp("2026-06-22"),
    )

    try:
        methodology.execute(context)
    finally:
        registry.unregister(provider_name)

    weights = context.cons["index_weight"]
    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1.0)
    assert float(weights @ context.cons["quality"]) == pytest.approx(
        0.65,
        abs=1.0e-8,
    )

    stage = methodology.to_recipe().nodes[0].stage
    assert set(stage.provider_capabilities) == {
        "load_universe",
        "load_third_party_data",
    }
    requests = {item.capability: item for item in stage.provider_requests}
    assert requests["load_third_party_data"].request_parameters["fields"] == (
        "quality",
    )
    assert requests["load_universe"].expected_provider_name == provider_name
