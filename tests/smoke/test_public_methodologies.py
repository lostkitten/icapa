"""Public execution and recipe coverage for every bundled methodology."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icapa.data_sources import register_provider, registry
from icapa.portfolio_construction import (
    DataContext,
    IndexRecipe,
    RecipeWeightProducer,
    StageRuntime,
)
from icapa.portfolio_construction.methodologies import (
    FactorTiltMethodology,
    MinimumVarianceMethodology,
    QuantileSelectionMethodology,
)


_PROVIDER_NAME = "public_methodology_test"
_UNIVERSE_ID = "PUBLIC_TEST_UNIVERSE"
_REFERENCE_DATE = pd.Timestamp("2026-06-05")
_EFFECTIVE_DATE = pd.Timestamp("2026-06-22")
_INSTRUMENT_IDS = ("A", "B", "C", "D", "E", "F")
_METHODOLOGY_NAMES = (
    "factor_tilt",
    "minimum_variance",
    "quantile_selection",
)


class _PublicMethodologyProvider:
    """Small deterministic provider shared by direct and recipe executions."""

    def load_universe(
        self,
        universe_id,
        reference_date,
        effective_date,
        **parameters,
    ):
        del parameters
        if universe_id != _UNIVERSE_ID:
            raise ValueError(f"unsupported test universe: {universe_id}")
        benchmark = np.array([0.10, 0.15, 0.20, 0.20, 0.15, 0.20])
        return pd.DataFrame(
            {
                "instrument_id": _INSTRUMENT_IDS,
                "name": [f"Instrument {item}" for item in _INSTRUMENT_IDS],
                "country": ["US", "US", "CA", "CA", "GB", "GB"],
                "industry": ["Technology", "Industrials"] * 3,
                "shares": [1.0] * len(_INSTRUMENT_IDS),
                "free_float": [1.0] * len(_INSTRUMENT_IDS),
                "price": benchmark * 1_000.0,
                "currency": ["USD"] * len(_INSTRUMENT_IDS),
                "base_currency": ["USD"] * len(_INSTRUMENT_IDS),
                "fx_rate": [1.0] * len(_INSTRUMENT_IDS),
                "market_cap": benchmark * 1_000_000.0,
                "benchmark_weight": benchmark,
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
        del data_type, parameters
        values = {
            "quality": [-1.5, -0.8, -0.2, 0.5, 1.0, 1.6],
            "value": [1.4, 0.9, 0.4, -0.1, -0.7, -1.2],
            "exposure": [0.10, 0.25, 0.40, 0.60, 0.80, 1.00],
        }
        positions = {
            instrument_id: offset
            for offset, instrument_id in enumerate(_INSTRUMENT_IDS)
        }
        result = {
            "instrument_id": list(instrument_ids),
            "reference_date": pd.Timestamp(reference_date),
        }
        for field in fields:
            result[field] = [
                values[field][positions[instrument_id]]
                for instrument_id in instrument_ids
            ]
        return pd.DataFrame(result)

    def load_daily_market_data(
        self,
        instrument_ids,
        start_date,
        end_date,
        **parameters,
    ):
        del parameters
        dates = pd.bdate_range(start_date, end_date)
        time = np.arange(len(dates), dtype=float)
        volatility = (0.0040, 0.0030, 0.0012, 0.0008, 0.0025, 0.0015)
        records = []
        for instrument_id in instrument_ids:
            position = _INSTRUMENT_IDS.index(instrument_id)
            price_returns = (
                0.00005 * (position + 1)
                + volatility[position]
                * np.sin(time / (2.0 + 0.3 * position))
                + 0.3
                * volatility[position]
                * np.cos(time / (4.0 + 0.2 * position))
            )
            records.extend(
                {
                    "instrument_id": instrument_id,
                    "business_date": business_date,
                    "price_return": float(price_return),
                    "gross_dividend": 0.0,
                    "net_dividend": 0.0,
                    "market_cap": 1_000_000.0 * (position + 1),
                }
                for business_date, price_return in zip(dates, price_returns)
            )
        return pd.DataFrame.from_records(records)


@pytest.fixture
def public_methodology_provider():
    provider = _PublicMethodologyProvider()
    register_provider(_PROVIDER_NAME, provider, replace=True)
    try:
        yield provider
    finally:
        registry.unregister(_PROVIDER_NAME)


def _build_methodology(name):
    common = {
        "universe_id": _UNIVERSE_ID,
        "universe_provider_name": _PROVIDER_NAME,
    }
    if name == "factor_tilt":
        return FactorTiltMethodology(
            factor_tilts={"quality": 0.7, "value": 0.3},
            factor_provider_name=_PROVIDER_NAME,
            **common,
        )
    if name == "minimum_variance":
        return MinimumVarianceMethodology(
            returns_provider_name=_PROVIDER_NAME,
            start_date="2026-03-02",
            end_date=str(_REFERENCE_DATE.date()),
            minimum_observations=20,
            **common,
        )
    if name == "quantile_selection":
        return QuantileSelectionMethodology(
            signal_weights={"quality": 0.7, "value": 0.3},
            signal_provider_name=_PROVIDER_NAME,
            selection_fraction=0.5,
            **common,
        )
    raise AssertionError(f"unsupported test methodology: {name}")


def _new_context():
    return DataContext(
        index_id="PUBLIC_TEST_INDEX",
        universe_id=_UNIVERSE_ID,
        reference_date=_REFERENCE_DATE,
        effective_date=_EFFECTIVE_DATE,
    )


def _assert_valid_weights(context):
    weights = context.cons["index_weight"]
    assert np.isfinite(weights).all()
    assert (weights >= -1.0e-12).all()
    assert float(weights.sum()) == pytest.approx(1.0, abs=1.0e-8)


def _assert_methodology_semantics(name, context):
    frame = context.cons
    if name == "factor_tilt":
        active_ratio = frame["index_weight"] / frame["benchmark_weight"]
        ordered = active_ratio.loc[frame["factor_score"].sort_values().index]
        assert (np.diff(ordered.to_numpy()) >= -1.0e-7).all()
    elif name == "minimum_variance":
        diagnostics = context.diagnostics["minimum_variance_optimization"]
        assert diagnostics["portfolio_variance"] <= (
            diagnostics["benchmark_variance"] + 1.0e-12
        )
    elif name == "quantile_selection":
        selected = frame["selected"].astype(bool)
        assert int(selected.sum()) == 3
        assert float(frame.loc[selected, "selection_score"].min()) >= float(
            frame.loc[~selected, "selection_score"].max()
        )


@pytest.mark.parametrize("methodology_name", _METHODOLOGY_NAMES)
def test_each_public_methodology_executes_with_basic_semantics(
    methodology_name,
    public_methodology_provider,
):
    del public_methodology_provider
    context = _new_context()

    _build_methodology(methodology_name).execute(context)

    _assert_valid_weights(context)
    _assert_methodology_semantics(methodology_name, context)


@pytest.mark.parametrize("methodology_name", _METHODOLOGY_NAMES)
def test_each_public_methodology_recipe_matches_direct_execution(
    methodology_name,
    public_methodology_provider,
):
    methodology = _build_methodology(methodology_name)
    direct = methodology.execute(_new_context())
    recipe = methodology.to_recipe()
    assert isinstance(recipe, IndexRecipe)

    capabilities = {
        capability
        for node in recipe.nodes
        for capability in node.stage.requirements.provider_capabilities
    }
    expected_capabilities = {
        "load_universe",
        (
            "load_daily_market_data"
            if methodology_name == "minimum_variance"
            else "load_third_party_data"
        ),
    }
    assert capabilities == expected_capabilities

    runtime = StageRuntime(
        providers={
            "load_universe": public_methodology_provider,
            "load_third_party_data": public_methodology_provider,
            "load_daily_market_data": public_methodology_provider,
        },
        provider_revisions={
            "load_universe": "public-test-v1",
            "load_third_party_data": "public-test-v1",
            "load_daily_market_data": "public-test-v1",
        },
        data_revision="public-test-v1",
        code_revision="public-test-v1",
    )
    recipe_context = _new_context()
    RecipeWeightProducer(recipe, runtime=runtime).execute(recipe_context)

    _assert_valid_weights(recipe_context)
    np.testing.assert_allclose(
        recipe_context.cons["index_weight"],
        direct.cons["index_weight"],
        atol=1.0e-8,
        rtol=0.0,
    )
