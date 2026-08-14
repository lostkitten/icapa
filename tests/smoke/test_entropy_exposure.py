"""Public integration tests for Entropy Exposure construction."""

from __future__ import annotations

import pandas as pd
import pytest

from icapa.portfolio_construction import DataContext
from icapa.portfolio_construction.engines import (
    EntropyExposureEngine,
    EntropyExposureMode,
    ExposureTarget,
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
