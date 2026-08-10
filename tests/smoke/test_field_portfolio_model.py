from __future__ import annotations

import numpy as np
import pandas as pd

from icapa.portfolio_construction import (
    FieldExposureConstraintSpec,
    GroupWeightConstraintSpec,
    LiquidityConstraintSpec,
    PortfolioModelSpec,
)


def _constituents() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instrument_id": ["A", "B", "C", "D"],
            "benchmark_weight": [0.30, 0.20, 0.25, 0.25],
            "previous_weight": [0.28, 0.22, 0.24, 0.26],
            "country": ["US", "US", "CA", "CA"],
            "issuer": ["I1", "I2", "I2", "I3"],
            "factor": [0.5, -0.2, 0.1, -0.1],
            "capacity": [50.0, 40.0, 60.0, 80.0],
        }
    )


def test_field_model_compiles_groups_issuer_exposure_and_liquidity_bounds():
    spec = PortfolioModelSpec(
        name="field_model",
        constituents=_constituents(),
        groups=(
            GroupWeightConstraintSpec(
                field="country",
                bounds={"US": (-0.05, 0.10), "CA": (-0.10, 0.05)},
                relative_to_field="benchmark_weight",
            ),
            GroupWeightConstraintSpec(
                field="issuer",
                bounds={"I2": (0.30, 0.55)},
            ),
        ),
        exposures=(
            FieldExposureConstraintSpec(
                field="factor",
                lower=-0.05,
                upper=0.20,
            ),
        ),
        liquidity=LiquidityConstraintSpec(
            capacity_field="capacity",
            portfolio_value=100.0,
            participation_rate=0.5,
        ),
        previous_weight_field="previous_weight",
        maximum_one_way_turnover=0.20,
        tracking_error_covariance=pd.DataFrame(
            np.eye(4) * 0.04,
            index=["A", "B", "C", "D"],
            columns=["A", "B", "C", "D"],
        ),
        maximum_tracking_error=0.10,
    )

    model = spec.compile()

    assert tuple(model.variables.instrument_ids) == ("A", "B", "C", "D")
    np.testing.assert_allclose(
        model.variables.upper_bounds,
        [0.25, 0.20, 0.30, 0.40],
    )
    assert {item.name for item in model.linear_constraints} == {
        "country:CA",
        "country:US",
        "issuer:I2",
        "exposure:factor",
    }
    assert {item.name for item in model.nonlinear_constraints} == {
        "one_way_turnover",
        "tracking_error_squared",
    }
    country_us = next(
        item
        for item in model.linear_constraints
        if item.name == "country:US"
    )
    assert country_us.lower == 0.45
    assert country_us.upper == 0.60
    assert model.metadata["field_model"]["has_liquidity_bounds"]


def test_field_model_aligns_covariance_by_instrument_id():
    covariance = pd.DataFrame(
        np.diag([0.01, 0.02, 0.03, 0.04]),
        index=["D", "C", "B", "A"],
        columns=["D", "C", "B", "A"],
    )
    model = PortfolioModelSpec(
        name="aligned_covariance",
        constituents=_constituents(),
        tracking_error_covariance=covariance,
        maximum_tracking_error=0.20,
    ).compile()
    constraint = model.nonlinear_constraints[0]

    assert constraint.name == "tracking_error_squared"
    assert constraint.function(
        np.asarray([0.31, 0.19, 0.25, 0.25])
    ) > 0
