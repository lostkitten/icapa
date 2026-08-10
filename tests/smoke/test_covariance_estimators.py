"""Focused tests for public point-in-time covariance estimators."""

import numpy as np
import pandas as pd
import pytest

from icapa.portfolio_construction import (
    CovarianceEstimator,
    CovarianceMissingDataPolicy,
    CovarianceShrinkageTarget,
    FactorCovarianceEstimator,
    ReturnWindowSpec,
    ShrinkageCovarianceEstimator,
    estimate_covariance_for_window,
)


def _returns() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=90)
    position = np.arange(len(dates), dtype=float)
    common_one = np.sin(position / 7.0) * 0.012
    common_two = np.cos(position / 11.0) * 0.009
    return pd.DataFrame(
        {
            "A": common_one + 0.20 * common_two,
            "B": 0.75 * common_one - 0.35 * common_two,
            "C": -0.25 * common_one + 0.90 * common_two,
            "D": 0.40 * common_one + 0.30 * common_two,
        },
        index=dates,
    )


def test_shrinkage_covariance_matches_its_declared_diagonal_target():
    returns = _returns()
    intensity = 0.35
    estimator = ShrinkageCovarianceEstimator(
        shrinkage=intensity,
        target=CovarianceShrinkageTarget.DIAGONAL,
        minimum_observations=40,
        ridge=0.0,
        ensure_positive_semidefinite=False,
        missing_data_policy=CovarianceMissingDataPolicy.COMPLETE_CASE,
    )

    estimate = estimator.estimate(returns)
    sample = returns.cov().to_numpy(dtype=float)
    expected = (1.0 - intensity) * sample + intensity * np.diag(
        np.diag(sample)
    )

    assert isinstance(estimator, CovarianceEstimator)
    np.testing.assert_allclose(estimate.matrix, expected, atol=1e-15, rtol=0.0)
    assert estimate.estimator_name == "shrinkage_covariance"
    assert estimate.metadata["shrinkage"] == pytest.approx(intensity)
    assert estimate.metadata["shrinkage_target"] == "diagonal"


def test_scaled_identity_shrinkage_is_deterministic_and_positive_semidefinite():
    estimator = ShrinkageCovarianceEstimator(
        shrinkage=0.6,
        target=CovarianceShrinkageTarget.SCALED_IDENTITY,
        minimum_observations=40,
    )

    first = estimator.estimate(_returns())
    second = estimator.estimate(_returns().copy(deep=True))

    np.testing.assert_array_equal(first.matrix, second.matrix)
    assert first.minimum_eigenvalue >= -1e-12
    assert first.metadata["shrinkage_target"] == "scaled_identity"


def test_factor_covariance_preserves_variances_and_uses_return_window():
    returns = _returns()
    window = ReturnWindowSpec(
        lookback=60,
        minimum_observations=50,
        end_lag_observations=1,
    )
    estimator = FactorCovarianceEstimator(
        factor_count=2,
        minimum_observations=50,
        ridge=1e-9,
    )

    resolved, estimate = estimate_covariance_for_window(
        estimator,
        returns,
        window,
        returns.index[75],
    )
    selected = returns.loc[list(resolved.business_dates)]
    expected_variances = selected.var(ddof=1).to_numpy(dtype=float) + 1e-9
    repeated = estimator.estimate(selected)

    assert isinstance(estimator, CovarianceEstimator)
    assert resolved.end_date == returns.index[74]
    assert resolved.observation_count == 60
    np.testing.assert_allclose(
        np.diag(estimate.matrix),
        expected_variances,
        atol=1e-15,
        rtol=0.0,
    )
    np.testing.assert_array_equal(estimate.matrix, repeated.matrix)
    assert estimate.minimum_eigenvalue >= -1e-12
    assert estimate.estimator_name == "factor_covariance"
    assert estimate.metadata["factor_count"] == 2
    assert 0.0 <= estimate.metadata["explained_variance_ratio"] <= 1.0
    assert estimate.metadata["window_end_date"] == resolved.end_date.isoformat()


@pytest.mark.parametrize("shrinkage", [-0.01, 1.01, np.nan])
def test_shrinkage_covariance_rejects_invalid_intensity(shrinkage):
    with pytest.raises(ValueError, match="shrinkage"):
        ShrinkageCovarianceEstimator(shrinkage=shrinkage)


@pytest.mark.parametrize("factor_count", [0, True, 1.5, "2"])
def test_factor_covariance_requires_a_positive_integer_factor_count(factor_count):
    with pytest.raises(ValueError, match="factor_count"):
        FactorCovarianceEstimator(factor_count=factor_count)


def test_factor_covariance_rejects_unsupported_factor_count_and_missing_sample():
    returns = _returns()
    with pytest.raises(ValueError, match="factor_count"):
        FactorCovarianceEstimator(factor_count=5).estimate(returns)

    incomplete = returns.copy()
    incomplete.loc[incomplete.index[:25], "A"] = np.nan
    incomplete.loc[incomplete.index[-25:], "D"] = np.nan
    with pytest.raises(ValueError, match="complete-case"):
        FactorCovarianceEstimator(
            factor_count=2,
            minimum_observations=60,
        ).estimate(incomplete)
