"""Tests for Entropy-Guided Multiplicative Update optimization contracts."""

import numpy as np
import pytest
from scipy.optimize import minimize

from icapa.portfolio_construction.optimization import (
    EGMUConstrainedElasticSolver,
    EGMUElasticSolver,
    EGMUNewtonSolver,
    EGMUProjectionSolver,
    LinearConstraintSpec,
    OptimizationError,
    relative_entropy,
)
from icapa.portfolio_construction.optimization.entropy_dual import (
    _stable_dual_increment,
)


def test_stable_dual_increment_normalizes_extended_precision_weights():
    upper_half = np.nextafter(
        np.nextafter(0.5, np.inf),
        np.inf,
    )
    gain = _stable_dual_increment(
        np.asarray([[0.0], [1.0]]),
        np.asarray([0.5, upper_half]),
        np.asarray([0.5]),
        np.asarray([0.0]),
        np.asarray([1.0]),
        0.0,
    )

    assert gain == 0.0


def test_egmu_newton_hits_multi_exposure_targets_and_beats_other_feasible_weights():
    prior = np.asarray([0.15, 0.20, 0.25, 0.40])
    exposures = np.asarray(
        [
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ]
    )
    target = np.asarray([0.25, 0.55])

    result = EGMUNewtonSolver(tolerance=1.0e-11).solve(
        exposures,
        prior,
        target,
    )

    assert result.status == "converged"
    assert result.weights.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(exposures.T @ result.weights, target, atol=1e-9)
    null_direction = np.asarray([0.25, -1.0, -0.25, 1.0])
    for displacement in (-0.05, 0.05):
        candidate = result.weights + displacement * null_direction
        assert np.all(candidate > 0)
        np.testing.assert_allclose(exposures.T @ candidate, target, atol=1e-9)
        assert relative_entropy(result.weights, prior) < relative_entropy(
            candidate,
            prior,
        )


def test_egmu_elastic_reports_a_controlled_residual_for_an_unreachable_target():
    prior = np.asarray([0.2, 0.3, 0.5])
    exposures = np.asarray([[0.0], [0.5], [1.0]])

    result = EGMUElasticSolver(
        penalty=10.0,
        tolerance=1.0e-11,
    ).solve(exposures, prior, [2.0])

    assert result.status == "converged"
    assert result.weights.sum() == pytest.approx(1.0)
    achieved = float((exposures.T @ result.weights).item())
    assert achieved < 1.0
    residual = achieved - 2.0
    assert residual == pytest.approx(
        -float(result.theta[0]) / 10.0,
        abs=1.0e-9,
    )


def test_egmu_projection_handles_equality_one_sided_and_weight_constraints():
    prior = np.asarray([0.20, 0.30, 0.50])
    constraints = (
        LinearConstraintSpec(
            coefficients=np.asarray([0.0, 0.5, 1.0]),
            lower=0.70,
            upper=np.inf,
            name="minimum_exposure",
        ),
        LinearConstraintSpec(
            coefficients=np.asarray([0.0, 1.0, 1.0]),
            lower=0.85,
            upper=0.85,
            name="group_bc",
        ),
    )

    result = EGMUProjectionSolver(tolerance=1.0e-10).solve(
        prior,
        constraints,
        lower_bounds=[0.10, 0.0, 0.0],
        upper_bounds=[0.15, 1.0, 1.0],
    )

    assert result.status == "converged"
    assert result.maximum_constraint_violation <= 1.0e-9
    assert result.weights[0] == pytest.approx(0.15, abs=1.0e-9)
    assert float(constraints[0].coefficients @ result.weights) >= 0.70 - 1e-9
    assert result.weights.sum() == pytest.approx(1.0)


def test_egmu_projection_rejects_targets_outside_the_prior_support_hull():
    with pytest.raises(OptimizationError, match="outside the exposure hull"):
        EGMUProjectionSolver().solve(
            [0.5, 0.5],
            [
                LinearConstraintSpec(
                    coefficients=np.asarray([0.0, 1.0]),
                    lower=1.5,
                    upper=np.inf,
                    name="unreachable",
                )
            ],
        )


def test_constrained_elastic_egmu_preserves_hard_caps_and_soft_one_sided_target():
    prior = np.asarray([0.2, 0.3, 0.5])
    exposures = np.asarray([[0.0], [0.5], [1.0]])
    result = EGMUConstrainedElasticSolver(
        penalty=20.0,
        tolerance=1.0e-9,
    ).solve(
        exposures,
        prior,
        [0.9],
        [np.inf],
        upper_bounds=[1.0, 1.0, 0.55],
    )

    assert result.status == "converged"
    assert result.weights.sum() == pytest.approx(1.0)
    assert result.weights[2] <= 0.55 + 1.0e-8
    assert float((exposures.T @ result.weights).item()) > float(
        (exposures.T @ prior).item()
    )
    assert result.maximum_constraint_violation <= 1.0e-8

    def objective(weights):
        kl_value = float(np.sum(weights * np.log(weights / prior)))
        achieved = float((exposures.T @ weights).item())
        return kl_value + 0.5 * 20.0 * max(0.9 - achieved, 0.0) ** 2

    reference = minimize(
        objective,
        prior,
        method="SLSQP",
        bounds=[(1.0e-12, 1.0)] * len(prior),
        constraints=(
            {"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
            {"type": "ineq", "fun": lambda weights: 0.55 - weights[2]},
        ),
        options={"ftol": 1.0e-13, "maxiter": 1_000},
    )
    assert reference.success
    np.testing.assert_allclose(result.weights, reference.x, atol=1.0e-6)


def test_egmu_requires_a_strictly_positive_prior():
    with pytest.raises(ValueError, match="strictly positive"):
        EGMUNewtonSolver().solve(
            np.asarray([[0.0], [1.0]]),
            [1.0, 0.0],
            [0.5],
        )


def test_egmu_prior_normalisation_is_invariant_to_large_common_scale():
    exposures = np.asarray([[0.0], [1.0]])

    result = EGMUNewtonSolver().solve(
        exposures,
        [1.0e308, 1.0e308],
        [0.6],
    )

    assert result.status == "converged"
    np.testing.assert_allclose(result.weights, [0.4, 0.6], atol=1.0e-8)


def test_bounded_simplex_block_scales_to_thousands_of_instruments():
    instrument_count = 5_000
    prior = np.full(instrument_count, 0.5 / (instrument_count - 1))
    prior[0] = 0.5
    upper = np.full(instrument_count, 1.0)
    upper[0] = 0.01

    result = EGMUProjectionSolver().solve(
        prior,
        (),
        upper_bounds=upper,
    )

    assert result.status == "converged"
    assert result.weights[0] == pytest.approx(0.01, abs=1.0e-9)
    assert result.weights.sum() == pytest.approx(1.0)


def test_bounded_simplex_bracketing_handles_tiny_prior_with_active_floor():
    result = EGMUProjectionSolver().solve(
        [1.0e-50, 0.4, 0.6],
        (),
        lower_bounds=[0.1, 0.0, 0.0],
    )

    assert result.status == "converged"
    np.testing.assert_allclose(result.weights, [0.1, 0.36, 0.54], atol=1e-9)
