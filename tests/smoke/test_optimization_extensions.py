"""Tests for additive model, diagnostics, and rolling risk contracts."""

import numpy as np
import pandas as pd
import pytest

from icapa.portfolio_construction.optimization import (
    ConstraintStatus,
    CovarianceMissingDataPolicy,
    FeasibilityStatus,
    LinearConstraintSpec,
    MinimumVarianceObjectiveSpec,
    OSQPBackend,
    OptimizationModelSpec,
    PortfolioSolverModelAdapter,
    ReturnWindowSpec,
    SampleCovarianceEstimator,
    ScipySLSQPSolver,
    SolverCapability,
    SolverRouter,
    SquaredDistanceObjectiveSpec,
    WeightVariableSpec,
    check_linear_feasibility,
    estimate_covariance_for_window,
    evaluate_problem_constraints,
)


def test_declarative_model_compiles_to_existing_solver_and_reports_constraints():
    variables = WeightVariableSpec(
        instrument_ids=("A", "B", "C"),
        initial_weights=[1 / 3, 1 / 3, 1 / 3],
        lower_bounds=[0.0, 0.0, 0.0],
        upper_bounds=[0.7, 0.7, 0.7],
    )
    model = OptimizationModelSpec(
        name="target_weights",
        variables=variables,
        objective=SquaredDistanceObjectiveSpec([0.6, 0.3, 0.1]),
    )
    backend = PortfolioSolverModelAdapter(ScipySLSQPSolver())
    result = backend.solve_model(model)
    report = evaluate_problem_constraints(
        model.compile(),
        result.weights,
        instrument_ids=variables.instrument_ids,
    )

    np.testing.assert_allclose(result.weights, [0.6, 0.3, 0.1], atol=1e-5)
    assert report.feasible is True
    assert report.maximum_violation <= 1e-7
    assert report.evaluations[0].status is ConstraintStatus.BINDING


def test_model_solver_capability_mismatch_fails_explicitly():
    variables = WeightVariableSpec(
        instrument_ids=("A", "B"),
        initial_weights=[0.5, 0.5],
        lower_bounds=[0.0, 0.0],
        upper_bounds=[1.0, 1.0],
    )
    model = OptimizationModelSpec(
        name="quadratic",
        variables=variables,
        objective=MinimumVarianceObjectiveSpec(np.eye(2)),
    )
    backend = PortfolioSolverModelAdapter(
        ScipySLSQPSolver(),
        supported_capabilities=frozenset({SolverCapability.LINEAR_CONSTRAINTS}),
    )
    with pytest.raises(ValueError, match="QUADRATIC".lower()):
        backend.solve_model(model)


def test_osqp_backend_and_explicit_router_solve_sparse_qp():
    variables = WeightVariableSpec(
        instrument_ids=("A", "B"),
        initial_weights=[0.5, 0.5],
        lower_bounds=[0.0, 0.0],
        upper_bounds=[1.0, 1.0],
    )
    model = OptimizationModelSpec(
        name="capped_target",
        variables=variables,
        objective=SquaredDistanceObjectiveSpec([0.8, 0.2]),
        linear_constraints=(
            LinearConstraintSpec(
                coefficients=np.array([1.0, 0.0]),
                lower=0.0,
                upper=0.6,
                name="asset_a_cap",
            ),
        ),
    )
    router = SolverRouter([OSQPBackend()])
    with pytest.raises(ValueError, match="backend_name"):
        router.solve_model(model)

    solved = router.solve_model_with_diagnostics(model, backend_name="osqp")
    np.testing.assert_allclose(solved.result.weights, [0.6, 0.4], atol=1e-7)
    assert solved.constraints.feasible is True
    assert solved.backend_name == "osqp"


def test_phase_one_feasibility_reports_conflicting_linear_constraints():
    variables = WeightVariableSpec(
        instrument_ids=("A", "B"),
        initial_weights=[0.5, 0.5],
        lower_bounds=[0.0, 0.0],
        upper_bounds=[1.0, 1.0],
    )
    model = OptimizationModelSpec(
        name="infeasible",
        variables=variables,
        objective=SquaredDistanceObjectiveSpec([0.5, 0.5]),
        linear_constraints=(
            LinearConstraintSpec(
                coefficients=np.array([1.0, 0.0]),
                lower=0.8,
                upper=np.inf,
                name="asset_a_floor",
            ),
            LinearConstraintSpec(
                coefficients=np.array([1.0, 0.0]),
                lower=-np.inf,
                upper=0.2,
                name="asset_a_cap",
            ),
        ),
    )
    report = check_linear_feasibility(model)
    assert report.status is FeasibilityStatus.INFEASIBLE
    assert report.minimum_total_violation == pytest.approx(0.6)
    assert report.constraints is not None
    assert report.constraints.feasible is False


def test_constraint_report_marks_violations_per_instrument():
    variables = WeightVariableSpec(
        instrument_ids=("A", "B"),
        initial_weights=[0.5, 0.5],
        lower_bounds=[0.0, 0.0],
        upper_bounds=[0.6, 0.6],
    )
    model = OptimizationModelSpec(
        name="diagnostic",
        variables=variables,
        objective=SquaredDistanceObjectiveSpec([0.5, 0.5]),
    )
    report = evaluate_problem_constraints(
        model.compile(),
        [0.7, 0.3],
        instrument_ids=variables.instrument_ids,
    )
    by_name = {item.name: item for item in report.evaluations}
    assert report.feasible is False
    assert by_name["weight:A"].status is ConstraintStatus.VIOLATED
    assert by_name["weight:A"].violation == pytest.approx(0.1)


def test_rolling_return_window_prevents_lookahead_and_covariance_is_psd():
    dates = pd.bdate_range("2024-01-02", periods=320)
    returns = pd.DataFrame(
        {
            "A": np.sin(np.arange(len(dates)) / 10.0) * 0.01,
            "B": np.cos(np.arange(len(dates)) / 12.0) * 0.008,
            "C": np.sin(np.arange(len(dates)) / 15.0) * 0.006,
        },
        index=dates,
    )
    reference_date = dates[250]
    window = ReturnWindowSpec(
        lookback=60,
        minimum_observations=40,
        end_lag_observations=1,
    )
    estimator = SampleCovarianceEstimator(
        minimum_observations=40,
        ridge=1e-8,
        missing_data_policy=CovarianceMissingDataPolicy.COMPLETE_CASE,
    )
    resolved, estimate = estimate_covariance_for_window(
        estimator,
        returns,
        window,
        reference_date,
    )

    assert resolved.observation_count == 60
    assert resolved.end_date == dates[249]
    assert resolved.end_date < reference_date
    assert estimate.matrix.shape == (3, 3)
    assert estimate.minimum_eigenvalue >= -1e-12
    assert estimate.metadata["window_kind"] == "observations"


def test_pairwise_covariance_requires_sufficient_overlap():
    dates = pd.bdate_range("2025-01-02", periods=30)
    returns = pd.DataFrame(
        {
            "A": [0.01] * 15 + [np.nan] * 15,
            "B": [np.nan] * 15 + [0.01] * 15,
        },
        index=dates,
    )
    estimator = SampleCovarianceEstimator(
        minimum_observations=10,
        missing_data_policy=CovarianceMissingDataPolicy.PAIRWISE,
    )
    with pytest.raises(ValueError, match="overlapping"):
        estimator.estimate(returns)
