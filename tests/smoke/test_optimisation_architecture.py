"""Smoke tests for solver-independent optimization contracts."""

import numpy as np
import pytest

from icapa.portfolio_construction.optimization import (
    OptimizationProblem,
    ScipySLSQPSolver,
    squared_distance_objective,
    turnover_constraint,
)


def test_solver_reports_diagnostics_and_supports_nonlinear_constraints():
    initial = np.array([0.5, 0.5])
    target = np.array([0.9, 0.1])
    objective, gradient = squared_distance_objective(target)
    result = ScipySLSQPSolver().solve(
        OptimizationProblem(
            name="turnover_limited_demo",
            objective=objective,
            initial_weights=initial,
            lower_bounds=np.zeros(2),
            upper_bounds=np.ones(2),
            gradient=gradient,
            nonlinear_constraints=(turnover_constraint(initial, 0.1),),
        )
    )

    assert float(result.weights.sum()) == pytest.approx(1.0)
    assert 0.5 * float(np.abs(result.weights - initial).sum()) <= 0.1 + 1e-7
    assert result.solver_name == "scipy_slsqp"
    assert result.maximum_constraint_violation <= 1e-7
