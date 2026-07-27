"""Extensible constrained portfolio-optimisation API."""

from .constraints import (
    LinearConstraintSpec,
    NonlinearConstraintSpec,
    group_constraint_specs,
    tracking_error_constraint,
    turnover_constraint,
    weight_bounds,
)
from .objectives import minimum_variance_objective, squared_distance_objective
from .problem import OptimizationError, OptimizationProblem, OptimizationResult
from .solvers import PortfolioSolver, ScipySLSQPSolver, solve_slsqp

__all__ = [
    "LinearConstraintSpec",
    "NonlinearConstraintSpec",
    "OptimizationError",
    "OptimizationProblem",
    "OptimizationResult",
    "PortfolioSolver",
    "ScipySLSQPSolver",
    "group_constraint_specs",
    "minimum_variance_objective",
    "solve_slsqp",
    "squared_distance_objective",
    "tracking_error_constraint",
    "turnover_constraint",
    "weight_bounds",
]
