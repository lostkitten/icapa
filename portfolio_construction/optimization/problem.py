"""Solver-independent optimization contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .constraints import LinearConstraintSpec, NonlinearConstraintSpec


class OptimizationError(RuntimeError):
    """Raised when a portfolio problem has no verified solution."""


@dataclass(frozen=True)
class OptimizationProblem:
    """A solver-independent constrained portfolio problem."""

    objective: Callable[[np.ndarray], float]
    initial_weights: Sequence[float]
    lower_bounds: Sequence[float]
    upper_bounds: Sequence[float]
    gradient: Callable[[np.ndarray], np.ndarray] | None = None
    linear_constraints: Sequence[LinearConstraintSpec] = field(default_factory=tuple)
    nonlinear_constraints: Sequence[NonlinearConstraintSpec] = field(default_factory=tuple)
    investment_level: float = 1.0
    name: str = "portfolio_optimization"


@dataclass(frozen=True)
class OptimizationResult:
    """Verified solver result with diagnostics suitable for reports."""

    weights: np.ndarray
    solver_name: str
    objective_value: float
    iterations: int
    message: str
    maximum_constraint_violation: float

    def as_dict(self) -> dict[str, float | int | str]:
        """Return serialisable diagnostics without embedding portfolio weights."""

        return {
            "solver_name": self.solver_name,
            "objective_value": self.objective_value,
            "iterations": self.iterations,
            "message": self.message,
            "maximum_constraint_violation": self.maximum_constraint_violation,
        }


__all__ = ["OptimizationError", "OptimizationProblem", "OptimizationResult"]
