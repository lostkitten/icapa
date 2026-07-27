"""Portfolio-solver interface and the default SciPy implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize

from .constraints import LinearConstraintSpec, NonlinearConstraintSpec
from .problem import OptimizationError, OptimizationProblem, OptimizationResult


@runtime_checkable
class PortfolioSolver(Protocol):
    """Interface implemented by constrained portfolio solvers."""

    def solve(self, problem: OptimizationProblem) -> OptimizationResult: ...


@dataclass(frozen=True)
class ScipySLSQPSolver:
    """Solve and independently verify a constrained portfolio problem."""

    tolerance: float = 1e-9
    max_iterations: int = 1_000

    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        if self.tolerance <= 0 or self.max_iterations <= 0:
            raise ValueError("tolerance and max_iterations must be positive")
        initial = np.asarray(problem.initial_weights, dtype=float)
        lower = np.asarray(problem.lower_bounds, dtype=float)
        upper = np.asarray(problem.upper_bounds, dtype=float)
        _validate_dimensions(initial, lower, upper)
        if lower.sum() > problem.investment_level + self.tolerance:
            raise OptimizationError("lower bounds exceed the investment level")
        if upper.sum() < problem.investment_level - self.tolerance:
            raise OptimizationError("upper bounds cannot reach the investment level")

        linear_specs = tuple(problem.linear_constraints)
        nonlinear_specs = tuple(problem.nonlinear_constraints)
        start = _bounded_simplex_start(
            initial,
            lower,
            upper,
            problem.investment_level,
            self.tolerance,
        )
        constraints: list[LinearConstraint | NonlinearConstraint] = [
            LinearConstraint(
                np.ones(len(start)),
                problem.investment_level,
                problem.investment_level,
            )
        ]
        for spec in linear_specs:
            coefficients = np.asarray(spec.coefficients, dtype=float)
            if coefficients.shape != start.shape or not np.isfinite(coefficients).all():
                raise ValueError(f"invalid coefficients for constraint {spec.name}")
            if spec.lower > spec.upper:
                raise OptimizationError(f"invalid bounds for constraint {spec.name}")
            constraints.append(
                LinearConstraint(coefficients, spec.lower, spec.upper)
            )
        for spec in nonlinear_specs:
            if spec.lower > spec.upper:
                raise OptimizationError(f"invalid bounds for constraint {spec.name}")
            constraints.append(
                NonlinearConstraint(
                    spec.function,
                    spec.lower,
                    spec.upper,
                    jac=spec.gradient if spec.gradient is not None else "2-point",
                )
            )

        raw = minimize(
            problem.objective,
            start,
            jac=problem.gradient,
            method="SLSQP",
            bounds=Bounds(lower, upper),
            constraints=constraints,
            options={
                "ftol": self.tolerance,
                "maxiter": self.max_iterations,
                "disp": False,
            },
        )
        weights = np.asarray(raw.x, dtype=float)
        violation = _maximum_violation(
            weights,
            lower,
            upper,
            linear_specs,
            nonlinear_specs,
            investment_level=problem.investment_level,
        )
        verification_tolerance = max(1e-7, self.tolerance * 100)
        if (
            not raw.success
            or not np.isfinite(weights).all()
            or violation > verification_tolerance
        ):
            detail = str(raw.message) if raw.message else "unknown solver failure"
            raise OptimizationError(
                f"{problem.name} did not produce a verified solution: {detail}; "
                f"maximum constraint violation={violation:.3g}"
            )
        return OptimizationResult(
            weights=weights,
            solver_name="scipy_slsqp",
            objective_value=float(problem.objective(weights)),
            iterations=int(getattr(raw, "nit", 0)),
            message=str(raw.message),
            maximum_constraint_violation=violation,
        )


def solve_slsqp(
    objective,
    initial_weights,
    lower_bounds,
    upper_bounds,
    *,
    linear_constraints=(),
    nonlinear_constraints=(),
    investment_level: float = 1.0,
    gradient=None,
    tolerance: float = 1e-9,
    max_iterations: int = 1_000,
) -> np.ndarray:
    """Compatibility wrapper returning only verified portfolio weights."""

    result = ScipySLSQPSolver(
        tolerance=tolerance,
        max_iterations=max_iterations,
    ).solve(
        OptimizationProblem(
            objective=objective,
            initial_weights=initial_weights,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            gradient=gradient,
            linear_constraints=tuple(linear_constraints),
            nonlinear_constraints=tuple(nonlinear_constraints),
            investment_level=investment_level,
        )
    )
    return result.weights


def _validate_dimensions(
    initial: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    if not (initial.ndim == lower.ndim == upper.ndim == 1):
        raise ValueError("weights and bounds must be one-dimensional")
    if len(initial) == 0 or not (len(initial) == len(lower) == len(upper)):
        raise ValueError("weights and bounds must have the same non-zero length")
    if (
        not np.isfinite(initial).all()
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
    ):
        raise ValueError("weights and bounds must be finite")
    if np.any(lower > upper):
        raise OptimizationError("one or more lower bounds exceed upper bounds")


def _bounded_simplex_start(
    initial: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    investment_level: float,
    tolerance: float,
) -> np.ndarray:
    weights = np.clip(initial, lower, upper)
    for _ in range(100):
        gap = investment_level - float(weights.sum())
        if abs(gap) <= tolerance:
            return weights
        slack = upper - weights if gap > 0 else weights - lower
        available = float(slack.sum())
        if available <= tolerance:
            break
        weights += np.sign(gap) * slack * min(1.0, abs(gap) / available)
        weights = np.clip(weights, lower, upper)
    if abs(float(weights.sum()) - investment_level) > max(1e-8, tolerance * 10):
        raise OptimizationError("could not construct an initial portfolio within the bounds")
    return weights


def _maximum_violation(
    weights: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    linear_specs: tuple[LinearConstraintSpec, ...],
    nonlinear_specs: tuple[NonlinearConstraintSpec, ...],
    *,
    investment_level: float,
) -> float:
    violations = [
        abs(float(weights.sum()) - investment_level),
        float(np.maximum(lower - weights, 0.0).max(initial=0.0)),
        float(np.maximum(weights - upper, 0.0).max(initial=0.0)),
    ]
    for spec in linear_specs:
        exposure = float(np.dot(spec.coefficients, weights))
        violations.extend(
            [
                max(spec.lower - exposure, 0.0),
                max(exposure - spec.upper, 0.0),
            ]
        )
    for spec in nonlinear_specs:
        value = float(spec.function(weights))
        violations.extend(
            [
                max(spec.lower - value, 0.0),
                max(value - spec.upper, 0.0),
            ]
        )
    return max(violations)


__all__ = ["PortfolioSolver", "ScipySLSQPSolver", "solve_slsqp"]
