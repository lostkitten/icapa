"""Additive declarative models and diagnostics for portfolio optimization.

The existing callable :class:`OptimizationProblem` remains the lowest-level
escape hatch. These contracts add model introspection, backend capability
matching, and per-constraint diagnostics without changing that API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .constraints import LinearConstraintSpec, NonlinearConstraintSpec
from .objectives import minimum_variance_objective, squared_distance_objective
from .problem import OptimizationProblem, OptimizationResult
from .solvers import PortfolioSolver


class SolverCapability(str, Enum):
    """Capabilities advertised by a model-aware optimization backend."""

    LINEAR_OBJECTIVE = "linear_objective"
    QUADRATIC_OBJECTIVE = "quadratic_objective"
    NONLINEAR_OBJECTIVE = "nonlinear_objective"
    LINEAR_CONSTRAINTS = "linear_constraints"
    NONLINEAR_CONSTRAINTS = "nonlinear_constraints"
    MIXED_INTEGER = "mixed_integer"
    SPARSE = "sparse"
    WARM_START = "warm_start"
    DUAL_VALUES = "dual_values"
    INFEASIBILITY_CERTIFICATE = "infeasibility_certificate"


class ConstraintStatus(str, Enum):
    """Post-solve status of one evaluated constraint."""

    SATISFIED = "satisfied"
    BINDING = "binding"
    VIOLATED = "violated"
    INVALID = "invalid"


@runtime_checkable
class ObjectiveSpec(Protocol):
    """Declarative objective compiled into the existing callable API."""

    @property
    def required_capabilities(self) -> frozenset[SolverCapability]: ...

    def build(
        self,
    ) -> tuple[
        Callable[[np.ndarray], float],
        Callable[[np.ndarray], np.ndarray] | None,
    ]: ...


@dataclass(frozen=True)
class LinearObjectiveSpec:
    """Minimize a linear function of portfolio weights."""

    coefficients: Sequence[float]

    @property
    def required_capabilities(self) -> frozenset[SolverCapability]:
        return frozenset({SolverCapability.LINEAR_OBJECTIVE})

    def build(self):
        coefficients = np.asarray(self.coefficients, dtype=float)
        if coefficients.ndim != 1 or not np.isfinite(coefficients).all():
            raise ValueError("linear objective coefficients must be a finite vector")

        def objective(weights: np.ndarray) -> float:
            return float(coefficients @ weights)

        def gradient(weights: np.ndarray) -> np.ndarray:
            return coefficients.copy()

        return objective, gradient

    def quadratic_terms(self, size: int) -> tuple[np.ndarray, np.ndarray, float]:
        coefficients = np.asarray(self.coefficients, dtype=float)
        if coefficients.shape != (size,) or not np.isfinite(coefficients).all():
            raise ValueError("linear objective shape must match the weight vector")
        return np.zeros((size, size)), coefficients.copy(), 0.0


@dataclass(frozen=True)
class SquaredDistanceObjectiveSpec:
    """Minimize squared Euclidean distance from explicit target weights."""

    target_weights: Sequence[float]

    @property
    def required_capabilities(self) -> frozenset[SolverCapability]:
        return frozenset({SolverCapability.QUADRATIC_OBJECTIVE})

    def build(self):
        return squared_distance_objective(self.target_weights)

    def quadratic_terms(self, size: int) -> tuple[np.ndarray, np.ndarray, float]:
        target = np.asarray(self.target_weights, dtype=float)
        if target.shape != (size,) or not np.isfinite(target).all():
            raise ValueError("target_weights shape must match the weight vector")
        return 2.0 * np.eye(size), -2.0 * target, float(target @ target)


@dataclass(frozen=True)
class MinimumVarianceObjectiveSpec:
    """Minimize total variance using an explicit covariance matrix."""

    covariance: np.ndarray

    @property
    def required_capabilities(self) -> frozenset[SolverCapability]:
        return frozenset({SolverCapability.QUADRATIC_OBJECTIVE})

    def build(self):
        return minimum_variance_objective(self.covariance)

    def quadratic_terms(self, size: int) -> tuple[np.ndarray, np.ndarray, float]:
        covariance = np.asarray(self.covariance, dtype=float)
        if covariance.shape != (size, size) or not np.isfinite(covariance).all():
            raise ValueError("covariance shape must match the weight vector")
        covariance = (covariance + covariance.T) / 2.0
        return 2.0 * covariance, np.zeros(size), 0.0


@dataclass(frozen=True)
class CallableObjectiveSpec:
    """Escape hatch for a custom objective and optional gradient."""

    function: Callable[[np.ndarray], float]
    gradient: Callable[[np.ndarray], np.ndarray] | None = None
    capabilities: frozenset[SolverCapability] = frozenset(
        {SolverCapability.NONLINEAR_OBJECTIVE}
    )

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("objective function must be callable")
        if self.gradient is not None and not callable(self.gradient):
            raise TypeError("objective gradient must be callable or None")
        object.__setattr__(
            self,
            "capabilities",
            frozenset(SolverCapability(item) for item in self.capabilities),
        )

    @property
    def required_capabilities(self) -> frozenset[SolverCapability]:
        return self.capabilities

    def build(self):
        return self.function, self.gradient


@dataclass(frozen=True)
class WeightVariableSpec:
    """Ordered instrument weight vector, bounds, and warm start."""

    instrument_ids: Sequence[Any]
    initial_weights: Sequence[float]
    lower_bounds: Sequence[float]
    upper_bounds: Sequence[float]
    investment_level: float = 1.0

    def __post_init__(self) -> None:
        identifiers = tuple(self.instrument_ids)
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("instrument_ids must be non-empty and unique")
        initial = _immutable_vector(self.initial_weights, "initial_weights")
        lower = _immutable_vector(self.lower_bounds, "lower_bounds")
        upper = _immutable_vector(self.upper_bounds, "upper_bounds")
        if not (len(identifiers) == len(initial) == len(lower) == len(upper)):
            raise ValueError("instrument IDs, weights, and bounds must have equal length")
        if np.any(lower > upper):
            raise ValueError("lower bounds must not exceed upper bounds")
        if not np.isfinite(self.investment_level):
            raise ValueError("investment_level must be finite")
        if lower.sum() > self.investment_level or upper.sum() < self.investment_level:
            raise ValueError("weight bounds cannot satisfy the investment level")
        object.__setattr__(self, "instrument_ids", identifiers)
        object.__setattr__(self, "initial_weights", initial)
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)
        object.__setattr__(self, "investment_level", float(self.investment_level))


@dataclass(frozen=True)
class OptimizationModelSpec:
    """Inspectable portfolio model compiled for a compatible backend."""

    name: str
    variables: WeightVariableSpec
    objective: ObjectiveSpec
    linear_constraints: Sequence[LinearConstraintSpec] = ()
    nonlinear_constraints: Sequence[NonlinearConstraintSpec] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("optimization model name must be a non-empty string")
        if not isinstance(self.objective, ObjectiveSpec):
            raise TypeError("objective must implement ObjectiveSpec")
        object.__setattr__(self, "linear_constraints", tuple(self.linear_constraints))
        object.__setattr__(
            self,
            "nonlinear_constraints",
            tuple(self.nonlinear_constraints),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def required_capabilities(self) -> frozenset[SolverCapability]:
        capabilities = {
            *self.objective.required_capabilities,
            SolverCapability.LINEAR_CONSTRAINTS,
        }
        if self.nonlinear_constraints:
            capabilities.add(SolverCapability.NONLINEAR_CONSTRAINTS)
        return frozenset(capabilities)

    def compile(self) -> OptimizationProblem:
        """Compile into the existing solver-independent callable problem."""

        objective, gradient = self.objective.build()
        return OptimizationProblem(
            name=self.name,
            objective=objective,
            gradient=gradient,
            initial_weights=self.variables.initial_weights,
            lower_bounds=self.variables.lower_bounds,
            upper_bounds=self.variables.upper_bounds,
            linear_constraints=self.linear_constraints,
            nonlinear_constraints=self.nonlinear_constraints,
            investment_level=self.variables.investment_level,
        )


@runtime_checkable
class ModelSolver(Protocol):
    """Backend that advertises capabilities before solving a model."""

    @property
    def backend_name(self) -> str: ...

    @property
    def capabilities(self) -> frozenset[SolverCapability]: ...

    def solve_model(self, model: OptimizationModelSpec) -> OptimizationResult: ...


@dataclass(frozen=True)
class PortfolioSolverModelAdapter:
    """Expose an existing ``PortfolioSolver`` as a model-aware backend."""

    solver: PortfolioSolver
    name: str = "portfolio_solver"
    supported_capabilities: frozenset[SolverCapability] = frozenset(
        {
            SolverCapability.LINEAR_OBJECTIVE,
            SolverCapability.QUADRATIC_OBJECTIVE,
            SolverCapability.NONLINEAR_OBJECTIVE,
            SolverCapability.LINEAR_CONSTRAINTS,
            SolverCapability.NONLINEAR_CONSTRAINTS,
            SolverCapability.WARM_START,
        }
    )

    def __post_init__(self) -> None:
        if not isinstance(self.solver, PortfolioSolver):
            raise TypeError("solver must implement PortfolioSolver")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("solver backend name must be a non-empty string")
        object.__setattr__(
            self,
            "supported_capabilities",
            frozenset(
                SolverCapability(item) for item in self.supported_capabilities
            ),
        )

    @property
    def backend_name(self) -> str:
        return self.name

    @property
    def capabilities(self) -> frozenset[SolverCapability]:
        return self.supported_capabilities

    def solve_model(self, model: OptimizationModelSpec) -> OptimizationResult:
        missing = model.required_capabilities - self.capabilities
        if missing:
            raise ValueError(
                "solver does not support required model capabilities: "
                + ", ".join(sorted(item.value for item in missing))
            )
        return self.solver.solve(model.compile())


@dataclass(frozen=True)
class ConstraintEvaluation:
    """Achieved value, slack, and status for one constraint."""

    name: str
    value: float
    lower: float
    upper: float
    lower_slack: float
    upper_slack: float
    violation: float
    status: ConstraintStatus
    dual_value: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ConstraintStatus(self.status))


@dataclass(frozen=True)
class ConstraintReport:
    """Constraint-level verification suitable for research review and reports."""

    evaluations: tuple[ConstraintEvaluation, ...]
    tolerance: float

    def __post_init__(self) -> None:
        if self.tolerance <= 0 or not np.isfinite(self.tolerance):
            raise ValueError("constraint-report tolerance must be finite and positive")
        object.__setattr__(self, "evaluations", tuple(self.evaluations))

    @property
    def feasible(self) -> bool:
        return all(
            item.status not in {ConstraintStatus.VIOLATED, ConstraintStatus.INVALID}
            for item in self.evaluations
        )

    @property
    def maximum_violation(self) -> float:
        if not self.evaluations:
            return 0.0
        return max(float(item.violation) for item in self.evaluations)

    def as_records(self) -> list[dict[str, Any]]:
        return [
            {
                "name": item.name,
                "value": item.value,
                "lower": item.lower,
                "upper": item.upper,
                "lower_slack": item.lower_slack,
                "upper_slack": item.upper_slack,
                "violation": item.violation,
                "status": item.status.value,
                "dual_value": item.dual_value,
            }
            for item in self.evaluations
        ]


def evaluate_problem_constraints(
    problem: OptimizationProblem,
    weights: Sequence[float],
    *,
    instrument_ids: Sequence[Any] | None = None,
    tolerance: float = 1e-7,
) -> ConstraintReport:
    """Evaluate every bound and explicit constraint in an existing problem."""

    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    values = np.asarray(weights, dtype=float)
    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)
    if values.ndim != 1 or values.shape != lower.shape or values.shape != upper.shape:
        raise ValueError("weights and problem bounds must have the same vector shape")
    names = (
        tuple(range(len(values)))
        if instrument_ids is None
        else tuple(instrument_ids)
    )
    if len(names) != len(values):
        raise ValueError("instrument_ids length must match weights")

    evaluations = [
        _evaluate_scalar(
            name="investment_level",
            value=float(values.sum()),
            lower=float(problem.investment_level),
            upper=float(problem.investment_level),
            tolerance=tolerance,
        )
    ]
    evaluations.extend(
        _evaluate_scalar(
            name=f"weight:{identifier}",
            value=float(value),
            lower=float(lower_bound),
            upper=float(upper_bound),
            tolerance=tolerance,
        )
        for identifier, value, lower_bound, upper_bound in zip(
            names,
            values,
            lower,
            upper,
        )
    )
    for spec in problem.linear_constraints:
        coefficients = np.asarray(spec.coefficients, dtype=float)
        achieved = (
            float(np.dot(coefficients, values))
            if coefficients.shape == values.shape and np.isfinite(coefficients).all()
            else float("nan")
        )
        evaluations.append(
            _evaluate_scalar(
                name=spec.name,
                value=achieved,
                lower=float(spec.lower),
                upper=float(spec.upper),
                tolerance=tolerance,
            )
        )
    for spec in problem.nonlinear_constraints:
        try:
            achieved = float(spec.function(values))
        except Exception:
            achieved = float("nan")
        evaluations.append(
            _evaluate_scalar(
                name=spec.name,
                value=achieved,
                lower=float(spec.lower),
                upper=float(spec.upper),
                tolerance=tolerance,
            )
        )
    return ConstraintReport(tuple(evaluations), float(tolerance))


def _evaluate_scalar(
    *,
    name: str,
    value: float,
    lower: float,
    upper: float,
    tolerance: float,
) -> ConstraintEvaluation:
    if (
        np.isnan(value)
        or np.isnan(lower)
        or np.isnan(upper)
        or lower > upper
    ):
        return ConstraintEvaluation(
            name=name,
            value=value,
            lower=lower,
            upper=upper,
            lower_slack=float("nan"),
            upper_slack=float("nan"),
            violation=float("inf"),
            status=ConstraintStatus.INVALID,
        )
    lower_slack = value - lower
    upper_slack = upper - value
    violation = max(lower - value, value - upper, 0.0)
    if violation > tolerance:
        status = ConstraintStatus.VIOLATED
    elif min(lower_slack, upper_slack) <= tolerance:
        status = ConstraintStatus.BINDING
    else:
        status = ConstraintStatus.SATISFIED
    return ConstraintEvaluation(
        name=name,
        value=value,
        lower=lower,
        upper=upper,
        lower_slack=lower_slack,
        upper_slack=upper_slack,
        violation=violation,
        status=status,
    )


def _immutable_vector(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    result.setflags(write=False)
    return result


__all__ = [
    "CallableObjectiveSpec",
    "ConstraintEvaluation",
    "ConstraintReport",
    "ConstraintStatus",
    "LinearObjectiveSpec",
    "MinimumVarianceObjectiveSpec",
    "ModelSolver",
    "ObjectiveSpec",
    "OptimizationModelSpec",
    "PortfolioSolverModelAdapter",
    "SolverCapability",
    "SquaredDistanceObjectiveSpec",
    "WeightVariableSpec",
    "evaluate_problem_constraints",
]
