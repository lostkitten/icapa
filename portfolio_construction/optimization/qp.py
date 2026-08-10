"""Sparse QP backend, explicit solver routing, and phase-one feasibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, eye, vstack

from .models import (
    ConstraintReport,
    ModelSolver,
    OptimizationModelSpec,
    SolverCapability,
    evaluate_problem_constraints,
)
from .problem import OptimizationError, OptimizationResult


class OptionalSolverDependencyError(ImportError):
    """Raised when an explicitly selected optional solver is not installed."""


class FeasibilityStatus(str, Enum):
    """Outcome of a phase-one feasibility calculation."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


@dataclass(frozen=True)
class ModelSolveResult:
    """Solver result plus constraint-level post-solve verification."""

    result: OptimizationResult
    constraints: ConstraintReport
    backend_name: str


@dataclass(frozen=True)
class FeasibilityReport:
    """Phase-one feasibility result and minimum-relaxation diagnostics."""

    status: FeasibilityStatus
    message: str
    minimum_total_violation: float
    candidate_weights: np.ndarray | None = None
    constraints: ConstraintReport | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", FeasibilityStatus(self.status))
        if self.minimum_total_violation < 0:
            raise ValueError("minimum_total_violation must be non-negative")
        if self.candidate_weights is not None:
            candidate = np.asarray(self.candidate_weights, dtype=float).copy()
            if candidate.ndim != 1 or not np.isfinite(candidate).all():
                raise ValueError("candidate_weights must be a finite vector")
            candidate.setflags(write=False)
            object.__setattr__(self, "candidate_weights", candidate)

    @property
    def feasible(self) -> bool:
        return self.status is FeasibilityStatus.FEASIBLE


@dataclass(frozen=True)
class OSQPBackend:
    """Optional sparse backend for convex linear and quadratic portfolio models."""

    tolerance: float = 1e-8
    max_iterations: int = 20_000
    polish: bool = True
    name: str = "osqp"

    def __post_init__(self) -> None:
        if self.tolerance <= 0 or not np.isfinite(self.tolerance):
            raise ValueError("OSQP tolerance must be finite and positive")
        if self.max_iterations <= 0:
            raise ValueError("OSQP max_iterations must be positive")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("OSQP backend name must be a non-empty string")

    @property
    def backend_name(self) -> str:
        return self.name

    @property
    def capabilities(self) -> frozenset[SolverCapability]:
        return frozenset(
            {
                SolverCapability.LINEAR_OBJECTIVE,
                SolverCapability.QUADRATIC_OBJECTIVE,
                SolverCapability.LINEAR_CONSTRAINTS,
                SolverCapability.SPARSE,
                SolverCapability.WARM_START,
                SolverCapability.DUAL_VALUES,
                SolverCapability.INFEASIBILITY_CERTIFICATE,
            }
        )

    def solve_model(self, model: OptimizationModelSpec) -> OptimizationResult:
        return self.solve_model_with_diagnostics(model).result

    def solve_model_with_diagnostics(
        self,
        model: OptimizationModelSpec,
    ) -> ModelSolveResult:
        missing = model.required_capabilities - self.capabilities
        if missing:
            raise ValueError(
                "OSQP does not support required model capabilities: "
                + ", ".join(sorted(item.value for item in missing))
            )
        if model.nonlinear_constraints:
            raise ValueError("OSQP does not support nonlinear constraints")
        quadratic_terms = getattr(model.objective, "quadratic_terms", None)
        if not callable(quadratic_terms):
            raise ValueError(
                "OSQP requires an objective exposing quadratic_terms(size)"
            )
        try:
            import osqp
        except ImportError as exc:
            raise OptionalSolverDependencyError(
                "OSQPBackend requires the optional 'osqp' dependency"
            ) from exc

        variables = model.variables
        size = len(variables.instrument_ids)
        quadratic, linear, _ = quadratic_terms(size)
        quadratic = np.asarray(quadratic, dtype=float)
        linear = np.asarray(linear, dtype=float)
        if (
            quadratic.shape != (size, size)
            or linear.shape != (size,)
            or not np.isfinite(quadratic).all()
            or not np.isfinite(linear).all()
        ):
            raise ValueError("objective returned invalid quadratic terms")
        quadratic = (quadratic + quadratic.T) / 2.0

        rows = [eye(size, format="csc"), csc_matrix(np.ones((1, size)))]
        lower = [
            np.asarray(variables.lower_bounds, dtype=float),
            np.asarray([variables.investment_level], dtype=float),
        ]
        upper = [
            np.asarray(variables.upper_bounds, dtype=float),
            np.asarray([variables.investment_level], dtype=float),
        ]
        for constraint in model.linear_constraints:
            coefficients = np.asarray(constraint.coefficients, dtype=float)
            if coefficients.shape != (size,) or not np.isfinite(coefficients).all():
                raise ValueError(
                    f"invalid coefficients for constraint {constraint.name!r}"
                )
            if constraint.lower > constraint.upper:
                raise ValueError(
                    f"invalid bounds for constraint {constraint.name!r}"
                )
            rows.append(csc_matrix(coefficients.reshape(1, -1)))
            lower.append(np.asarray([constraint.lower], dtype=float))
            upper.append(np.asarray([constraint.upper], dtype=float))

        solver = osqp.OSQP()
        solver.setup(
            P=csc_matrix(quadratic),
            q=linear,
            A=vstack(rows, format="csc"),
            l=np.concatenate(lower),
            u=np.concatenate(upper),
            eps_abs=self.tolerance,
            eps_rel=self.tolerance,
            max_iter=self.max_iterations,
            polishing=self.polish,
            verbose=False,
        )
        solver.warm_start(x=np.asarray(variables.initial_weights, dtype=float))
        try:
            raw = solver.solve(raise_error=False)
        except TypeError:
            raw = solver.solve()
        status = str(raw.info.status)
        weights = (
            np.asarray(raw.x, dtype=float)
            if raw.x is not None
            else np.full(size, np.nan)
        )
        problem = model.compile()
        report = evaluate_problem_constraints(
            problem,
            weights,
            instrument_ids=variables.instrument_ids,
            tolerance=max(1e-7, self.tolerance * 100),
        )
        if (
            not status.lower().startswith("solved")
            or not np.isfinite(weights).all()
            or not report.feasible
        ):
            raise OptimizationError(
                f"{model.name} did not produce a verified OSQP solution: {status}; "
                f"maximum constraint violation={report.maximum_violation:.3g}"
            )
        result = OptimizationResult(
            weights=weights,
            solver_name=self.backend_name,
            objective_value=float(problem.objective(weights)),
            iterations=int(getattr(raw.info, "iter", 0)),
            message=status,
            maximum_constraint_violation=report.maximum_violation,
        )
        return ModelSolveResult(
            result=result,
            constraints=report,
            backend_name=self.backend_name,
        )


class SolverRouter:
    """Select one explicitly named compatible backend without silent fallback."""

    def __init__(
        self,
        backends: Sequence[ModelSolver],
        *,
        default_backend: str | None = None,
    ) -> None:
        values = tuple(backends)
        names = [backend.backend_name for backend in values]
        if not values or len(set(names)) != len(names):
            raise ValueError("solver backends must be non-empty with unique names")
        self._backends = {backend.backend_name: backend for backend in values}
        if default_backend is not None and default_backend not in self._backends:
            raise KeyError(f"default solver backend is not registered: {default_backend}")
        self.default_backend = default_backend

    def compatible_backends(self, model: OptimizationModelSpec) -> tuple[str, ...]:
        """Return registered backends satisfying the model's declared capabilities."""

        return tuple(
            name
            for name, backend in self._backends.items()
            if model.required_capabilities.issubset(backend.capabilities)
        )

    def solve_model(
        self,
        model: OptimizationModelSpec,
        *,
        backend_name: str | None = None,
    ) -> OptimizationResult:
        """Solve with one explicit/default backend and never try another on failure."""

        selected = backend_name or self.default_backend
        if selected is None:
            raise ValueError(
                "backend_name is required when the solver router has no default"
            )
        try:
            backend = self._backends[selected]
        except KeyError as exc:
            raise KeyError(f"solver backend is not registered: {selected}") from exc
        missing = model.required_capabilities - backend.capabilities
        if missing:
            raise ValueError(
                f"solver backend {selected!r} is incompatible with the model: "
                + ", ".join(sorted(item.value for item in missing))
            )
        return backend.solve_model(model)

    def solve_model_with_diagnostics(
        self,
        model: OptimizationModelSpec,
        *,
        backend_name: str | None = None,
        tolerance: float = 1e-7,
    ) -> ModelSolveResult:
        selected = backend_name or self.default_backend
        if selected is None:
            raise ValueError(
                "backend_name is required when the solver router has no default"
            )
        try:
            backend = self._backends[selected]
        except KeyError as exc:
            raise KeyError(f"solver backend is not registered: {selected}") from exc
        missing = model.required_capabilities - backend.capabilities
        if missing:
            raise ValueError(
                f"solver backend {selected!r} is incompatible with the model: "
                + ", ".join(sorted(item.value for item in missing))
            )
        extended = getattr(backend, "solve_model_with_diagnostics", None)
        if callable(extended):
            return extended(model)
        result = backend.solve_model(model)
        report = evaluate_problem_constraints(
            model.compile(),
            result.weights,
            instrument_ids=model.variables.instrument_ids,
            tolerance=tolerance,
        )
        return ModelSolveResult(result, report, selected)


def check_linear_feasibility(
    model: OptimizationModelSpec,
    *,
    tolerance: float = 1e-8,
) -> FeasibilityReport:
    """Run a phase-one LP and report the minimum aggregate relaxation.

    Nonlinear constraints require a backend-specific feasibility routine and are
    therefore reported as unsupported rather than silently ignored.
    """

    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("feasibility tolerance must be finite and positive")
    if model.nonlinear_constraints:
        return FeasibilityReport(
            status=FeasibilityStatus.UNSUPPORTED,
            message="phase-one LP does not support nonlinear constraints",
            minimum_total_violation=float("inf"),
        )
    variables = model.variables
    size = len(variables.instrument_ids)
    inequality_rows, inequality_bounds = _linear_inequalities(model)
    bounds = list(zip(variables.lower_bounds, variables.upper_bounds))
    exact = linprog(
        np.zeros(size),
        A_ub=(
            None
            if not inequality_rows
            else np.asarray(inequality_rows, dtype=float)
        ),
        b_ub=(
            None
            if not inequality_bounds
            else np.asarray(inequality_bounds, dtype=float)
        ),
        A_eq=np.ones((1, size)),
        b_eq=np.asarray([variables.investment_level]),
        bounds=bounds,
        method="highs",
    )
    if exact.success:
        candidate = np.asarray(exact.x, dtype=float)
        report = evaluate_problem_constraints(
            model.compile(),
            candidate,
            instrument_ids=variables.instrument_ids,
            tolerance=tolerance,
        )
        return FeasibilityReport(
            status=(
                FeasibilityStatus.FEASIBLE
                if report.feasible
                else FeasibilityStatus.ERROR
            ),
            message=str(exact.message),
            minimum_total_violation=report.maximum_violation,
            candidate_weights=candidate,
            constraints=report,
        )

    # Relax every linear inequality and both directions of the investment level.
    rows = [*inequality_rows, np.ones(size), -np.ones(size)]
    right_hand_sides = [
        *inequality_bounds,
        variables.investment_level,
        -variables.investment_level,
    ]
    slack_count = len(rows)
    augmented = np.zeros((slack_count, size + slack_count))
    for position, row in enumerate(rows):
        augmented[position, :size] = row
        augmented[position, size + position] = -1.0
    objective = np.concatenate([np.zeros(size), np.ones(slack_count)])
    relaxed = linprog(
        objective,
        A_ub=augmented,
        b_ub=np.asarray(right_hand_sides, dtype=float),
        bounds=[*bounds, *((0.0, None) for _ in range(slack_count))],
        method="highs",
    )
    if not relaxed.success:
        return FeasibilityReport(
            status=FeasibilityStatus.ERROR,
            message=str(relaxed.message),
            minimum_total_violation=float("inf"),
        )
    candidate = np.asarray(relaxed.x[:size], dtype=float)
    report = evaluate_problem_constraints(
        model.compile(),
        candidate,
        instrument_ids=variables.instrument_ids,
        tolerance=tolerance,
    )
    minimum_relaxation = float(relaxed.fun)
    return FeasibilityReport(
        status=(
            FeasibilityStatus.FEASIBLE
            if minimum_relaxation <= tolerance and report.feasible
            else FeasibilityStatus.INFEASIBLE
        ),
        message=(
            "linear constraints require aggregate relaxation "
            f"{minimum_relaxation:.6g}"
        ),
        minimum_total_violation=minimum_relaxation,
        candidate_weights=candidate,
        constraints=report,
    )


def _linear_inequalities(
    model: OptimizationModelSpec,
) -> tuple[list[np.ndarray], list[float]]:
    size = len(model.variables.instrument_ids)
    rows: list[np.ndarray] = []
    right_hand_sides: list[float] = []
    for constraint in model.linear_constraints:
        coefficients = np.asarray(constraint.coefficients, dtype=float)
        if coefficients.shape != (size,) or not np.isfinite(coefficients).all():
            raise ValueError(
                f"invalid coefficients for constraint {constraint.name!r}"
            )
        if constraint.lower > constraint.upper:
            raise ValueError(f"invalid bounds for constraint {constraint.name!r}")
        if np.isfinite(constraint.upper):
            rows.append(coefficients)
            right_hand_sides.append(float(constraint.upper))
        if np.isfinite(constraint.lower):
            rows.append(-coefficients)
            right_hand_sides.append(float(-constraint.lower))
    return rows, right_hand_sides


__all__ = [
    "FeasibilityReport",
    "FeasibilityStatus",
    "ModelSolveResult",
    "OSQPBackend",
    "OptionalSolverDependencyError",
    "SolverRouter",
    "check_linear_feasibility",
]
