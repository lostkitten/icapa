"""Minimum-relative-entropy exposure-targeting portfolio engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

from icapa.portfolio_construction.optimization import (
    EGMUConstrainedElasticSolver,
    EGMUNewtonSolver,
    EGMUProjectionSolver,
    LinearConstraintSpec,
    OptimizationError,
    group_constraint_specs,
    relative_entropy,
    weight_bounds,
)


class TargetDirection(str, Enum):
    """Interpret an exposure target as a point or one-sided bound."""

    EQUAL = "equal"
    AT_LEAST = "at_least"
    AT_MOST = "at_most"


class EntropyExposureMode(str, Enum):
    """Select hard targeting, elastic targeting, or an explicit fallback."""

    HARD = "hard"
    ELASTIC = "elastic"
    HARD_THEN_ELASTIC = "hard_then_elastic"


@dataclass(frozen=True)
class ExposureTarget:
    """A target or bound on one weighted instrument-level exposure."""

    field: str
    target: float
    direction: TargetDirection = TargetDirection.EQUAL
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise ValueError("target field names cannot be empty")
        target = float(self.target)
        tolerance = float(self.tolerance)
        if not np.isfinite(target):
            raise ValueError(f"target value must be finite: {self.field}")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError(
                f"target tolerance must be finite and non-negative: {self.field}"
            )
        object.__setattr__(self, "field", self.field.strip())
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(
            self,
            "direction",
            TargetDirection(self.direction),
        )


@dataclass
class EntropyExposureEngine:
    """Match requested exposures while minimizing KL divergence from a benchmark.

    Exact point targets use the low-dimensional EGMU Newton solver when no
    additional portfolio constraints are active.  Intervals, one-sided
    targets, group limits, and instrument bounds use EGMU's multiplicative
    Bregman--Dykstra projections.  Elastic targeting is explicit because it
    may intentionally leave target residuals.
    """

    targets: Sequence[ExposureTarget]
    mode: EntropyExposureMode = EntropyExposureMode.HARD
    elastic_penalty: float = 100.0
    input_weight_column: str = "benchmark_weight"
    output_weight_column: str = "index_weight"
    minimum_weight: float = 0.0
    maximum_weight: float = 1.0
    capacity_multiple: float | None = None
    group_tolerances: Mapping[str, float] = field(default_factory=dict)
    max_iterations: int = 1_000
    tolerance: float = 1.0e-8
    newton_ridge: float = 1.0e-10
    newton_solver: EGMUNewtonSolver | None = field(default=None, repr=False)
    elastic_solver: EGMUConstrainedElasticSolver | None = field(
        default=None,
        repr=False,
    )
    projection_solver: EGMUProjectionSolver | None = field(
        default=None,
        repr=False,
    )

    def execute(self, data_context):
        targets = tuple(self.targets)
        if not targets:
            raise ValueError("at least one exposure target is required")
        if any(not isinstance(target, ExposureTarget) for target in targets):
            raise TypeError("targets must contain ExposureTarget values")
        target_fields = [target.field for target in targets]
        if len(set(target_fields)) != len(target_fields):
            raise ValueError("each target field may appear only once")
        mode = EntropyExposureMode(self.mode)
        self._validate_configuration()

        columns = list(
            dict.fromkeys(
                [
                    self.input_weight_column,
                    *target_fields,
                    *self.group_tolerances,
                ]
            )
        )
        frame = data_context.get_dataframe(
            columns,
            include_excluded_instruments=False,
        )
        if frame.empty or not frame.index.is_unique:
            raise ValueError(
                "the investable universe must be non-empty with a unique index"
            )

        benchmark = frame[self.input_weight_column].to_numpy(dtype=float)
        if (
            not np.isfinite(benchmark).all()
            or np.any(benchmark < 0)
            or np.max(benchmark) <= 0
        ):
            raise ValueError(
                "benchmark weights must be finite, non-negative, and non-zero"
            )
        benchmark = benchmark / np.max(benchmark)
        benchmark = benchmark / benchmark.sum()
        exposure_matrix = frame.loc[:, target_fields].to_numpy(dtype=float)
        if not np.isfinite(exposure_matrix).all():
            invalid = [
                field_name
                for position, field_name in enumerate(target_fields)
                if not np.isfinite(exposure_matrix[:, position]).all()
            ]
            raise ValueError(
                "target fields contain non-finite values: "
                + ", ".join(invalid)
            )

        lower_weights, upper_weights = weight_bounds(
            benchmark,
            minimum_weight=self.minimum_weight,
            maximum_weight=self.maximum_weight,
            capacity_multiple=self.capacity_multiple,
        )
        target_constraints = [
            _target_constraint(target, exposure_matrix[:, position])
            for position, target in enumerate(targets)
        ]
        group_constraints = group_constraint_specs(
            frame,
            benchmark,
            self.group_tolerances,
        )

        positive_support = benchmark > 0
        if np.any(lower_weights[~positive_support] > self.tolerance):
            raise OptimizationError(
                "a positive minimum weight cannot be assigned outside the "
                "benchmark support under relative entropy"
            )
        support_prior = benchmark[positive_support]
        support_exposures = exposure_matrix[positive_support]
        supported_targets = [
            _restricted_constraint(item, positive_support)
            for item in target_constraints
        ]
        supported_groups = [
            _restricted_constraint(item, positive_support)
            for item in group_constraints
        ]
        support_lower = lower_weights[positive_support]
        support_upper = upper_weights[positive_support]

        hard_error: OptimizationError | None = None
        selected_mode = (
            EntropyExposureMode.HARD
            if mode is EntropyExposureMode.HARD_THEN_ELASTIC
            else mode
        )
        if mode is EntropyExposureMode.ELASTIC:
            solution, algorithm = self._solve_elastic(
                support_exposures,
                support_prior,
                targets,
                supported_groups,
                support_lower,
                support_upper,
            )
        else:
            try:
                solution, algorithm = self._solve_hard(
                    support_exposures,
                    support_prior,
                    targets,
                    supported_targets,
                    supported_groups,
                    support_lower,
                    support_upper,
                )
            except OptimizationError as exc:
                hard_error = exc
                if mode is not EntropyExposureMode.HARD_THEN_ELASTIC:
                    raise
                solution, algorithm = self._solve_elastic(
                    support_exposures,
                    support_prior,
                    targets,
                    supported_groups,
                    support_lower,
                    support_upper,
                )
                selected_mode = EntropyExposureMode.ELASTIC

        support_weights = np.asarray(solution.weights, dtype=float)
        weights = np.zeros(len(frame), dtype=float)
        weights[positive_support] = support_weights
        if (
            not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.0, atol=1.0e-10, rtol=0.0)
        ):
            raise OptimizationError(
                "Entropy Exposure did not produce valid fully invested weights"
            )

        verification_constraints = list(group_constraints)
        if selected_mode is EntropyExposureMode.HARD:
            verification_constraints = [
                *target_constraints,
                *verification_constraints,
            ]
        maximum_violation = _maximum_violation(
            weights,
            verification_constraints,
            lower_weights=lower_weights,
            upper_weights=upper_weights,
        )
        verification_tolerance = max(1.0e-7, self.tolerance * 100.0)
        if maximum_violation > verification_tolerance:
            raise OptimizationError(
                "Entropy Exposure did not satisfy its hard constraints: "
                f"maximum violation={maximum_violation:.3g}"
            )

        full_frame = data_context.get_dataframe(
            [],
            include_excluded_instruments=True,
        )
        result = full_frame.iloc[:, 0:0].copy()
        result[self.output_weight_column] = 0.0
        result.loc[frame.index, self.output_weight_column] = weights
        data_context.set_dataframe(
            result,
            columns=[self.output_weight_column],
        )

        achieved = exposure_matrix.T @ weights
        target_diagnostics = _target_diagnostics(targets, achieved)
        diagnostic_constraints = list(group_constraints)
        if selected_mode is EntropyExposureMode.HARD:
            diagnostic_constraints = [
                *target_constraints,
                *diagnostic_constraints,
            ]
        constraint_diagnostics = _constraint_diagnostics(
            weights,
            diagnostic_constraints,
            tolerance=verification_tolerance,
            instrument_ids=frame.index.to_numpy(),
            lower_weights=lower_weights,
            upper_weights=upper_weights,
        )
        optimization_diagnostics = {
            "algorithm": algorithm,
            "requested_mode": mode.value,
            "mode": selected_mode.value,
            "status": solution.status,
            "iterations": int(solution.iterations),
            "relative_entropy": relative_entropy(
                support_weights,
                support_prior,
            ),
            "maximum_constraint_violation": float(maximum_violation),
            "target_count": len(targets),
            "support_size": int(positive_support.sum()),
            "zero_prior_count": int((~positive_support).sum()),
        }
        target_violations = [
            _constraint_violation(float(value), spec)
            for value, spec in zip(achieved, target_constraints)
        ]
        optimization_diagnostics["maximum_target_violation"] = float(
            max(target_violations, default=0.0)
        )
        if selected_mode is EntropyExposureMode.ELASTIC:
            optimization_diagnostics["soft_target_loss"] = float(
                0.5
                * self.elastic_penalty
                * np.sum(np.square(target_violations))
            )
        if hasattr(solution, "gradient_norm"):
            optimization_diagnostics["gradient_norm"] = float(
                solution.gradient_norm
            )
            optimization_diagnostics["dual_value"] = float(
                solution.dual_value
            )
        if hasattr(solution, "maximum_weight_change"):
            optimization_diagnostics["maximum_weight_change"] = float(
                solution.maximum_weight_change
            )
        if hard_error is not None:
            optimization_diagnostics["fallback_reason"] = str(hard_error)

        for key in (
            "entropy_exposure_optimization",
            "entropy_exposure_optimisation",
        ):
            data_context.diagnostics[key] = dict(optimization_diagnostics)
        data_context.diagnostics["target_diagnostics"] = target_diagnostics
        data_context.diagnostics[
            "constraint_diagnostics"
        ] = constraint_diagnostics
        return data_context

    def _solve_hard(
        self,
        exposures,
        prior,
        targets,
        target_constraints,
        group_constraints,
        lower_weights,
        upper_weights,
    ):
        point_targets = all(
            target.direction is TargetDirection.EQUAL
            and target.tolerance == 0.0
            for target in targets
        )
        box_active = bool(
            np.any(np.asarray(lower_weights) > 0.0)
            or np.any(np.asarray(upper_weights) < 1.0)
        )
        if point_targets and not group_constraints and not box_active:
            solver = self.newton_solver or EGMUNewtonSolver(
                tolerance=self.tolerance,
                max_iterations=self.max_iterations,
                ridge=self.newton_ridge,
            )
            solution = solver.solve(
                exposures,
                prior,
                [target.target for target in targets],
            )
            if solution.status != "converged":
                raise OptimizationError(
                    "EGMU Newton did not converge: "
                    f"status={solution.status}, "
                    f"maximum exposure residual="
                    f"{solution.maximum_exposure_residual:.3g}"
                )
            return solution, "egmu_newton"

        solver = self.projection_solver or EGMUProjectionSolver(
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            root_tolerance=min(self.tolerance * 0.1, 1.0e-12),
        )
        solution = solver.solve(
            prior,
            [*target_constraints, *group_constraints],
            lower_bounds=lower_weights,
            upper_bounds=upper_weights,
        )
        if solution.status != "converged":
            raise OptimizationError(
                "EGMU projection did not converge: "
                f"maximum constraint violation="
                f"{solution.maximum_constraint_violation:.3g}"
            )
        return solution, "egmu_bregman_dykstra"

    def _solve_elastic(
        self,
        exposures,
        prior,
        targets,
        group_constraints,
        lower_weights,
        upper_weights,
    ):
        target_bounds = [
            _target_constraint(
                target,
                exposures[:, position],
            )
            for position, target in enumerate(targets)
        ]
        solver = self.elastic_solver or EGMUConstrainedElasticSolver(
            penalty=self.elastic_penalty,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
        )
        solution = solver.solve(
            exposures,
            prior,
            [item.lower for item in target_bounds],
            [item.upper for item in target_bounds],
            constraints=group_constraints,
            lower_bounds=lower_weights,
            upper_bounds=upper_weights,
        )
        if solution.status != "converged":
            raise OptimizationError(
                "elastic EGMU did not converge: "
                f"status={solution.status}, maximum hard-constraint violation="
                f"{solution.maximum_constraint_violation:.3g}"
            )
        return solution, "egmu_elastic_bregman_dykstra"

    def _validate_configuration(self) -> None:
        if not isinstance(self.input_weight_column, str) or not (
            self.input_weight_column.strip()
        ):
            raise ValueError("input_weight_column must not be empty")
        if not isinstance(self.output_weight_column, str) or not (
            self.output_weight_column.strip()
        ):
            raise ValueError("output_weight_column must not be empty")
        if not isinstance(self.max_iterations, int) or self.max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if not np.isfinite(self.newton_ridge) or self.newton_ridge < 0:
            raise ValueError("newton_ridge must be finite and non-negative")
        if not np.isfinite(self.elastic_penalty) or self.elastic_penalty <= 0:
            raise ValueError("elastic_penalty must be finite and positive")
        if (
            not np.isfinite(self.minimum_weight)
            or not np.isfinite(self.maximum_weight)
            or self.minimum_weight < 0
            or self.maximum_weight <= 0
            or self.minimum_weight > self.maximum_weight
        ):
            raise ValueError(
                "weight bounds must satisfy 0 <= minimum_weight <= "
                "maximum_weight"
            )
        if self.capacity_multiple is not None and (
            not np.isfinite(self.capacity_multiple)
            or self.capacity_multiple <= 0
        ):
            raise ValueError("capacity_multiple must be finite and positive")
        for field_name, tolerance in self.group_tolerances.items():
            if not isinstance(field_name, str) or not field_name.strip():
                raise ValueError("group field names cannot be empty")
            if not np.isfinite(tolerance) or tolerance < 0:
                raise ValueError(
                    "group tolerances must be finite and non-negative"
                )


def _target_constraint(
    target: ExposureTarget,
    coefficients: np.ndarray,
) -> LinearConstraintSpec:
    if target.direction is TargetDirection.EQUAL:
        lower = target.target - target.tolerance
        upper = target.target + target.tolerance
    elif target.direction is TargetDirection.AT_LEAST:
        lower = target.target - target.tolerance
        upper = np.inf
    else:
        lower = -np.inf
        upper = target.target + target.tolerance
    return LinearConstraintSpec(
        coefficients=np.asarray(coefficients, dtype=float),
        lower=float(lower),
        upper=float(upper),
        name=f"target:{target.field}",
    )


def _restricted_constraint(
    spec: LinearConstraintSpec,
    support: np.ndarray,
) -> LinearConstraintSpec:
    return LinearConstraintSpec(
        coefficients=np.asarray(spec.coefficients, dtype=float)[support],
        lower=spec.lower,
        upper=spec.upper,
        name=spec.name,
    )


def _maximum_violation(
    weights: np.ndarray,
    constraints: Sequence[LinearConstraintSpec],
    *,
    lower_weights: np.ndarray,
    upper_weights: np.ndarray,
) -> float:
    maximum = max(
        float(np.max(lower_weights - weights, initial=0.0)),
        float(np.max(weights - upper_weights, initial=0.0)),
        0.0,
    )
    for spec in constraints:
        achieved = float(np.asarray(spec.coefficients, dtype=float) @ weights)
        maximum = max(maximum, _constraint_violation(achieved, spec))
    return maximum


def _constraint_violation(
    achieved: float,
    spec: LinearConstraintSpec,
) -> float:
    return max(
        float(spec.lower) - achieved,
        achieved - float(spec.upper),
        0.0,
    )


def _target_diagnostics(
    targets: Sequence[ExposureTarget],
    achieved: np.ndarray,
) -> list[dict]:
    rows = []
    for target, value in zip(targets, achieved):
        spec = _target_constraint(target, np.ones(1))
        rows.append(
            {
                "name": target.field,
                "requested": target.target,
                "achieved": float(value),
                "direction": target.direction.value,
                "tolerance": target.tolerance,
                "lower": _finite_or_none(spec.lower),
                "upper": _finite_or_none(spec.upper),
            }
        )
    return rows


def _constraint_diagnostics(
    weights: np.ndarray,
    constraints: Sequence[LinearConstraintSpec],
    *,
    tolerance: float,
    instrument_ids: np.ndarray,
    lower_weights: np.ndarray,
    upper_weights: np.ndarray,
) -> list[dict]:
    rows = []
    for spec in constraints:
        value = float(np.asarray(spec.coefficients, dtype=float) @ weights)
        violation = _constraint_violation(value, spec)
        rows.append(
            {
                "name": spec.name,
                "value": value,
                "lower": _finite_or_none(spec.lower),
                "upper": _finite_or_none(spec.upper),
                "violation": float(violation),
                "status": (
                    "violated" if violation > tolerance else "satisfied"
                ),
            }
        )
    for instrument_id, value, lower, upper in zip(
        instrument_ids,
        weights,
        lower_weights,
        upper_weights,
    ):
        if lower <= 0.0 and upper >= 1.0:
            continue
        violation = max(float(lower - value), float(value - upper), 0.0)
        rows.append(
            {
                "name": f"weight:{instrument_id}",
                "value": float(value),
                "lower": float(lower),
                "upper": float(upper),
                "violation": violation,
                "status": (
                    "violated" if violation > tolerance else "satisfied"
                ),
            }
        )
    return rows


def _finite_or_none(value: float) -> float | None:
    converted = float(value)
    return converted if np.isfinite(converted) else None


__all__ = [
    "EntropyExposureEngine",
    "EntropyExposureMode",
    "ExposureTarget",
    "TargetDirection",
]
