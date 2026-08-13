"""KL projections for constrained Entropy-Guided Multiplicative Updates."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np

from .constraints import LinearConstraintSpec
from .problem import OptimizationError

@dataclass(frozen=True)
class EGMUProjectionResult:
    """Result of a hard or constrained-elastic KL projection."""

    weights: np.ndarray
    iterations: int
    elapsed_seconds: float
    status: str
    maximum_constraint_violation: float
    maximum_weight_change: float
    relative_entropy: float

    def __post_init__(self) -> None:
        weights = _readonly_vector(self.weights, "weights")
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        for value, label in (
            (self.elapsed_seconds, "elapsed_seconds"),
            (
                self.maximum_constraint_violation,
                "maximum_constraint_violation",
            ),
            (self.maximum_weight_change, "maximum_weight_change"),
            (self.relative_entropy, "relative_entropy"),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("status must not be empty")
        object.__setattr__(self, "weights", weights)

@dataclass(frozen=True)
class EGMUProjectionSolver:
    """Project a prior onto linear constraints in relative-entropy geometry."""

    tolerance: float = 1.0e-8
    max_iterations: int = 10_000
    root_tolerance: float = 1.0e-12
    max_root_iterations: int = 120
    max_bracket_expansions: int = 1_024

    def solve(
        self,
        prior_weights: Sequence[float],
        constraints: Sequence[LinearConstraintSpec],
        *,
        lower_bounds: Sequence[float] | None = None,
        upper_bounds: Sequence[float] | None = None,
    ) -> EGMUProjectionResult:
        """Return the KL projection onto all supplied hard constraints."""

        return egmu_project_linear(
            prior_weights,
            constraints,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            root_tolerance=self.root_tolerance,
            max_root_iterations=self.max_root_iterations,
            max_bracket_expansions=self.max_bracket_expansions,
        )

@dataclass(frozen=True)
class EGMUConstrainedElasticSolver:
    """Combine soft exposure slabs with hard portfolio constraints."""

    penalty: float = 100.0
    tolerance: float = 1.0e-8
    max_iterations: int = 1_000
    root_tolerance: float = 1.0e-12
    max_root_iterations: int = 120
    max_bracket_expansions: int = 1_024

    def solve(
        self,
        exposures: np.ndarray,
        prior_weights: Sequence[float],
        target_lower_bounds: Sequence[float],
        target_upper_bounds: Sequence[float],
        *,
        constraints: Sequence[LinearConstraintSpec] = (),
        lower_bounds: Sequence[float] | None = None,
        upper_bounds: Sequence[float] | None = None,
    ) -> EGMUProjectionResult:
        """Return the constrained elastic minimum-KL portfolio."""

        return egmu_project_elastic(
            exposures,
            prior_weights,
            target_lower_bounds,
            target_upper_bounds,
            constraints=constraints,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            penalty=self.penalty,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            root_tolerance=self.root_tolerance,
            max_root_iterations=self.max_root_iterations,
            max_bracket_expansions=self.max_bracket_expansions,
        )


def egmu_project_linear(
    prior_weights: Sequence[float],
    constraints: Sequence[LinearConstraintSpec],
    *,
    lower_bounds: Sequence[float] | None = None,
    upper_bounds: Sequence[float] | None = None,
    tolerance: float = 1.0e-8,
    max_iterations: int = 10_000,
    root_tolerance: float = 1.0e-12,
    max_root_iterations: int = 120,
    max_bracket_expansions: int = 1_024,
) -> EGMUProjectionResult:
    """KL-project a positive prior onto linear slabs and a weight box.

    Scalar slab corrections and one bounded-simplex block use ``O(N + J)``
    storage for ``N`` instruments and ``J`` constraints.
    """

    started = perf_counter()
    prior = _validate_prior(prior_weights)
    specs, coefficients = _validated_constraints(constraints, len(prior))
    lower, upper, box_active = _validated_box(
        len(prior),
        lower_bounds,
        upper_bounds,
    )
    _validate_projection_configuration(
        tolerance=tolerance,
        max_iterations=max_iterations,
        root_tolerance=root_tolerance,
        max_root_iterations=max_root_iterations,
        max_bracket_expansions=max_bracket_expansions,
    )
    if not specs and not box_active:
        return _projection_result(prior, prior, started, 0, "converged", 0.0)

    weights = prior.copy()
    slab_corrections = np.zeros(len(specs), dtype=float)
    box_correction = np.zeros(len(prior), dtype=float) if box_active else None
    maximum_change = float("inf")
    maximum_violation = _maximum_violation(
        weights,
        specs,
        coefficients,
        lower,
        upper,
    )
    status = "max_iterations"
    iterations = 0

    for cycle in range(1, max_iterations + 1):
        previous = weights.copy()
        for position, (spec, vector) in enumerate(zip(specs, coefficients)):
            corrected = _apply_scalar_correction(
                weights,
                vector,
                slab_corrections[position],
            )
            weights, tilt = _project_linear_slab(
                corrected,
                vector,
                lower=float(spec.lower),
                upper=float(spec.upper),
                tolerance=root_tolerance,
                max_root_iterations=max_root_iterations,
                max_bracket_expansions=max_bracket_expansions,
                name=spec.name,
            )
            slab_corrections[position] = -tilt

        if box_correction is not None:
            corrected = _normalised_log_weights(
                np.log(weights) + box_correction
            )
            weights = _project_bounded_simplex(
                corrected,
                lower,
                upper,
                tolerance=root_tolerance,
                max_iterations=max_root_iterations,
                max_bracket_expansions=max_bracket_expansions,
            )
            box_correction = np.log(corrected) - np.log(weights)

        iterations = cycle
        maximum_change = float(np.max(np.abs(weights - previous)))
        cycle_change = float(np.sum(np.abs(weights - previous)))
        maximum_violation = _maximum_violation(
            weights,
            specs,
            coefficients,
            lower,
            upper,
        )
        if maximum_violation <= tolerance and cycle_change <= tolerance:
            status = "converged"
            break

    return _projection_result(
        weights,
        prior,
        started,
        iterations,
        status,
        maximum_change,
        maximum_violation,
    )


def egmu_project_elastic(
    exposures: np.ndarray,
    prior_weights: Sequence[float],
    target_lower_bounds: Sequence[float],
    target_upper_bounds: Sequence[float],
    *,
    constraints: Sequence[LinearConstraintSpec] = (),
    lower_bounds: Sequence[float] | None = None,
    upper_bounds: Sequence[float] | None = None,
    penalty: float = 100.0,
    tolerance: float = 1.0e-8,
    max_iterations: int = 1_000,
    root_tolerance: float = 1.0e-12,
    max_root_iterations: int = 120,
    max_bracket_expansions: int = 1_024,
) -> EGMUProjectionResult:
    """Solve squared-distance exposure targeting with hard constraints.

    Soft targets, hard slabs, and the weight box are separate generalized-
    Dykstra blocks.
    """

    started = perf_counter()
    prior = _validate_prior(prior_weights)
    matrix, target_lower, target_upper = _validated_target_box(
        exposures,
        len(prior),
        target_lower_bounds,
        target_upper_bounds,
    )
    specs, coefficients = _validated_constraints(constraints, len(prior))
    lower, upper, box_active = _validated_box(
        len(prior),
        lower_bounds,
        upper_bounds,
    )
    _validate_projection_configuration(
        tolerance=tolerance,
        max_iterations=max_iterations,
        root_tolerance=root_tolerance,
        max_root_iterations=max_root_iterations,
        max_bracket_expansions=max_bracket_expansions,
    )
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("penalty must be finite and positive")

    weights = prior.copy()
    target_corrections = np.zeros(matrix.shape[1], dtype=float)
    slab_corrections = np.zeros(len(specs), dtype=float)
    box_correction = np.zeros(len(prior), dtype=float) if box_active else None
    maximum_change = float("inf")
    maximum_violation = _maximum_violation(
        weights,
        specs,
        coefficients,
        lower,
        upper,
    )
    status = "max_iterations"
    iterations = 0

    for cycle in range(1, max_iterations + 1):
        previous = weights.copy()
        for position in range(matrix.shape[1]):
            vector = matrix[:, position]
            corrected = _apply_scalar_correction(
                weights,
                vector,
                target_corrections[position],
            )
            weights, tilt = _elastic_slab_prox(
                corrected,
                vector,
                lower=float(target_lower[position]),
                upper=float(target_upper[position]),
                penalty=float(penalty),
                tolerance=root_tolerance,
                max_root_iterations=max_root_iterations,
                max_bracket_expansions=max_bracket_expansions,
            )
            target_corrections[position] = -tilt

        for position, (spec, vector) in enumerate(zip(specs, coefficients)):
            corrected = _apply_scalar_correction(
                weights,
                vector,
                slab_corrections[position],
            )
            weights, tilt = _project_linear_slab(
                corrected,
                vector,
                lower=float(spec.lower),
                upper=float(spec.upper),
                tolerance=root_tolerance,
                max_root_iterations=max_root_iterations,
                max_bracket_expansions=max_bracket_expansions,
                name=spec.name,
            )
            slab_corrections[position] = -tilt

        if box_correction is not None:
            corrected = _normalised_log_weights(
                np.log(weights) + box_correction
            )
            weights = _project_bounded_simplex(
                corrected,
                lower,
                upper,
                tolerance=root_tolerance,
                max_iterations=max_root_iterations,
                max_bracket_expansions=max_bracket_expansions,
            )
            box_correction = np.log(corrected) - np.log(weights)

        iterations = cycle
        maximum_change = float(np.max(np.abs(weights - previous)))
        cycle_change = float(np.sum(np.abs(weights - previous)))
        maximum_violation = _maximum_violation(
            weights,
            specs,
            coefficients,
            lower,
            upper,
        )
        if maximum_violation <= tolerance and cycle_change <= tolerance:
            status = "converged"
            break

    return _projection_result(
        weights,
        prior,
        started,
        iterations,
        status,
        maximum_change,
        maximum_violation,
    )


def relative_entropy(
    weights: Sequence[float],
    prior_weights: Sequence[float],
) -> float:
    """Return ``KL(weights || prior_weights)`` on a common support."""

    values = _validated_probability(weights, "weights", allow_zero=True)
    prior = _validated_probability(
        prior_weights,
        "prior_weights",
        allow_zero=True,
    )
    if prior.shape != values.shape:
        raise ValueError("weights and prior_weights must have the same shape")
    if np.any((values > 0) & (prior == 0)):
        return float("inf")
    positive = values > 0
    result = float(
        np.sum(values[positive] * np.log(values[positive] / prior[positive]))
    )
    if result < -1.0e-10:
        raise ValueError("relative entropy is negative beyond numerical noise")
    return max(0.0, result)


def _validated_target_box(
    exposures: np.ndarray,
    instrument_count: int,
    target_lower_bounds: Sequence[float],
    target_upper_bounds: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(exposures, dtype=float)
    lower = np.asarray(target_lower_bounds, dtype=float).reshape(-1)
    upper = np.asarray(target_upper_bounds, dtype=float).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != instrument_count:
        raise ValueError(
            "exposures must have shape (instrument_count, target_count)"
        )
    if matrix.shape[1] == 0 or lower.shape != (matrix.shape[1],):
        raise ValueError("target bounds must match non-empty exposure columns")
    if upper.shape != lower.shape:
        raise ValueError("target lower and upper bounds must have equal length")
    if (
        not np.isfinite(matrix).all()
        or np.isnan(lower).any()
        or np.isnan(upper).any()
        or np.any(lower > upper)
    ):
        raise ValueError("exposures and target bounds must be valid")
    return matrix, lower, upper


def _validated_constraints(
    constraints: Sequence[LinearConstraintSpec],
    size: int,
) -> tuple[tuple[LinearConstraintSpec, ...], tuple[np.ndarray, ...]]:
    specs = tuple(constraints)
    coefficients = []
    for spec in specs:
        if not isinstance(spec, LinearConstraintSpec):
            raise TypeError("constraints must contain LinearConstraintSpec values")
        vector = np.asarray(spec.coefficients, dtype=float)
        if vector.shape != (size,) or not np.isfinite(vector).all():
            raise ValueError(f"invalid coefficients for constraint {spec.name!r}")
        if (
            not isinstance(spec.name, str)
            or not spec.name
            or np.isnan(float(spec.lower))
            or np.isnan(float(spec.upper))
            or float(spec.lower) > float(spec.upper)
        ):
            raise ValueError(f"invalid constraint {spec.name!r}")
        coefficients.append(vector)
    return specs, tuple(coefficients)


def _validated_box(
    size: int,
    lower_bounds: Sequence[float] | None,
    upper_bounds: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    lower = (
        np.zeros(size, dtype=float)
        if lower_bounds is None
        else np.asarray(lower_bounds, dtype=float).reshape(-1)
    )
    upper = (
        np.ones(size, dtype=float)
        if upper_bounds is None
        else np.asarray(upper_bounds, dtype=float).reshape(-1)
    )
    if (
        lower.shape != (size,)
        or upper.shape != (size,)
        or not np.isfinite(lower).all()
        or np.isnan(upper).any()
        or np.any(lower < 0)
        or np.any(upper < 0)
        or np.any(lower > upper)
    ):
        raise ValueError("weight bounds must be valid vectors matching the prior")
    feasibility_tolerance = 1.0e-12
    if lower.sum() > 1.0 + feasibility_tolerance:
        raise OptimizationError("minimum weights exceed the investment level")
    if upper.sum() < 1.0 - feasibility_tolerance:
        raise OptimizationError("maximum weights cannot reach the investment level")
    active = bool(np.any(lower > 0) or np.any(upper < 1.0))
    return lower, upper, active


def _validate_projection_configuration(
    *,
    tolerance: float,
    max_iterations: int,
    root_tolerance: float,
    max_root_iterations: int,
    max_bracket_expansions: int,
) -> None:
    for value, label in (
        (tolerance, "tolerance"),
        (root_tolerance, "root_tolerance"),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be finite and positive")
    for value, label in (
        (max_iterations, "max_iterations"),
        (max_root_iterations, "max_root_iterations"),
        (max_bracket_expansions, "max_bracket_expansions"),
    ):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")


def _validate_prior(prior_weights: Sequence[float]) -> np.ndarray:
    prior = np.asarray(prior_weights, dtype=float).reshape(-1)
    if len(prior) == 0 or not np.isfinite(prior).all() or np.any(prior <= 0):
        raise ValueError("prior_weights must be a finite, strictly positive vector")
    scaled = prior / float(np.max(prior))
    result = scaled / float(scaled.sum())
    if not np.isfinite(result).all() or np.any(
        result < np.finfo(float).tiny
    ):
        raise ValueError(
            "prior_weights have too much dynamic range to preserve support"
        )
    return result


def _validated_probability(
    values: Sequence[float],
    name: str,
    *,
    allow_zero: bool,
) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    invalid_sign = np.any(result < 0) if allow_zero else np.any(result <= 0)
    if (
        len(result) == 0
        or not np.isfinite(result).all()
        or invalid_sign
        or not np.isclose(result.sum(), 1.0, atol=1.0e-10, rtol=0.0)
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} probability vector")
    return result


def _apply_scalar_correction(
    weights: np.ndarray,
    coefficients: np.ndarray,
    correction: float,
) -> np.ndarray:
    if correction == 0.0:
        return weights
    return _normalised_log_weights(
        np.log(weights) + correction * coefficients
    )


def _project_linear_slab(
    prior: np.ndarray,
    coefficients: np.ndarray,
    *,
    lower: float,
    upper: float,
    tolerance: float,
    max_root_iterations: int,
    max_bracket_expansions: int,
    name: str,
) -> tuple[np.ndarray, float]:
    achieved = float(coefficients @ prior)
    if lower - tolerance <= achieved <= upper + tolerance:
        return prior, 0.0
    target = lower if achieved < lower else upper
    minimum = float(np.min(coefficients))
    maximum = float(np.max(coefficients))
    scale = max(1.0, abs(target), abs(minimum), abs(maximum))
    effective_tolerance = max(
        tolerance,
        np.finfo(float).eps * scale * 32.0,
    )
    if target < minimum - effective_tolerance or target > maximum + effective_tolerance:
        raise OptimizationError(
            f"constraint {name!r} is outside the exposure hull of the EGMU support"
        )
    if (
        abs(target - minimum) <= effective_tolerance
        or abs(target - maximum) <= effective_tolerance
    ):
        raise OptimizationError(
            f"constraint {name!r} requires an exposure-hull boundary portfolio"
        )
    return _solve_moment_root(
        prior,
        coefficients,
        target=target,
        penalty=None,
        tolerance=effective_tolerance,
        max_root_iterations=max_root_iterations,
        max_bracket_expansions=max_bracket_expansions,
        name=name,
    )


def _elastic_slab_prox(
    prior: np.ndarray,
    coefficients: np.ndarray,
    *,
    lower: float,
    upper: float,
    penalty: float,
    tolerance: float,
    max_root_iterations: int,
    max_bracket_expansions: int,
) -> tuple[np.ndarray, float]:
    achieved = float(coefficients @ prior)
    if lower - tolerance <= achieved <= upper + tolerance:
        return prior, 0.0
    target = lower if achieved < lower else upper
    return _solve_moment_root(
        prior,
        coefficients,
        target=target,
        penalty=penalty,
        tolerance=tolerance,
        max_root_iterations=max_root_iterations,
        max_bracket_expansions=max_bracket_expansions,
        name="elastic exposure target",
    )


def _solve_moment_root(
    prior: np.ndarray,
    coefficients: np.ndarray,
    *,
    target: float,
    penalty: float | None,
    tolerance: float,
    max_root_iterations: int,
    max_bracket_expansions: int,
    name: str,
) -> tuple[np.ndarray, float]:
    def state(tilt: float) -> tuple[float, np.ndarray]:
        weights = _exponential_tilt(prior, coefficients, tilt)
        residual = float(coefficients @ weights - target)
        if penalty is not None:
            residual += tilt / penalty
        return residual, weights

    initial, _ = state(0.0)
    if abs(initial) <= tolerance:
        return prior, 0.0
    if initial < 0:
        left, right = 0.0, 1.0
        for _ in range(max_bracket_expansions):
            value, _ = state(right)
            if value >= 0:
                break
            right *= 2.0
        else:
            raise OptimizationError(f"could not bracket KL root for {name!r}")
    else:
        left, right = -1.0, 0.0
        for _ in range(max_bracket_expansions):
            value, _ = state(left)
            if value <= 0:
                break
            left *= 2.0
        else:
            raise OptimizationError(f"could not bracket KL root for {name!r}")

    candidate = prior
    tilt = 0.0
    for _ in range(max_root_iterations):
        tilt = 0.5 * (left + right)
        value, candidate = state(tilt)
        if abs(value) <= tolerance:
            return candidate, tilt
        if value < 0:
            left = tilt
        else:
            right = tilt
    raise OptimizationError(f"KL projection root did not converge for {name!r}")


def _project_bounded_simplex(
    prior: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
    max_bracket_expansions: int,
) -> np.ndarray:
    clipped = np.clip(prior, lower, upper)
    residual_at_one = float(clipped.sum() - 1.0)
    if abs(residual_at_one) <= tolerance:
        return _require_positive_box_result(clipped)

    if residual_at_one > 0:
        left, right = 0.0, 1.0
    else:
        left, right = 1.0, 2.0
        for _ in range(max_bracket_expansions):
            if float(np.clip(right * prior, lower, upper).sum()) >= 1.0:
                break
            right *= 2.0
        else:
            raise OptimizationError(
                "could not bracket bounded-simplex projection"
            )

    result = clipped
    for _ in range(max_iterations):
        scale = 0.5 * (left + right)
        result = np.clip(scale * prior, lower, upper)
        residual = float(result.sum() - 1.0)
        if abs(residual) <= tolerance:
            return _require_positive_box_result(result)
        if residual < 0:
            left = scale
        else:
            right = scale
    raise OptimizationError("bounded-simplex KL projection did not converge")


def _require_positive_box_result(weights: np.ndarray) -> np.ndarray:
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise OptimizationError(
            "weight bounds require leaving the strictly positive EGMU support"
        )
    return weights


def _maximum_violation(
    weights: np.ndarray,
    specs: tuple[LinearConstraintSpec, ...],
    coefficients: tuple[np.ndarray, ...],
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    maximum = max(
        float(np.max(lower - weights, initial=0.0)),
        float(np.max(weights - upper, initial=0.0)),
        0.0,
    )
    for spec, vector in zip(specs, coefficients):
        achieved = float(vector @ weights)
        maximum = max(
            maximum,
            float(spec.lower) - achieved,
            achieved - float(spec.upper),
        )
    return max(0.0, maximum)


def _projection_result(
    weights: np.ndarray,
    prior: np.ndarray,
    started: float,
    iterations: int,
    status: str,
    maximum_change: float,
    maximum_violation: float = 0.0,
) -> EGMUProjectionResult:
    return EGMUProjectionResult(
        weights=weights,
        iterations=iterations,
        elapsed_seconds=perf_counter() - started,
        status=status,
        maximum_constraint_violation=maximum_violation,
        maximum_weight_change=maximum_change,
        relative_entropy=relative_entropy(weights, prior),
    )


def _exponential_tilt(
    prior: np.ndarray,
    coefficients: np.ndarray,
    tilt: float,
) -> np.ndarray:
    return _normalised_log_weights(np.log(prior) + tilt * coefficients)


def _normalised_log_weights(log_values: np.ndarray) -> np.ndarray:
    if not np.isfinite(log_values).all():
        raise OptimizationError("EGMU correction produced non-finite scores")
    maximum = float(np.max(log_values))
    shifted = np.exp(log_values - maximum)
    total = float(shifted.sum())
    result = shifted / total
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise OptimizationError(
            "EGMU correction left the numerically positive support"
        )
    return result


def _readonly_vector(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    result.setflags(write=False)
    return result


__all__ = [
    "EGMUConstrainedElasticSolver",
    "EGMUProjectionResult",
    "EGMUProjectionSolver",
    "egmu_project_elastic",
    "egmu_project_linear",
    "relative_entropy",
]
