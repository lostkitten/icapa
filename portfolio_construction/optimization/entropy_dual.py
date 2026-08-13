"""Low-dimensional EGMU dual solvers for exposure targets."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EGMUResult:
    """Result of a hard or elastic EGMU dual solve."""

    weights: np.ndarray
    theta: np.ndarray
    iterations: int
    elapsed_seconds: float
    status: str
    gradient_norm: float
    dual_value: float
    maximum_exposure_residual: float

    def __post_init__(self) -> None:
        weights = _readonly_vector(self.weights, "weights")
        theta = _readonly_vector(self.theta, "theta")
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        for value, label in (
            (self.elapsed_seconds, "elapsed_seconds"),
            (self.gradient_norm, "gradient_norm"),
            (self.maximum_exposure_residual, "maximum_exposure_residual"),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        if not np.isfinite(self.dual_value):
            raise ValueError("dual_value must be finite")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("status must not be empty")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "theta", theta)


@dataclass(frozen=True)
class EGMUNewtonSolver:
    """Solve exact exposure targets with damped dual Newton updates."""

    tolerance: float = 1.0e-8
    max_iterations: int = 100
    ridge: float = 1.0e-10
    armijo_constant: float = 1.0e-4
    backtracking_factor: float = 0.5
    max_backtracking: int = 60

    def solve(
        self,
        exposures: np.ndarray,
        prior_weights: Sequence[float],
        targets: Sequence[float],
    ) -> EGMUResult:
        """Return an exact-target solve result; callers must check status."""

        return egmu_newton(
            exposures,
            prior_weights,
            targets,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            ridge=self.ridge,
            armijo_constant=self.armijo_constant,
            backtracking_factor=self.backtracking_factor,
            max_backtracking=self.max_backtracking,
        )


@dataclass(frozen=True)
class EGMUElasticSolver:
    """Solve point-valued soft exposure targets with the elastic EGMU dual."""

    penalty: float = 100.0
    tolerance: float = 1.0e-8
    max_iterations: int = 100
    ridge: float = 1.0e-10
    armijo_constant: float = 1.0e-4
    backtracking_factor: float = 0.5
    max_backtracking: int = 60

    def solve(
        self,
        exposures: np.ndarray,
        prior_weights: Sequence[float],
        targets: Sequence[float],
    ) -> EGMUResult:
        """Return the elastic minimum-KL target portfolio."""

        return egmu_elastic(
            exposures,
            prior_weights,
            targets,
            penalty=self.penalty,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            ridge=self.ridge,
            armijo_constant=self.armijo_constant,
            backtracking_factor=self.backtracking_factor,
            max_backtracking=self.max_backtracking,
        )


def egmu_newton(
    exposures: np.ndarray,
    prior_weights: Sequence[float],
    targets: Sequence[float],
    *,
    tolerance: float = 1.0e-8,
    max_iterations: int = 100,
    ridge: float = 1.0e-10,
    armijo_constant: float = 1.0e-4,
    backtracking_factor: float = 0.5,
    max_backtracking: int = 60,
) -> EGMUResult:
    """Solve an exact minimum-relative-entropy exposure projection.

    Primal weights have the multiplicative form
    ``w_i(theta) proportional to b_i * exp(x_i @ theta)``. Newton updates
    therefore solve only an exposure-dimensional covariance system.
    """

    started = perf_counter()
    matrix, prior, target = _validate_target_inputs(
        exposures,
        prior_weights,
        targets,
    )
    _validate_configuration(
        tolerance=tolerance,
        max_iterations=max_iterations,
        ridge=ridge,
        armijo_constant=armijo_constant,
        backtracking_factor=backtracking_factor,
        max_backtracking=max_backtracking,
    )
    theta = np.zeros(matrix.shape[1], dtype=float)
    identity = np.eye(matrix.shape[1], dtype=float)
    weights, dual_value = _dual_state(matrix, prior, target, theta)
    iterations = 0
    status = "max_iterations"

    while iterations < max_iterations:
        mean = matrix.T @ weights
        gradient = target - mean
        if float(np.linalg.norm(gradient)) <= tolerance:
            status = "converged"
            break
        centered = matrix - mean
        covariance = centered.T @ (weights[:, None] * centered)
        step = _linear_step(covariance + ridge * identity, gradient)
        directional_derivative = float(gradient @ step)
        if (
            not np.isfinite(directional_derivative)
            or directional_derivative <= 0
        ):
            status = "invalid_newton_step"
            break

        accepted = False
        step_size = 1.0
        for _ in range(max_backtracking):
            candidate_theta = theta + step_size * step
            candidate_weights, candidate_value = _dual_state(
                matrix,
                prior,
                target,
                candidate_theta,
            )
            gain = _stable_dual_increment(
                matrix,
                weights,
                target,
                theta,
                step,
                step_size,
            )
            candidate_gradient = target - matrix.T @ candidate_weights
            if _accept_step(
                gain=gain,
                required_gain=(
                    armijo_constant * step_size * directional_derivative
                ),
                current_gradient_norm=float(np.linalg.norm(gradient)),
                candidate_gradient_norm=float(
                    np.linalg.norm(candidate_gradient)
                ),
                objective_scale=dual_value,
            ):
                theta = candidate_theta
                weights = candidate_weights
                dual_value = candidate_value
                accepted = True
                break
            step_size *= backtracking_factor
        if not accepted:
            status = "line_search_failed"
            break
        iterations += 1

    residual = matrix.T @ weights - target
    gradient_norm = float(np.linalg.norm(residual))
    if gradient_norm <= tolerance:
        status = "converged"
    return EGMUResult(
        weights=weights,
        theta=theta,
        iterations=iterations,
        elapsed_seconds=perf_counter() - started,
        status=status,
        gradient_norm=gradient_norm,
        dual_value=dual_value,
        maximum_exposure_residual=float(
            np.max(np.abs(residual), initial=0.0)
        ),
    )


def egmu_elastic(
    exposures: np.ndarray,
    prior_weights: Sequence[float],
    targets: Sequence[float],
    *,
    penalty: float = 100.0,
    tolerance: float = 1.0e-8,
    max_iterations: int = 100,
    ridge: float = 1.0e-10,
    armijo_constant: float = 1.0e-4,
    backtracking_factor: float = 0.5,
    max_backtracking: int = 60,
) -> EGMUResult:
    """Solve the elastic EGMU point-target problem in the exposure dual."""

    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("penalty must be finite and positive")
    started = perf_counter()
    matrix, prior, target = _validate_target_inputs(
        exposures,
        prior_weights,
        targets,
    )
    _validate_configuration(
        tolerance=tolerance,
        max_iterations=max_iterations,
        ridge=ridge,
        armijo_constant=armijo_constant,
        backtracking_factor=backtracking_factor,
        max_backtracking=max_backtracking,
    )
    theta = np.zeros(matrix.shape[1], dtype=float)
    identity = np.eye(matrix.shape[1], dtype=float)

    def elastic_state(value: np.ndarray) -> tuple[np.ndarray, float]:
        weights, base_value = _dual_state(matrix, prior, target, value)
        return weights, float(base_value - 0.5 * (value @ value) / penalty)

    weights, dual_value = elastic_state(theta)
    iterations = 0
    status = "max_iterations"

    while iterations < max_iterations:
        mean = matrix.T @ weights
        gradient = target - mean - theta / penalty
        if float(np.linalg.norm(gradient)) <= tolerance:
            status = "converged"
            break
        centered = matrix - mean
        covariance = centered.T @ (weights[:, None] * centered)
        system = covariance + (1.0 / penalty + ridge) * identity
        step = _linear_step(system, gradient)
        directional_derivative = float(gradient @ step)
        if (
            not np.isfinite(directional_derivative)
            or directional_derivative <= 0
        ):
            status = "invalid_newton_step"
            break

        accepted = False
        step_size = 1.0
        for _ in range(max_backtracking):
            candidate_theta = theta + step_size * step
            candidate_weights, candidate_value = elastic_state(
                candidate_theta
            )
            gain = _stable_dual_increment(
                matrix,
                weights,
                target,
                theta,
                step,
                step_size,
                penalty=penalty,
            )
            candidate_gradient = (
                target
                - matrix.T @ candidate_weights
                - candidate_theta / penalty
            )
            if _accept_step(
                gain=gain,
                required_gain=(
                    armijo_constant * step_size * directional_derivative
                ),
                current_gradient_norm=float(np.linalg.norm(gradient)),
                candidate_gradient_norm=float(
                    np.linalg.norm(candidate_gradient)
                ),
                objective_scale=dual_value,
            ):
                theta = candidate_theta
                weights = candidate_weights
                dual_value = candidate_value
                accepted = True
                break
            step_size *= backtracking_factor
        if not accepted:
            status = "line_search_failed"
            break
        iterations += 1

    mean = matrix.T @ weights
    gradient = target - mean - theta / penalty
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm <= tolerance:
        status = "converged"
    residual = mean - target
    return EGMUResult(
        weights=weights,
        theta=theta,
        iterations=iterations,
        elapsed_seconds=perf_counter() - started,
        status=status,
        gradient_norm=gradient_norm,
        dual_value=dual_value,
        maximum_exposure_residual=float(
            np.max(np.abs(residual), initial=0.0)
        ),
    )


def _validate_target_inputs(
    exposures: np.ndarray,
    prior_weights: Sequence[float],
    targets: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(exposures, dtype=float)
    prior = np.asarray(prior_weights, dtype=float).reshape(-1)
    target = np.asarray(targets, dtype=float).reshape(-1)
    if matrix.ndim != 2:
        raise ValueError(
            "exposures must have shape (instrument_count, target_count)"
        )
    if (
        len(prior) == 0
        or prior.shape != (matrix.shape[0],)
        or not np.isfinite(prior).all()
        or np.any(prior <= 0)
    ):
        raise ValueError(
            "prior_weights must be a finite, strictly positive vector "
            "matching exposure rows"
        )
    if matrix.shape[1] == 0 or target.shape != (matrix.shape[1],):
        raise ValueError("targets must match the non-empty exposure columns")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("exposures and targets must be finite")
    scaled_prior = prior / float(np.max(prior))
    prior = scaled_prior / float(scaled_prior.sum())
    if (
        not np.isfinite(prior).all()
        or np.any(prior < np.finfo(float).tiny)
    ):
        raise ValueError(
            "prior_weights have too much dynamic range to preserve support"
        )
    return matrix, prior, target


def _validate_configuration(
    *,
    tolerance: float,
    max_iterations: int,
    ridge: float,
    armijo_constant: float,
    backtracking_factor: float,
    max_backtracking: int,
) -> None:
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and non-negative")
    if not np.isfinite(armijo_constant) or not 0 < armijo_constant < 1:
        raise ValueError("armijo_constant must be between zero and one")
    if (
        not np.isfinite(backtracking_factor)
        or not 0 < backtracking_factor < 1
    ):
        raise ValueError("backtracking_factor must be between zero and one")
    if not isinstance(max_backtracking, int) or max_backtracking <= 0:
        raise ValueError("max_backtracking must be a positive integer")


def _dual_state(
    exposures: np.ndarray,
    prior: np.ndarray,
    target: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, float]:
    scores = np.log(prior) + exposures @ theta
    if not np.isfinite(scores).all():
        raise ValueError("EGMU dual scores became non-finite")
    maximum = float(np.max(scores))
    shifted = np.exp(scores - maximum)
    partition = float(shifted.sum())
    weights = shifted / partition
    log_partition = float(np.log(partition) + maximum)
    return weights, float(theta @ target - log_partition)


def _stable_dual_increment(
    exposures: np.ndarray,
    weights: np.ndarray,
    target: np.ndarray,
    theta: np.ndarray,
    step: np.ndarray,
    step_size: float,
    *,
    penalty: float | None = None,
) -> float:
    """Evaluate a dual change without subtracting two order-one values."""

    extended = np.longdouble
    extended_weights = weights.astype(extended)
    exposure_step = (exposures @ step).astype(extended)
    mean_step = np.sum(extended_weights * exposure_step, dtype=extended)
    centered = exposure_step - mean_step
    scaled = extended(step_size) * centered
    maximum = np.max(scaled)
    log_mgf = maximum + np.log(
        np.sum(
            extended_weights * np.exp(scaled - maximum),
            dtype=extended,
        )
    )
    target_step = np.sum(
        target.astype(extended) * step.astype(extended),
        dtype=extended,
    )
    gain = extended(step_size) * (target_step - mean_step) - log_mgf
    if penalty is not None:
        extended_theta = theta.astype(extended)
        extended_step = step.astype(extended)
        gain -= (
            extended(step_size)
            * np.sum(extended_theta * extended_step, dtype=extended)
            / extended(penalty)
        )
        gain -= (
            extended(0.5)
            * extended(step_size) ** 2
            * np.sum(extended_step * extended_step, dtype=extended)
            / extended(penalty)
        )
    return float(gain)


def _accept_step(
    *,
    gain: float,
    required_gain: float,
    current_gradient_norm: float,
    candidate_gradient_norm: float,
    objective_scale: float,
) -> bool:
    if gain >= required_gain:
        return True
    numerical_slack = float(
        64.0
        * np.finfo(np.longdouble).eps
        * max(1.0, abs(objective_scale))
    )
    return (
        gain >= -numerical_slack
        and candidate_gradient_norm < current_gradient_norm
    )


def _linear_step(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, vector, rcond=None)[0]


def _readonly_vector(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    result.setflags(write=False)
    return result


__all__ = [
    "EGMUElasticSolver",
    "EGMUNewtonSolver",
    "EGMUResult",
    "egmu_elastic",
    "egmu_newton",
]
