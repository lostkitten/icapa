"""Objective builders for reusable portfolio-optimization problems."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def squared_distance_objective(
    target_weights: Sequence[float],
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """Return objective and gradient for squared distance to target weights."""

    target = np.asarray(target_weights, dtype=float)
    if target.ndim != 1 or not np.isfinite(target).all():
        raise ValueError("target_weights must be a finite one-dimensional array")

    def objective(weights: np.ndarray) -> float:
        difference = weights - target
        return float(difference @ difference)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return 2.0 * (weights - target)

    return objective, gradient


def minimum_variance_objective(
    covariance: np.ndarray,
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """Return objective and gradient for total portfolio variance."""

    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("covariance must be finite")
    matrix = (matrix + matrix.T) / 2.0

    def objective(weights: np.ndarray) -> float:
        return float(weights @ matrix @ weights)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return 2.0 * matrix @ weights

    return objective, gradient


__all__ = ["minimum_variance_objective", "squared_distance_objective"]
