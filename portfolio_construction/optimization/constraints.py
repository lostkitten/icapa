"""Reusable portfolio-constraint specifications and builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LinearConstraintSpec:
    """A named lower and upper bound on a linear portfolio exposure."""

    coefficients: np.ndarray
    lower: float
    upper: float
    name: str


@dataclass(frozen=True)
class NonlinearConstraintSpec:
    """A named lower and upper bound on a scalar portfolio function."""

    function: Callable[[np.ndarray], float]
    lower: float
    upper: float
    name: str
    gradient: Callable[[np.ndarray], np.ndarray] | None = None


def weight_bounds(
    benchmark: Sequence[float],
    *,
    minimum_weight: float = 0.0,
    maximum_weight: float = 1.0,
    capacity_multiple: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build instrument bounds, optionally capped relative to benchmark."""

    benchmark_array = np.asarray(benchmark, dtype=float)
    if benchmark_array.ndim != 1 or not np.isfinite(benchmark_array).all():
        raise ValueError("benchmark weights must be a finite one-dimensional array")
    if minimum_weight < 0 or maximum_weight <= 0 or minimum_weight > maximum_weight:
        raise ValueError("weight bounds must satisfy 0 <= minimum_weight <= maximum_weight")
    if capacity_multiple is not None and capacity_multiple <= 0:
        raise ValueError("capacity_multiple must be positive")

    lower = np.full(len(benchmark_array), float(minimum_weight))
    upper = np.full(len(benchmark_array), float(maximum_weight))
    if capacity_multiple is not None:
        upper = np.minimum(upper, capacity_multiple * benchmark_array)
    if np.any(lower > upper):
        raise ValueError("minimum weights exceed one or more capacity-adjusted maxima")
    return lower, upper


def group_constraint_specs(
    frame: pd.DataFrame,
    benchmark: Sequence[float],
    tolerances: Mapping[str, float] | None,
) -> list[LinearConstraintSpec]:
    """Keep configured group weights close to their benchmark weights."""

    if not tolerances:
        return []
    benchmark_array = np.asarray(benchmark, dtype=float)
    if len(frame) != len(benchmark_array):
        raise ValueError("frame and benchmark weights must have the same length")

    constraints: list[LinearConstraintSpec] = []
    for column, tolerance in tolerances.items():
        if column not in frame:
            raise ValueError(f"group column is missing: {column}")
        if tolerance < 0:
            raise ValueError(f"group tolerance must be non-negative: {column}")
        series = frame[column]
        for value in pd.unique(series):
            mask = (
                series.isna().to_numpy()
                if pd.isna(value)
                else series.eq(value).to_numpy()
            )
            reference_weight = float(benchmark_array[mask].sum())
            constraints.append(
                LinearConstraintSpec(
                    coefficients=mask.astype(float),
                    lower=max(0.0, reference_weight - tolerance),
                    upper=min(1.0, reference_weight + tolerance),
                    name=f"{column}={value!r}",
                )
            )
    return constraints


def turnover_constraint(
    reference_weights: Sequence[float],
    maximum_one_way_turnover: float,
) -> NonlinearConstraintSpec:
    """Limit one-way turnover relative to an explicit reference portfolio."""

    reference = np.asarray(reference_weights, dtype=float)
    if reference.ndim != 1 or not np.isfinite(reference).all():
        raise ValueError("reference_weights must be a finite one-dimensional array")
    if maximum_one_way_turnover < 0:
        raise ValueError("maximum_one_way_turnover must be non-negative")

    return NonlinearConstraintSpec(
        function=lambda weights: 0.5 * float(np.abs(weights - reference).sum()),
        lower=0.0,
        upper=float(maximum_one_way_turnover),
        name="one_way_turnover",
    )


def tracking_error_constraint(
    covariance: np.ndarray,
    benchmark: Sequence[float],
    maximum_tracking_error: float,
) -> NonlinearConstraintSpec:
    """Limit ex-ante tracking error using a supplied covariance matrix."""

    matrix = np.asarray(covariance, dtype=float)
    reference = np.asarray(benchmark, dtype=float)
    if matrix.shape != (len(reference), len(reference)):
        raise ValueError("covariance shape must match benchmark weights")
    if not np.isfinite(matrix).all() or not np.isfinite(reference).all():
        raise ValueError("covariance and benchmark weights must be finite")
    if maximum_tracking_error < 0:
        raise ValueError("maximum_tracking_error must be non-negative")

    def squared_tracking_error(weights: np.ndarray) -> float:
        active = weights - reference
        return float(active @ matrix @ active)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return 2.0 * matrix @ (weights - reference)

    return NonlinearConstraintSpec(
        function=squared_tracking_error,
        lower=0.0,
        upper=float(maximum_tracking_error) ** 2,
        name="tracking_error_squared",
        gradient=gradient,
    )


__all__ = [
    "LinearConstraintSpec",
    "NonlinearConstraintSpec",
    "group_constraint_specs",
    "tracking_error_constraint",
    "turnover_constraint",
    "weight_bounds",
]
