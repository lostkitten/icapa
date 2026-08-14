"""Generic multiplicative factor-tilt portfolio engine.

Standardized (cross-sectionally z-scored) factor exposures are combined
into a composite factor score; benchmark weights are tilted by a monotone
non-negative multiplier of that score and projected onto the constrained
simplex by Euclidean (least-squares) projection.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Mapping

import numpy as np
from scipy.special import ndtr

from icapa.portfolio_construction.optimization import (
    OptimizationProblem,
    PortfolioSolver,
    ScipySLSQPSolver,
    group_constraint_specs,
    squared_distance_objective,
    weight_bounds,
)
from icapa.portfolio_construction.recipes.factors import (
    factor_output_name,
)


class TiltScheme(str, Enum):
    """Select the monotone map from composite factor scores to tilt multipliers."""

    EXPONENTIAL = "exponential"
    POWER = "power"
    PROPORTIONAL = "proportional"


@dataclass
class FactorTiltEngine:
    """Apply monotone multiplicative factor tilts and project onto generic constraints."""

    factor_tilts: Mapping[str, float]
    tilt_scheme: TiltScheme = TiltScheme.EXPONENTIAL
    tilt_strength: float = 1.0
    score_clip: float = 20.0
    input_weight_column: str = "benchmark_weight"
    output_weight_column: str = "index_weight"
    factor_score_column: str = "factor_score"
    factor_suffix: str = "_zscore"
    minimum_weight: float = 0.0
    maximum_weight: float = 1.0
    capacity_multiple: float | None = None
    group_tolerances: Mapping[str, float] = dataclass_field(default_factory=dict)
    max_iterations: int = 1_000
    tolerance: float = 1e-9
    solver: PortfolioSolver | None = dataclass_field(default=None, repr=False)

    def execute(self, data_context):
        if not self.factor_tilts:
            raise ValueError(
                "factor_tilts must map at least one factor field to a tilt coefficient"
            )
        if self.score_clip <= 0:
            raise ValueError(
                "score_clip must be a strictly positive bound on the composite factor score"
            )
        strength = float(self.tilt_strength)
        if not np.isfinite(strength) or strength <= 0:
            raise ValueError("tilt_strength must be a finite, strictly positive scalar")
        scheme = TiltScheme(self.tilt_scheme)
        factor_columns = [
            factor_output_name(field, self.factor_suffix) for field in self.factor_tilts
        ]
        columns = list(
            dict.fromkeys(
                [self.input_weight_column, *factor_columns, *self.group_tolerances]
            )
        )
        frame = data_context.get_dataframe(
            columns, include_excluded_instruments=False
        )
        if frame.empty or not frame.index.is_unique:
            raise ValueError("the investable universe must be non-empty with a unique index")

        benchmark = frame[self.input_weight_column].to_numpy(dtype=float)
        if not np.isfinite(benchmark).all() or np.any(benchmark < 0) or benchmark.sum() <= 0:
            raise ValueError("benchmark weights must be finite, non-negative, and non-zero")
        benchmark = benchmark / benchmark.sum()

        score = np.zeros(len(frame), dtype=float)
        for (field, coefficient), column in zip(self.factor_tilts.items(), factor_columns):
            if not np.isfinite(float(coefficient)):
                raise ValueError(f"tilt coefficient must be finite: {field}")
            values = frame[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"standardized factor contains non-finite values: {field}"
                )
            score += float(coefficient) * values
        score = np.clip(score, -self.score_clip, self.score_clip)
        desired = benchmark * self._tilt_multipliers(scheme, strength, score)
        if desired.sum() <= 0 or not np.isfinite(desired).all():
            raise ValueError(
                "the multiplicative tilt produced no investable mass: "
                "tilted weights must be finite with positive total weight"
            )
        desired /= desired.sum()

        lower, upper = weight_bounds(
            benchmark,
            minimum_weight=self.minimum_weight,
            maximum_weight=self.maximum_weight,
            capacity_multiple=self.capacity_multiple,
        )
        constraints = group_constraint_specs(
            frame, benchmark, self.group_tolerances
        )
        objective, gradient = squared_distance_objective(desired)
        solution = (
            self.solver
            or ScipySLSQPSolver(
                tolerance=self.tolerance,
                max_iterations=self.max_iterations,
            )
        ).solve(
            OptimizationProblem(
                name="factor_tilt",
                objective=objective,
                initial_weights=desired,
                lower_bounds=lower,
                upper_bounds=upper,
                gradient=gradient,
                linear_constraints=constraints,
            )
        )
        weights = solution.weights
        result = frame.iloc[:, 0:0].copy()
        result[self.factor_score_column] = score
        result[self.output_weight_column] = weights
        data_context.set_dataframe(
            result,
            columns=[self.factor_score_column, self.output_weight_column],
        )
        diagnostics = solution.as_dict()
        data_context.diagnostics["factor_tilt_optimization"] = diagnostics
        data_context.diagnostics["factor_tilt_optimisation"] = diagnostics
        return data_context

    def _tilt_multipliers(self, scheme, strength, score):
        """Map the clipped composite score to non-negative tilt multipliers."""

        if scheme is TiltScheme.EXPONENTIAL:
            return np.exp(strength * score)
        if scheme is TiltScheme.POWER:
            return ndtr(score) ** strength
        return np.maximum(1.0 + strength * score, 0.0)


__all__ = ["FactorTiltEngine", "TiltScheme"]
