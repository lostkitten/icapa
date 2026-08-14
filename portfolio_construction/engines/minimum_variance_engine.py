"""Generic minimum-variance portfolio engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from icapa.portfolio_construction.optimization import (
    CovarianceEstimator,
    NonlinearConstraintSpec,
    OptimizationProblem,
    PortfolioSolver,
    ReturnWindowSpec,
    SampleCovarianceEstimator,
    ScipySLSQPSolver,
    estimate_covariance_for_window,
    group_constraint_specs,
    minimum_variance_objective,
    tracking_error_constraint,
    weight_bounds,
)


@dataclass
class MinimumVarianceEngine:
    """Estimate a covariance matrix and minimize total portfolio variance."""

    return_column: str = "net_total_return"
    input_weight_column: str = "benchmark_weight"
    output_weight_column: str = "index_weight"
    minimum_observations: int = 20
    covariance_ridge: float = 1e-8
    return_window: ReturnWindowSpec | None = None
    covariance_estimator: CovarianceEstimator | None = field(
        default=None,
        repr=False,
    )
    minimum_weight: float = 0.0
    maximum_weight: float = 1.0
    capacity_multiple: float | None = None
    group_tolerances: Mapping[str, float] = field(default_factory=dict)
    maximum_tracking_error: float | None = None
    maximum_one_way_turnover: float | None = None
    turnover_reference_column: str | None = None
    max_iterations: int = 1_000
    tolerance: float = 1e-9
    solver: PortfolioSolver | None = field(default=None, repr=False)

    def execute(self, data_context):
        if self.minimum_observations < 2:
            raise ValueError("minimum_observations must be at least two")
        if self.covariance_ridge < 0:
            raise ValueError("covariance_ridge must be non-negative")
        frame, benchmark = self._investable_frame(data_context)

        daily = getattr(data_context, "daily", None)
        if daily is None or daily.empty:
            raise ValueError("daily return data must be loaded before minimum variance runs")
        daily_frame = daily.reset_index() if isinstance(daily.index, pd.MultiIndex) else daily.copy()
        required = {"instrument_id", "business_date", self.return_column}
        missing = sorted(required.difference(daily_frame.columns))
        if missing:
            raise ValueError(f"daily return data is missing columns: {missing}")
        daily_frame["business_date"] = pd.to_datetime(
            daily_frame["business_date"], errors="raise"
        )
        if daily_frame.duplicated(["instrument_id", "business_date"]).any():
            raise ValueError("daily return data contains duplicate instrument-date rows")
        daily_frame[self.return_column] = pd.to_numeric(
            daily_frame[self.return_column], errors="coerce"
        )
        returns = daily_frame.pivot(
            index="business_date", columns="instrument_id", values=self.return_column
        ).reindex(columns=frame.index)
        window = self.return_window or ReturnWindowSpec(
            lookback=252,
            minimum_observations=self.minimum_observations,
        )
        estimator = self.covariance_estimator or SampleCovarianceEstimator(
            minimum_observations=self.minimum_observations,
            ridge=self.covariance_ridge,
        )
        resolved_window, covariance_estimate = estimate_covariance_for_window(
            estimator,
            returns,
            window,
            data_context.reference_date,
        )
        return self._solve_with_covariance(
            data_context,
            frame,
            benchmark,
            covariance_estimate.matrix,
            risk_diagnostics={
                "observations": covariance_estimate.observations,
                "return_window": {
                    "start_date": resolved_window.start_date,
                    "end_date": resolved_window.end_date,
                    "observation_count": resolved_window.observation_count,
                    "kind": window.kind.value,
                    "lookback": window.lookback,
                },
                "covariance_estimator": covariance_estimate.estimator_name,
                "covariance_metadata": dict(covariance_estimate.metadata),
            },
        )

    def execute_with_covariance(
        self,
        data_context,
        covariance,
        *,
        risk_diagnostics: Mapping[str, object] | None = None,
    ):
        """Solve from a precomputed point-in-time covariance artifact."""

        frame, benchmark = self._investable_frame(data_context)
        return self._solve_with_covariance(
            data_context,
            frame,
            benchmark,
            covariance,
            risk_diagnostics=dict(risk_diagnostics or {}),
        )

    @staticmethod
    def _turnover_constraint_spec(
        reference: np.ndarray,
        maximum_one_way_turnover: float,
    ) -> NonlinearConstraintSpec:
        """Bound one-way turnover with a kink-free surrogate of the L1 norm.

        SLSQP stalls on the non-differentiable |w - reference| at the kinks,
        so the constraint uses 0.5 * sum(sqrt(delta^2 + eps^2) - eps) with an
        analytic gradient; it under-estimates true turnover by at most
        0.5 * n * eps, which is negligible at eps = 1e-9.
        """

        epsilon = 1e-9

        def function(weights: np.ndarray) -> float:
            delta = np.asarray(weights, dtype=float) - reference
            smoothed = np.sqrt(delta * delta + epsilon * epsilon) - epsilon
            return 0.5 * float(smoothed.sum())

        def gradient(weights: np.ndarray) -> np.ndarray:
            delta = np.asarray(weights, dtype=float) - reference
            return 0.5 * delta / np.sqrt(delta * delta + epsilon * epsilon)

        return NonlinearConstraintSpec(
            function=function,
            lower=0.0,
            upper=float(maximum_one_way_turnover),
            name="one_way_turnover",
            gradient=gradient,
        )

    def _investable_frame(self, data_context):
        if (self.maximum_one_way_turnover is None) != (
            self.turnover_reference_column is None
        ):
            raise ValueError(
                "maximum_one_way_turnover and turnover_reference_column "
                "must be supplied together"
            )
        for limit_name in ("maximum_tracking_error", "maximum_one_way_turnover"):
            limit = getattr(self, limit_name)
            if limit is not None and (
                not np.isfinite(float(limit)) or float(limit) < 0
            ):
                raise ValueError(
                    f"{limit_name} must be a finite, non-negative scalar"
                )
        reference_columns = (
            [self.turnover_reference_column] if self.turnover_reference_column else []
        )
        columns = list(
            dict.fromkeys(
                [self.input_weight_column, *reference_columns, *self.group_tolerances]
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
        return frame, benchmark

    def _solve_with_covariance(
        self,
        data_context,
        frame,
        benchmark,
        covariance,
        *,
        risk_diagnostics: Mapping[str, object],
    ):
        covariance = np.asarray(covariance, dtype=float)
        if (
            covariance.shape != (len(frame), len(frame))
            or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0.0)
        ):
            raise ValueError(
                "covariance must be a finite symmetric matrix aligned to the universe"
            )

        lower, upper = weight_bounds(
            benchmark,
            minimum_weight=self.minimum_weight,
            maximum_weight=self.maximum_weight,
            capacity_multiple=self.capacity_multiple,
        )
        constraints = group_constraint_specs(
            frame, benchmark, self.group_tolerances
        )
        # SLSQP's stopping tolerance is absolute; normalize the objective to
        # O(1) so tiny daily-return variances still terminate meaningfully.
        objective_scale = float(benchmark @ covariance @ benchmark)
        if not objective_scale > 0:
            objective_scale = float(np.trace(covariance)) / max(len(benchmark), 1)
        if not objective_scale > 0:
            objective_scale = 1.0

        nonlinear_constraints = []
        if self.maximum_tracking_error is not None:
            nonlinear_constraints.append(
                tracking_error_constraint(
                    covariance / objective_scale,
                    benchmark,
                    float(self.maximum_tracking_error) / np.sqrt(objective_scale),
                )
            )
        turnover_reference = None
        if self.maximum_one_way_turnover is not None:
            turnover_reference = frame[self.turnover_reference_column].to_numpy(
                dtype=float
            )
            if not np.isfinite(turnover_reference).all() or np.any(
                turnover_reference < 0
            ):
                raise ValueError(
                    "turnover reference weights must be finite and non-negative"
                )
            nonlinear_constraints.append(
                self._turnover_constraint_spec(
                    turnover_reference,
                    float(self.maximum_one_way_turnover),
                )
            )
        objective, gradient = minimum_variance_objective(covariance / objective_scale)
        solution = (
            self.solver
            or ScipySLSQPSolver(
                tolerance=self.tolerance,
                max_iterations=self.max_iterations,
            )
        ).solve(
            OptimizationProblem(
                name="minimum_variance",
                objective=objective,
                initial_weights=benchmark,
                lower_bounds=lower,
                upper_bounds=upper,
                gradient=gradient,
                linear_constraints=constraints,
                nonlinear_constraints=tuple(nonlinear_constraints),
            )
        )
        weights = solution.weights
        result = frame.iloc[:, 0:0].copy()
        result[self.output_weight_column] = weights
        data_context.set_dataframe(result, columns=[self.output_weight_column])
        active = weights - benchmark
        diagnostics = {
            **solution.as_dict(),
            "objective_scale": objective_scale,
            "benchmark_variance": float(benchmark @ covariance @ benchmark),
            "portfolio_variance": float(weights @ covariance @ weights),
            "tracking_error": float(
                np.sqrt(max(float(active @ covariance @ active), 0.0))
            ),
            **dict(risk_diagnostics),
        }
        if turnover_reference is not None:
            diagnostics["one_way_turnover"] = 0.5 * float(
                np.abs(weights - turnover_reference).sum()
            )
        data_context.diagnostics["minimum_variance_optimization"] = diagnostics
        data_context.diagnostics["minimum_variance_optimisation"] = diagnostics
        return data_context


__all__ = ["MinimumVarianceEngine"]
