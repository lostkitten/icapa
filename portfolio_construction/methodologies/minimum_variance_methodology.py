"""Generic minimum-variance methodology."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from icapa.portfolio_construction.optimization import (
    CovarianceEstimator,
    PortfolioSolver,
    ReturnWindowSpec,
)
from icapa.portfolio_construction.rules.data_loading.add_returns import AddReturns
from icapa.portfolio_construction.rules.data_loading.load_universe import (
    LoadUniverse,
)
from icapa.portfolio_construction.engines.minimum_variance_engine import (
    MinimumVarianceEngine,
)


@dataclass
class MinimumVarianceMethodology:
    """Load daily returns, estimate covariance, and minimize portfolio variance.

    ``maximum_tracking_error`` bounds ex-ante tracking error versus the
    benchmark in the same return periodicity as the covariance estimate.
    ``maximum_one_way_turnover`` bounds one-way turnover against the explicit
    ``turnover_reference_column`` weights; the two must be supplied together.
    """

    universe_id: str
    universe_provider_name: str
    returns_provider_name: str
    index_id: str | None = None
    universe_provider_parameters: dict = field(default_factory=dict)
    returns_provider_parameters: dict = field(default_factory=dict)
    start_date: str | None = None
    end_date: str | None = None
    return_column: str = "net_total_return"
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
    solver: PortfolioSolver | None = field(default=None, repr=False)

    def to_recipe(self):
        """Return the cacheable data/risk/weighting recipe preset."""

        from .recipe_presets import minimum_variance_recipe_preset

        return minimum_variance_recipe_preset(self)

    def execute(self, data_context):
        if not self.universe_id:
            raise ValueError("universe_id is required")
        if not self.returns_provider_name:
            raise ValueError("returns_provider_name is required")
        if self.index_id is not None:
            data_context.index_id = self.index_id
        LoadUniverse(
            universe_id=self.universe_id,
            provider_name=self.universe_provider_name,
            provider_parameters=dict(self.universe_provider_parameters),
        ).execute(data_context)
        AddReturns(
            provider_name=self.returns_provider_name,
            start_date=self.start_date,
            end_date=self.end_date,
            provider_parameters=dict(self.returns_provider_parameters),
        ).execute(data_context)
        MinimumVarianceEngine(
            return_column=self.return_column,
            minimum_observations=self.minimum_observations,
            covariance_ridge=self.covariance_ridge,
            return_window=self.return_window,
            covariance_estimator=self.covariance_estimator,
            minimum_weight=self.minimum_weight,
            maximum_weight=self.maximum_weight,
            capacity_multiple=self.capacity_multiple,
            group_tolerances=self.group_tolerances,
            maximum_tracking_error=self.maximum_tracking_error,
            maximum_one_way_turnover=self.maximum_one_way_turnover,
            turnover_reference_column=self.turnover_reference_column,
            solver=self.solver,
        ).execute(data_context)
        return data_context


__all__ = ["MinimumVarianceMethodology"]
