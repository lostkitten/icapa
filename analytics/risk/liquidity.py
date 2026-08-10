"""Simple AUM and trading-capacity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class LiquidityCapacitySpec:
    """AUM and participation assumptions for capacity analysis."""

    assets_under_management: float
    participation_rate: float = 0.2
    trading_days: int = 1

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.assets_under_management)
            or self.assets_under_management <= 0
        ):
            raise ValueError(
                "assets_under_management must be finite and positive"
            )
        if (
            not np.isfinite(self.participation_rate)
            or not 0 < self.participation_rate <= 1
        ):
            raise ValueError("participation_rate must be in (0, 1]")
        if self.trading_days <= 0:
            raise ValueError("trading_days must be positive")


@dataclass(frozen=True, slots=True)
class LiquidityCapacityResult:
    """Instrument detail and portfolio-level capacity summary."""

    detail: pd.DataFrame
    summary: pd.Series

    def __post_init__(self) -> None:
        object.__setattr__(self, "detail", self.detail.copy(deep=True))
        object.__setattr__(self, "summary", self.summary.copy(deep=True))


def calculate_liquidity_capacity(
    weights: pd.Series,
    average_daily_value_traded: pd.Series,
    *,
    spec: LiquidityCapacitySpec,
    weight_tolerance: float = 1e-8,
) -> LiquidityCapacityResult:
    """Calculate days-to-trade and capacity-weight breaches."""

    if not isinstance(spec, LiquidityCapacitySpec):
        raise TypeError("spec must be a LiquidityCapacitySpec")
    portfolio = _numeric_series(weights, "weights")
    liquidity = _numeric_series(
        average_daily_value_traded,
        "average_daily_value_traded",
    )
    if set(portfolio.index) != set(liquidity.index):
        raise ValueError(
            "weights and average_daily_value_traded must use the same instruments"
        )
    liquidity = liquidity.reindex(portfolio.index)
    if (portfolio < -weight_tolerance).any():
        raise ValueError("weights must be non-negative")
    total = float(portfolio.sum())
    if not np.isclose(total, 1.0, atol=weight_tolerance, rtol=0.0):
        raise ValueError(f"weights must sum to one; observed {total:.12g}")
    if (liquidity <= 0).any():
        raise ValueError("average_daily_value_traded must be positive")

    allocation = portfolio * spec.assets_under_management
    daily_capacity = liquidity * spec.participation_rate
    capacity_weight_limit = (
        daily_capacity
        * spec.trading_days
        / spec.assets_under_management
    )
    days_to_trade = allocation / daily_capacity
    detail = pd.DataFrame(
        {
            "index_weight": portfolio,
            "average_daily_value_traded": liquidity,
            "allocation_value": allocation,
            "daily_trading_capacity": daily_capacity,
            "days_to_trade": days_to_trade,
            "capacity_weight_limit": capacity_weight_limit,
            "capacity_breach": (
                portfolio > capacity_weight_limit + weight_tolerance
            ),
        },
        index=portfolio.index.copy(),
    )
    detail.index.name = portfolio.index.name or "instrument_id"
    weighted_days = float((portfolio * days_to_trade).sum())
    summary = pd.Series(
        {
            "assets_under_management": spec.assets_under_management,
            "participation_rate": spec.participation_rate,
            "trading_days": float(spec.trading_days),
            "capacity_breach_count": float(detail["capacity_breach"].sum()),
            "weight_in_breach": float(
                detail.loc[detail["capacity_breach"], "index_weight"].sum()
            ),
            "maximum_days_to_trade": float(days_to_trade.max()),
            "weighted_average_days_to_trade": weighted_days,
        },
        dtype=float,
        name="value",
    )
    return LiquidityCapacityResult(detail=detail, summary=summary)


def _numeric_series(value: pd.Series, label: str) -> pd.Series:
    if not isinstance(value, pd.Series) or value.empty:
        raise ValueError(f"{label} must be a non-empty pandas Series")
    if value.index.has_duplicates:
        raise ValueError(f"{label} index must be unique")
    result = pd.to_numeric(value.copy(deep=True), errors="coerce")
    if result.isna().any() or not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} must contain finite numeric values")
    return result.astype(float)
