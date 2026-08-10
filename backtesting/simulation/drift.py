"""Explicit constituent-weight drift strategies for daily index simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class WeightDriftStrategy(Protocol):
    """Transform opening weights into closing weights for one business day."""

    name: str
    requires_previous_observation: bool

    def drift(
        self,
        opening_weights: pd.Series,
        observations: pd.DataFrame,
        *,
        business_date: pd.Timestamp,
        previous_observations: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Return finite, non-negative closing weights that sum to one."""


def _selected_observations(
    observations: pd.DataFrame,
    instrument_ids: pd.Index,
    *,
    business_date: pd.Timestamp,
    label: str,
) -> pd.DataFrame:
    missing = instrument_ids.difference(observations.index)
    if len(missing):
        raise ValueError(
            f"{label} is missing held instruments on {business_date.date()}: "
            f"{list(missing)}"
        )
    return observations.loc[instrument_ids]


def _normalize(values: pd.Series, *, business_date: pd.Timestamp) -> pd.Series:
    converted = pd.to_numeric(values, errors="raise").astype(float)
    if (
        not np.isfinite(converted).all()
        or (converted < 0).any()
        or float(converted.sum()) <= 0
    ):
        raise ValueError(f"weight drift is invalid on {business_date.date()}")
    result = converted / float(converted.sum())
    result.index.name = "instrument_id"
    return result


@dataclass(frozen=True, slots=True)
class PriceReturnDrift:
    """Drift weights by the constituent price-return factor."""

    name: str = "price_return"
    requires_previous_observation: bool = False

    def drift(
        self,
        opening_weights: pd.Series,
        observations: pd.DataFrame,
        *,
        business_date: pd.Timestamp,
        previous_observations: pd.DataFrame | None = None,
    ) -> pd.Series:
        selected = _selected_observations(
            observations,
            opening_weights.index,
            business_date=business_date,
            label="daily market data",
        )
        return _normalize(
            opening_weights * (1.0 + selected["price_return"]),
            business_date=business_date,
        )


@dataclass(frozen=True, slots=True)
class CapitalizationDrift:
    """Set held-instrument weights from current absolute market capitalization.

    Each review target is validated against capitalization weights before this
    model is applied. This prevents an arbitrary target from being silently
    replaced by capitalization weights on the first simulated day.

    This is an explicit model. V1 ``WeightDrift.MARKET_CAP`` is adapted to a
    distinct compatibility strategy so old configurations never silently adopt
    future changes to this model.
    """

    target_weight_tolerance: float = 1e-8
    lookback_calendar_days: int = 14
    name: str = "capitalization"
    requires_previous_observation: bool = False

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.target_weight_tolerance)
            or self.target_weight_tolerance < 0
        ):
            raise ValueError("target_weight_tolerance must be finite and non-negative")
        if self.lookback_calendar_days <= 0:
            raise ValueError("lookback_calendar_days must be positive")

    def validate_target(
        self,
        target_weights: pd.Series,
        capitalization_observations: pd.DataFrame | None,
        *,
        business_date: pd.Timestamp,
        portfolio_label: str,
    ) -> None:
        """Require a target to match capitalization weights on its selected set."""

        if capitalization_observations is None:
            raise ValueError(
                f"{portfolio_label} CapitalizationDrift requires a market-cap "
                f"observation before {business_date.date()}"
            )
        selected_ids = target_weights.index[target_weights > 0]
        selected = _selected_observations(
            capitalization_observations,
            selected_ids,
            business_date=business_date,
            label="capitalization initialization data",
        )
        capitalization = pd.to_numeric(
            selected["market_cap"],
            errors="raise",
        ).astype(float)
        if (
            not np.isfinite(capitalization).all()
            or (capitalization <= 0).any()
            or float(capitalization.sum()) <= 0
        ):
            raise ValueError(
                f"{portfolio_label} CapitalizationDrift requires finite, positive "
                f"market caps on {business_date.date()}"
            )
        expected = capitalization / float(capitalization.sum())
        actual = target_weights.loc[selected_ids]
        maximum_difference = float((actual - expected).abs().max())
        if maximum_difference > self.target_weight_tolerance:
            raise ValueError(
                f"{portfolio_label} target is not capitalization weighted on "
                f"{business_date.date()}; maximum absolute difference "
                f"{maximum_difference:.12g} exceeds tolerance "
                f"{self.target_weight_tolerance:.12g}"
            )

    def drift(
        self,
        opening_weights: pd.Series,
        observations: pd.DataFrame,
        *,
        business_date: pd.Timestamp,
        previous_observations: pd.DataFrame | None = None,
    ) -> pd.Series:
        selected = _selected_observations(
            observations,
            opening_weights.index,
            business_date=business_date,
            label="daily market data",
        )
        values = selected["market_cap"].where(opening_weights > 0, 0.0)
        return _normalize(values, business_date=business_date)


@dataclass(frozen=True, slots=True)
class RelativeCapitalizationDrift:
    """Drift prior weights by each constituent's relative capitalization change."""

    lookback_calendar_days: int = 14
    name: str = "relative_capitalization"
    requires_previous_observation: bool = True

    def __post_init__(self) -> None:
        if self.lookback_calendar_days <= 0:
            raise ValueError("lookback_calendar_days must be positive")

    def drift(
        self,
        opening_weights: pd.Series,
        observations: pd.DataFrame,
        *,
        business_date: pd.Timestamp,
        previous_observations: pd.DataFrame | None = None,
    ) -> pd.Series:
        if previous_observations is None:
            raise ValueError(
                "RelativeCapitalizationDrift requires a prior market-cap observation "
                f"before {business_date.date()}"
            )
        current = _selected_observations(
            observations,
            opening_weights.index,
            business_date=business_date,
            label="daily market data",
        )
        previous = _selected_observations(
            previous_observations,
            opening_weights.index,
            business_date=business_date,
            label="previous daily market data",
        )
        previous_caps = pd.to_numeric(previous["market_cap"], errors="raise")
        current_caps = pd.to_numeric(current["market_cap"], errors="raise")
        if (previous_caps <= 0).any():
            invalid = list(previous_caps.index[previous_caps <= 0])
            raise ValueError(
                "RelativeCapitalizationDrift requires positive prior market caps; "
                f"invalid instruments: {invalid}"
            )
        ratios = current_caps / previous_caps
        return _normalize(
            opening_weights * ratios,
            business_date=business_date,
        )


@dataclass(frozen=True, slots=True)
class AbsoluteMarketCapCompatibilityDrift:
    """Explicit adapter for the historical absolute-market-cap behavior."""

    name: str = "absolute_market_cap_compatibility"
    requires_previous_observation: bool = False

    def drift(
        self,
        opening_weights: pd.Series,
        observations: pd.DataFrame,
        *,
        business_date: pd.Timestamp,
        previous_observations: pd.DataFrame | None = None,
    ) -> pd.Series:
        selected = _selected_observations(
            observations,
            opening_weights.index,
            business_date=business_date,
            label="daily market data",
        )
        return _normalize(
            selected["market_cap"].where(opening_weights > 0, 0.0),
            business_date=business_date,
        )


@dataclass(frozen=True, slots=True)
class LegacyAbsoluteMarketCapDrift(AbsoluteMarketCapCompatibilityDrift):
    """Retain the v1 class and identity for cache replay."""

    name: str = "legacy_absolute_market_cap"


__all__ = [
    "AbsoluteMarketCapCompatibilityDrift",
    "CapitalizationDrift",
    "LegacyAbsoluteMarketCapDrift",
    "PriceReturnDrift",
    "RelativeCapitalizationDrift",
    "WeightDriftStrategy",
]
