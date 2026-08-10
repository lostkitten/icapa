"""Immutable-ish data contracts for the analytics layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


class AnalyticsValidationError(ValueError):
    """Raised when an analytics input violates a required data contract."""


@dataclass(frozen=True, slots=True)
class AnalyticsDiagnostic:
    """One non-fatal observation made while producing analytics."""

    level: Literal["info", "warning"]
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class BrinsonInput:
    """Explicit, pre-aligned inputs for single-period Brinson attribution.

    ``data`` contains one row per instrument and attribution period. Column
    names are configurable so callers can adapt existing result tables without
    changing the analytics implementation.
    """

    data: pd.DataFrame
    period_column: str = "period"
    instrument_column: str = "instrument_id"
    classification_column: str = "industry"
    portfolio_weight_column: str = "index_weight"
    benchmark_weight_column: str = "benchmark_weight"
    return_column: str = "asset_return"

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", self.data.copy(deep=True))
        names = (
            self.period_column,
            self.instrument_column,
            self.classification_column,
            self.portfolio_weight_column,
            self.benchmark_weight_column,
            self.return_column,
        )
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise AnalyticsValidationError("Brinson column names must be non-empty strings")


@dataclass(frozen=True, slots=True)
class BrinsonAttribution:
    """Detailed and period-total Brinson-Fachler attribution."""

    detail: pd.DataFrame
    totals: pd.DataFrame

    def __post_init__(self) -> None:
        object.__setattr__(self, "detail", self.detail.copy(deep=True))
        object.__setattr__(self, "totals", self.totals.copy(deep=True))


@dataclass(frozen=True, slots=True)
class ResearchAnalyticsInputs:
    """Optional, explicitly aligned inputs for research-only analytics.

    The plugin runner can derive several diagnostics from review contexts. This
    bundle is the extension point for data that is not part of the v1
    ``BacktestResult`` compatibility contract:

    - ``selection_reasons``: decision records by effective date and instrument;
    - ``target_diagnostics``: requested and achieved target values;
    - ``constraint_diagnostics``: constraint values, bounds, and solver status;
    - ``factor_signal_data``: instrument-level factor or signal observations;
    - ``liquidity_capacity_data``: instrument-level liquidity or capacity data;
    - ``freshness_data``: source observation dates and their review cutoffs;
    - ``attribution_input``: pre-aligned multi-period Brinson input.

    DataFrames are copied on construction. Plugins validate their required
    columns and never infer missing observations as zero.
    """

    selection_reasons: pd.DataFrame | None = None
    target_diagnostics: pd.DataFrame | None = None
    constraint_diagnostics: pd.DataFrame | None = None
    factor_signal_data: pd.DataFrame | None = None
    liquidity_capacity_data: pd.DataFrame | None = None
    freshness_data: pd.DataFrame | None = None
    attribution_input: BrinsonInput | None = None

    def __post_init__(self) -> None:
        for name in (
            "selection_reasons",
            "target_diagnostics",
            "constraint_diagnostics",
            "factor_signal_data",
            "liquidity_capacity_data",
            "freshness_data",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, pd.DataFrame):
                raise AnalyticsValidationError(
                    f"{name} must be a pandas DataFrame or None"
                )
            if value is not None:
                object.__setattr__(self, name, value.copy(deep=True))
        if (
            self.attribution_input is not None
            and not isinstance(self.attribution_input, BrinsonInput)
        ):
            raise AnalyticsValidationError(
                "attribution_input must be a BrinsonInput or None"
            )
        if self.attribution_input is not None:
            value = self.attribution_input
            object.__setattr__(
                self,
                "attribution_input",
                BrinsonInput(
                    data=value.data,
                    period_column=value.period_column,
                    instrument_column=value.instrument_column,
                    classification_column=value.classification_column,
                    portfolio_weight_column=value.portfolio_weight_column,
                    benchmark_weight_column=value.benchmark_weight_column,
                    return_column=value.return_column,
                ),
            )


@dataclass(frozen=True, slots=True)
class AnalyticsResult:
    """All analytics generated from one backtest and optional simulation."""

    review_validation: pd.DataFrame
    review_metrics: pd.DataFrame
    country_exposures: pd.DataFrame
    industry_exposures: pd.DataFrame
    target_review_weight_change: pd.DataFrame
    formal_turnover: pd.DataFrame
    performance: pd.Series
    drawdowns: pd.DataFrame
    brinson: BrinsonAttribution | None
    diagnostics: tuple[AnalyticsDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "review_validation",
            "review_metrics",
            "country_exposures",
            "industry_exposures",
            "target_review_weight_change",
            "drawdowns",
        ):
            object.__setattr__(self, name, getattr(self, name).copy(deep=True))
        object.__setattr__(
            self,
            "formal_turnover",
            _with_turnover_aliases(self.formal_turnover),
        )
        object.__setattr__(self, "performance", self.performance.copy(deep=True))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def one_way_turnover(self) -> pd.DataFrame:
        """Return the canonical one-way turnover table."""

        return self.formal_turnover.copy(deep=True)

    @property
    def has_warnings(self) -> bool:
        """Whether the result contains one or more warning diagnostics."""

        return any(item.level == "warning" for item in self.diagnostics)

    def tables(self) -> dict[str, pd.DataFrame | pd.Series]:
        """Return defensive copies of tabular outputs for a report renderer."""

        result: dict[str, pd.DataFrame | pd.Series] = {
            "review_validation": self.review_validation.copy(deep=True),
            "review_metrics": self.review_metrics.copy(deep=True),
            "country_exposures": self.country_exposures.copy(deep=True),
            "industry_exposures": self.industry_exposures.copy(deep=True),
            "target_review_weight_change": (
                self.target_review_weight_change.copy(deep=True)
            ),
            "one_way_turnover": self.one_way_turnover,
            "formal_turnover": self.formal_turnover.copy(deep=True),
            "performance": self.performance.copy(deep=True),
            "drawdowns": self.drawdowns.copy(deep=True),
        }
        if self.brinson is not None:
            result["brinson_detail"] = self.brinson.detail.copy(deep=True)
            result["brinson_totals"] = self.brinson.totals.copy(deep=True)
        return result


def _with_turnover_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose canonical and historical turnover columns side by side."""

    result = frame.copy(deep=True)
    if "one_way_turnover" not in result:
        source = next(
            (
                name
                for name in (
                    "formal_one_way_turnover",
                    "index_one_way_turnover",
                    "index_turnover",
                    "formal_turnover",
                    "turnover",
                )
                if name in result
            ),
            None,
        )
        result["one_way_turnover"] = (
            result[source]
            if source is not None
            else pd.Series(index=result.index, dtype=float)
        )
    if "formal_one_way_turnover" not in result:
        result["formal_one_way_turnover"] = result["one_way_turnover"]
    return result
