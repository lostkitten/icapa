"""Data coverage and freshness analytics plugins."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from ..contracts import AnalyticsDiagnostic, AnalyticsValidationError
from .api import AnalyticsContext, AnalyticsPluginResult, MissingAnalyticsInput
from .support_freshness import freshness_records_from_reviews, normalise_freshness_input

@dataclass(frozen=True, slots=True)
class _DataCoveragePlugin:
    plugin_id: str = "data_coverage"
    version: str = "1"
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset(
        {"data_coverage.reviews", "data_coverage.simulation"}
    )

    def run(
        self,
        context: AnalyticsContext,
        parameters: Mapping[str, Any],
    ) -> AnalyticsPluginResult:
        del parameters
        reviews = getattr(context.backtest_result, "reviews", None)
        if not isinstance(reviews, Mapping):
            raise AnalyticsValidationError(
                "backtest_result.reviews must be a mapping"
            )
        review_rows: list[dict[str, Any]] = []
        for effective_date, review in sorted(reviews.items()):
            frame = getattr(review, "cons", None)
            if not isinstance(frame, pd.DataFrame):
                continue
            cells = max(frame.shape[0] * frame.shape[1], 1)
            review_rows.append(
                {
                    "effective_date": pd.Timestamp(effective_date).normalize(),
                    "rows": len(frame),
                    "columns": len(frame.columns),
                    "missing_cells": int(frame.isna().sum().sum()),
                    "missing_fraction": float(frame.isna().sum().sum() / cells),
                }
            )
        simulation_rows: list[dict[str, Any]] = []
        if context.simulation_result is not None:
            daily = getattr(context.simulation_result, "daily", None)
            if isinstance(daily, pd.DataFrame) and not daily.empty:
                simulation_rows.append(
                    {
                        "start_date": pd.Timestamp(daily.index.min()).normalize(),
                        "end_date": pd.Timestamp(daily.index.max()).normalize(),
                        "business_days": len(daily),
                        "missing_cells": int(daily.isna().sum().sum()),
                    }
                )
        return AnalyticsPluginResult(
            tables={
                "reviews": pd.DataFrame(review_rows),
                "simulation": pd.DataFrame(simulation_rows),
            }
        )


@dataclass(frozen=True, slots=True)
class _DataFreshnessPlugin:
    plugin_id: str = "data_freshness"
    version: str = "1"
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset({"data_freshness.sources"})

    def run(
        self,
        context: AnalyticsContext,
        parameters: Mapping[str, Any],
    ) -> AnalyticsPluginResult:
        requested = tuple(
            str(item) for item in parameters.get("date_fields", ())
        )
        explicit = context.research_inputs.freshness_data
        if explicit is not None:
            records = normalise_freshness_input(explicit)
        else:
            records = freshness_records_from_reviews(
                context.backtest_result,
                requested,
            )
        if records.empty:
            raise MissingAnalyticsInput(
                "point-in-time source observation dates are unavailable"
            )
        summary_rows: list[dict[str, Any]] = []
        diagnostics: list[AnalyticsDiagnostic] = []
        keys = ["effective_date", "reference_date", "source"]
        for key, group in records.groupby(keys, dropna=False, sort=True):
            effective_date, reference_date, source = key
            observations = group["observation_date"].dropna()
            future_count = int(
                (observations > pd.Timestamp(reference_date)).sum()
            )
            ages = (
                pd.Timestamp(reference_date) - observations
            ).dt.days
            summary_rows.append(
                {
                    "effective_date": effective_date,
                    "reference_date": reference_date,
                    "source": source,
                    "row_count": int(len(group)),
                    "available_count": int(len(observations)),
                    "missing_count": int(group["observation_date"].isna().sum()),
                    "future_observation_count": future_count,
                    "oldest_observation_date": (
                        observations.min() if not observations.empty else pd.NaT
                    ),
                    "latest_observation_date": (
                        observations.max() if not observations.empty else pd.NaT
                    ),
                    "maximum_age_days": (
                        int(ages.max()) if not ages.empty else pd.NA
                    ),
                    "median_age_days": (
                        float(ages.median()) if not ages.empty else np.nan
                    ),
                }
            )
            if future_count:
                diagnostics.append(
                    AnalyticsDiagnostic(
                        level="warning",
                        code="future_dated_research_input",
                        message=(
                            f"{source} contains {future_count} observation(s) "
                            f"after the {pd.Timestamp(reference_date).date()} "
                            "review cutoff"
                        ),
                    )
                )
        return AnalyticsPluginResult(
            tables={"sources": pd.DataFrame(summary_rows)},
            diagnostics=tuple(diagnostics),
        )
