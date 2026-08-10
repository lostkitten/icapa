"""Constituent membership, reason, and turnover analytics plugins."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from .api import AnalyticsContext, AnalyticsPluginResult, MissingAnalyticsInput
from .support_reviews import (normalise_reason_input, reason_values, review_frames, review_weights)

@dataclass(frozen=True, slots=True)
class _ConstituentChangePlugin:
    plugin_id: str = "constituent_change"
    version: str = "1"
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset(
        {
            "constituent_change.detail",
            "constituent_change.membership_stability",
        }
    )

    def run(
        self,
        context: AnalyticsContext,
        parameters: Mapping[str, Any],
    ) -> AnalyticsPluginResult:
        del parameters
        weights = review_weights(context.backtest_result)
        dates = list(weights.index.get_level_values("effective_date").unique())
        detail_rows: list[dict[str, Any]] = []
        stability_rows: list[dict[str, Any]] = []
        tolerance = context.spec.weight_tolerance
        for previous_date, effective_date in zip(dates, dates[1:]):
            previous = weights.xs(previous_date, level="effective_date")
            current = weights.xs(effective_date, level="effective_date")
            previous, current = previous.align(current, join="outer", fill_value=0.0)
            previous_members = set(previous[previous > tolerance].index)
            current_members = set(current[current > tolerance].index)
            union = previous_members | current_members
            intersection = previous_members & current_members
            stability_rows.append(
                {
                    "effective_date": effective_date,
                    "previous_effective_date": previous_date,
                    "previous_count": len(previous_members),
                    "current_count": len(current_members),
                    "entrants": len(current_members - previous_members),
                    "exits": len(previous_members - current_members),
                    "membership_jaccard": (
                        len(intersection) / len(union) if union else 1.0
                    ),
                }
            )
            for instrument_id in previous.index:
                previous_weight = float(previous.loc[instrument_id])
                current_weight = float(current.loc[instrument_id])
                if previous_weight <= tolerance < current_weight:
                    status = "entrant"
                elif current_weight <= tolerance < previous_weight:
                    status = "exit"
                elif current_weight > previous_weight + tolerance:
                    status = "weight_increase"
                elif current_weight < previous_weight - tolerance:
                    status = "weight_decrease"
                else:
                    status = "unchanged"
                detail_rows.append(
                    {
                        "effective_date": effective_date,
                        "previous_effective_date": previous_date,
                        "instrument_id": instrument_id,
                        "previous_weight": previous_weight,
                        "current_weight": current_weight,
                        "weight_change": current_weight - previous_weight,
                        "status": status,
                    }
                )
        detail_columns = (
            "effective_date",
            "previous_effective_date",
            "instrument_id",
            "previous_weight",
            "current_weight",
            "weight_change",
            "status",
        )
        stability_columns = (
            "effective_date",
            "previous_effective_date",
            "previous_count",
            "current_count",
            "entrants",
            "exits",
            "membership_jaccard",
        )
        return AnalyticsPluginResult(
            tables={
                "detail": pd.DataFrame(detail_rows, columns=detail_columns),
                "membership_stability": pd.DataFrame(
                    stability_rows,
                    columns=stability_columns,
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class _SelectionReasonsPlugin:
    plugin_id: str = "selection_reasons"
    version: str = "1"
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset(
        {"selection_reasons.detail", "selection_reasons.summary"}
    )

    def run(
        self,
        context: AnalyticsContext,
        parameters: Mapping[str, Any],
    ) -> AnalyticsPluginResult:
        del parameters
        explicit = context.research_inputs.selection_reasons
        if explicit is not None:
            detail = normalise_reason_input(explicit)
        else:
            rows: list[dict[str, Any]] = []
            for effective_date, _, frame in review_frames(
                context.backtest_result
            ):
                for column, decision in (
                    ("selection_reason", "selected"),
                    ("exclusion_reason", "excluded"),
                ):
                    if column not in frame.columns:
                        continue
                    for instrument_id, raw_reasons in frame[column].items():
                        for reason in reason_values(raw_reasons):
                            rows.append(
                                {
                                    "effective_date": effective_date,
                                    "instrument_id": instrument_id,
                                    "decision": decision,
                                    "reason": reason,
                                    "source": column,
                                }
                            )
            detail = pd.DataFrame(
                rows,
                columns=[
                    "effective_date",
                    "instrument_id",
                    "decision",
                    "reason",
                    "source",
                ],
            )
        if detail.empty:
            raise MissingAnalyticsInput(
                "selection and exclusion reason records are unavailable"
            )
        detail = detail.drop_duplicates().sort_values(
            ["effective_date", "decision", "reason", "instrument_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        summary = (
            detail.groupby(
                ["effective_date", "decision", "reason"],
                dropna=False,
                sort=True,
            )
            .agg(instrument_count=("instrument_id", "nunique"))
            .reset_index()
        )
        return AnalyticsPluginResult(
            tables={"detail": detail, "summary": summary}
        )


@dataclass(frozen=True, slots=True)
class _WeightChangeContributorsPlugin:
    plugin_id: str = "weight_change_contributors"
    version: str = "1"
    requires: frozenset[str] = frozenset({"constituent_change.detail"})
    provides: frozenset[str] = frozenset(
        {
            "weight_change_contributors.detail",
            "weight_change_contributors.summary",
        }
    )

    def run(
        self,
        context: AnalyticsContext,
        parameters: Mapping[str, Any],
    ) -> AnalyticsPluginResult:
        del parameters
        changes = context.available_tables["constituent_change.detail"]
        columns = [
            "effective_date",
            "previous_effective_date",
            "instrument_id",
            "status",
            "previous_weight",
            "current_weight",
            "weight_change",
            "absolute_weight_change",
            "one_way_contribution",
            "share_of_one_way_turnover",
        ]
        if changes.empty:
            return AnalyticsPluginResult(
                tables={
                    "detail": pd.DataFrame(columns=columns),
                    "summary": pd.DataFrame(
                        columns=[
                            "effective_date",
                            "status",
                            "instrument_count",
                            "net_weight_change",
                            "absolute_weight_change",
                            "one_way_turnover",
                            "share_of_one_way_turnover",
                        ]
                    ),
                }
            )
        detail = changes.copy(deep=True)
        detail["absolute_weight_change"] = detail["weight_change"].abs()
        detail["one_way_contribution"] = (
            0.5 * detail["absolute_weight_change"]
        )
        totals = detail.groupby("effective_date")[
            "one_way_contribution"
        ].transform("sum")
        detail["share_of_one_way_turnover"] = np.where(
            totals > 0,
            detail["one_way_contribution"] / totals,
            0.0,
        )
        detail = detail.loc[:, columns].sort_values(
            ["effective_date", "one_way_contribution", "instrument_id"],
            ascending=[True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        summary = (
            detail.groupby(
                ["effective_date", "status"],
                dropna=False,
                sort=True,
            )
            .agg(
                instrument_count=("instrument_id", "nunique"),
                net_weight_change=("weight_change", "sum"),
                absolute_weight_change=("absolute_weight_change", "sum"),
                one_way_turnover=("one_way_contribution", "sum"),
                share_of_one_way_turnover=(
                    "share_of_one_way_turnover",
                    "sum",
                ),
            )
            .reset_index()
        )
        return AnalyticsPluginResult(
            tables={"detail": detail, "summary": summary}
        )


@dataclass(frozen=True, slots=True)
class _TurnoverDecompositionPlugin:
    plugin_id: str = "turnover_decomposition"
    version: str = "1"
    requires: frozenset[str] = frozenset({"constituent_change.detail"})
    provides: frozenset[str] = frozenset({"turnover_decomposition.detail"})

    def run(
        self,
        context: AnalyticsContext,
        parameters: Mapping[str, Any],
    ) -> AnalyticsPluginResult:
        del parameters
        changes = context.available_tables["constituent_change.detail"]
        if changes.empty:
            return AnalyticsPluginResult(
                tables={
                    "detail": pd.DataFrame(
                        columns=[
                            "effective_date",
                            "instrument_id",
                            "status",
                            "absolute_weight_change",
                            "one_way_contribution",
                        ]
                    )
                }
            )
        result = changes[
            ["effective_date", "instrument_id", "status", "weight_change"]
        ].copy()
        result["absolute_weight_change"] = result["weight_change"].abs()
        result["one_way_contribution"] = 0.5 * result["absolute_weight_change"]
        return AnalyticsPluginResult(
            tables={
                "detail": result.drop(columns="weight_change").sort_values(
                    ["effective_date", "absolute_weight_change", "instrument_id"],
                    ascending=[True, False, True],
                    kind="mergesort",
                ).reset_index(drop=True)
            }
        )
