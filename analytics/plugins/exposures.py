"""Factor, signal, liquidity, and capacity analytics plugins."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import pandas as pd
from ..contracts import AnalyticsValidationError
from .api import AnalyticsContext, AnalyticsPluginResult, MissingAnalyticsInput
from .support_exposures import (CAPACITY_LIMIT_FIELDS, LIQUIDITY_FIELDS, capacity_table, exposure_type, instrument_research_frame, is_factor_or_signal_field, weighted_field_statistics)

@dataclass(frozen=True, slots=True)
class _FactorSignalExposurePlugin:
    plugin_id: str = "factor_signal_exposure"
    version: str = "1"
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset(
        {"factor_signal_exposure.exposures"}
    )

    def run(
        self,
        context: AnalyticsContext,
        parameters: Mapping[str, Any],
    ) -> AnalyticsPluginResult:
        frame = instrument_research_frame(
            context.backtest_result,
            context.research_inputs.factor_signal_data,
            input_name="factor_signal_data",
        )
        requested = tuple(str(item) for item in parameters.get("fields", ()))
        if requested:
            missing = sorted(set(requested).difference(frame.columns))
            if missing:
                raise AnalyticsValidationError(
                    f"factor/signal exposure fields are missing: {missing}"
                )
            fields = requested
        else:
            fields = tuple(
                column
                for column in frame.columns
                if is_factor_or_signal_field(column)
                and pd.api.types.is_numeric_dtype(frame[column])
            )
        if not fields:
            raise MissingAnalyticsInput(
                "factor or signal fields are unavailable; supply fields explicitly "
                "or use canonical factor/signal column names"
            )
        rows: list[dict[str, Any]] = []
        for effective_date, group in frame.groupby(
            "effective_date",
            sort=True,
        ):
            for field_name in fields:
                rows.append(
                    {
                        "effective_date": pd.Timestamp(
                            effective_date
                        ).normalize(),
                        "exposure_type": exposure_type(field_name),
                        "field": field_name,
                        **weighted_field_statistics(
                            group,
                            field_name,
                            context.spec.weight_tolerance,
                        ),
                    }
                )
        return AnalyticsPluginResult(
            tables={
                "exposures": pd.DataFrame(rows).sort_values(
                    ["effective_date", "exposure_type", "field"],
                    kind="mergesort",
                ).reset_index(drop=True)
            }
        )


@dataclass(frozen=True, slots=True)
class _LiquidityCapacityCoveragePlugin:
    plugin_id: str = "liquidity_capacity_coverage"
    version: str = "1"
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset(
        {
            "liquidity_capacity_coverage.coverage",
            "liquidity_capacity_coverage.capacity",
        }
    )

    def run(
        self,
        context: AnalyticsContext,
        parameters: Mapping[str, Any],
    ) -> AnalyticsPluginResult:
        frame = instrument_research_frame(
            context.backtest_result,
            context.research_inputs.liquidity_capacity_data,
            input_name="liquidity_capacity_data",
        )
        requested = tuple(
            str(item) for item in parameters.get("liquidity_fields", ())
        )
        capacity_field = str(
            parameters.get("capacity_limit_field", "")
        ).strip()
        if requested:
            missing = sorted(set(requested).difference(frame.columns))
            if missing:
                raise AnalyticsValidationError(
                    f"liquidity fields are missing: {missing}"
                )
            liquidity_fields = requested
        else:
            liquidity_fields = tuple(
                column
                for column in LIQUIDITY_FIELDS
                if column in frame.columns
            )
        if capacity_field:
            if capacity_field not in frame.columns:
                raise AnalyticsValidationError(
                    f"capacity limit field is missing: {capacity_field}"
                )
        else:
            capacity_field = next(
                (
                    column
                    for column in CAPACITY_LIMIT_FIELDS
                    if column in frame.columns
                ),
                "",
            )
        if not liquidity_fields and not capacity_field:
            raise MissingAnalyticsInput(
                "liquidity and capacity fields are unavailable"
            )

        coverage_rows: list[dict[str, Any]] = []
        coverage_fields = list(liquidity_fields)
        if capacity_field and capacity_field not in coverage_fields:
            coverage_fields.append(capacity_field)
        for effective_date, group in frame.groupby(
            "effective_date",
            sort=True,
        ):
            for field_name in coverage_fields:
                coverage_rows.append(
                    {
                        "effective_date": pd.Timestamp(
                            effective_date
                        ).normalize(),
                        "field": field_name,
                        **weighted_field_statistics(
                            group,
                            field_name,
                            context.spec.weight_tolerance,
                        ),
                    }
                )
        coverage = pd.DataFrame(
            coverage_rows,
            columns=[
                "effective_date",
                "field",
                "instrument_count",
                "available_count",
                "missing_count",
                "index_weight_coverage",
                "benchmark_weight_coverage",
                "index_exposure",
                "benchmark_exposure",
                "active_exposure",
            ],
        )
        capacity = capacity_table(
            frame,
            capacity_field,
            context.spec.weight_tolerance,
        )
        return AnalyticsPluginResult(
            tables={"coverage": coverage, "capacity": capacity}
        )
