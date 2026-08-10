"""Target, constraint, and methodology diagnostic analytics plugins."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any
import pandas as pd
from ..contracts import AnalyticsValidationError
from .api import AnalyticsContext, AnalyticsPluginResult, MissingAnalyticsInput
from .support_diagnostics import (diagnostic_table, normalise_constraint_diagnostics, normalise_target_diagnostics)
from .support_common import json_scalar

@dataclass(frozen=True, slots=True)
class _TargetAttainmentPlugin:
    plugin_id: str = "target_attainment"
    version: str = "1"
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset({"target_attainment.detail"})

    def run(
        self,
        context: AnalyticsContext,
        parameters: Mapping[str, Any],
    ) -> AnalyticsPluginResult:
        del parameters
        source = context.research_inputs.target_diagnostics
        if source is None:
            source = diagnostic_table(
                context.backtest_result,
                keys=(
                    "target_diagnostics",
                    "target_attainment",
                    "targets",
                ),
            )
        if source is None or source.empty:
            raise MissingAnalyticsInput(
                "requested-versus-achieved target diagnostics are unavailable"
            )
        detail = normalise_target_diagnostics(source)
        return AnalyticsPluginResult(tables={"detail": detail})


@dataclass(frozen=True, slots=True)
class _ConstraintDiagnosticsPlugin:
    plugin_id: str = "constraint_diagnostics"
    version: str = "1"
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset(
        {
            "constraint_diagnostics.detail",
            "constraint_diagnostics.summary",
        }
    )

    def run(
        self,
        context: AnalyticsContext,
        parameters: Mapping[str, Any],
    ) -> AnalyticsPluginResult:
        tolerance = float(
            parameters.get("binding_tolerance", context.spec.weight_tolerance)
        )
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise AnalyticsValidationError(
                "constraint binding_tolerance must be finite and positive"
            )
        source = context.research_inputs.constraint_diagnostics
        if source is None:
            source = diagnostic_table(
                context.backtest_result,
                keys=("constraint_diagnostics", "constraints"),
            )
        if source is None or source.empty:
            raise MissingAnalyticsInput(
                "constraint binding, slack, and violation diagnostics are unavailable"
            )
        detail = normalise_constraint_diagnostics(source, tolerance)
        summary = (
            detail.groupby("effective_date", dropna=False, sort=True)
            .agg(
                constraint_count=("constraint_name", "size"),
                binding_count=("binding", "sum"),
                violated_count=("violated", "sum"),
                maximum_violation=("violation", "max"),
            )
            .reset_index()
        )
        return AnalyticsPluginResult(
            tables={"detail": detail, "summary": summary}
        )


@dataclass(frozen=True, slots=True)
class _MethodologyDiagnosticsPlugin:
    plugin_id: str = "methodology_diagnostics"
    version: str = "1"
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset(
        {
            "methodology_diagnostics.records",
            "methodology_diagnostics.constraints",
        }
    )

    def run(
        self,
        context: AnalyticsContext,
        parameters: Mapping[str, Any],
    ) -> AnalyticsPluginResult:
        del parameters
        reviews = getattr(context.backtest_result, "reviews", None)
        if not isinstance(reviews, Mapping):
            raise MissingAnalyticsInput("review diagnostics are unavailable")
        records: list[dict[str, Any]] = []
        constraints: list[dict[str, Any]] = []
        for effective_date, review in sorted(reviews.items()):
            diagnostics = getattr(review, "diagnostics", None)
            if not isinstance(diagnostics, Mapping) or not diagnostics:
                continue
            normalized_date = pd.Timestamp(effective_date).normalize()
            for key, value in sorted(diagnostics.items(), key=lambda item: str(item[0])):
                if key in {"constraints", "constraint_diagnostics"} and isinstance(
                    value,
                    Sequence,
                ) and not isinstance(value, (str, bytes)):
                    for item in value:
                        if isinstance(item, Mapping):
                            constraints.append(
                                {
                                    "effective_date": normalized_date,
                                    **{
                                        str(name): json_scalar(field_value)
                                        for name, field_value in item.items()
                                    },
                                }
                            )
                    continue
                records.append(
                    {
                        "effective_date": normalized_date,
                        "name": str(key),
                        "value": json_scalar(value),
                    }
                )
        if not records and not constraints:
            raise MissingAnalyticsInput("methodology diagnostics are unavailable")
        return AnalyticsPluginResult(
            tables={
                "records": pd.DataFrame(records),
                "constraints": pd.DataFrame(constraints),
            }
        )
