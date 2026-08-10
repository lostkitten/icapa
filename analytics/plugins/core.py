"""Core analytics plugins."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import pandas as pd
from ..core import analyze_backtest
from .api import AnalyticsContext, AnalyticsPluginResult
from .support_common import json_scalar

@dataclass(frozen=True, slots=True)
class _CoreAnalyticsPlugin:
    plugin_id: str = "core_analytics"
    version: str = "1"
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset({"core_analytics.result"})

    def run(
        self,
        context: AnalyticsContext,
        parameters: Mapping[str, Any],
    ) -> AnalyticsPluginResult:
        del parameters
        result = analyze_backtest(
            context.backtest_result,
            context.simulation_result,
            return_columns=context.spec.return_series.columns,
            annualization_factor=context.spec.annualization_factor,
            weight_tolerance=context.spec.weight_tolerance,
        )
        return AnalyticsPluginResult(
            metrics={
                str(name): json_scalar(value)
                for name, value in result.performance.items()
            },
            tables={
                name: value.to_frame()
                if isinstance(value, pd.Series)
                else value
                for name, value in result.tables().items()
            },
            diagnostics=result.diagnostics,
            metadata={"legacy_result": result},
        )


@dataclass(frozen=True, slots=True)
class _LegacyParityPlugin(_CoreAnalyticsPlugin):
    """Retain the v1 plugin identifier for persisted specifications."""

    plugin_id: str = "legacy_parity"
    provides: frozenset[str] = frozenset({"legacy_parity.result"})
