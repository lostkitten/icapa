"""Multi-period attribution analytics plugin."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from ..attribution.brinson import calculate_brinson_attribution
from ..contracts import BrinsonInput
from .api import AnalyticsContext, AnalyticsPluginResult, MissingAnalyticsInput
from .support_attribution import carino_link_attribution

@dataclass(frozen=True, slots=True)
class _MultiPeriodAttributionPlugin:
    plugin_id: str = "multi_period_attribution"
    version: str = "1"
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset(
        {
            "multi_period_attribution.detail",
            "multi_period_attribution.totals",
            "multi_period_attribution.linked_detail",
            "multi_period_attribution.linked_totals",
        }
    )

    def run(
        self,
        context: AnalyticsContext,
        parameters: Mapping[str, Any],
    ) -> AnalyticsPluginResult:
        del parameters
        attribution_input = context.research_inputs.attribution_input
        if attribution_input is None:
            candidate = getattr(
                context.simulation_result,
                "attribution_input",
                None,
            )
            if isinstance(candidate, BrinsonInput):
                attribution_input = candidate
        if attribution_input is None:
            raise MissingAnalyticsInput(
                "pre-aligned multi-period attribution input is unavailable"
            )
        result = calculate_brinson_attribution(
            attribution_input,
            weight_tolerance=context.spec.weight_tolerance,
        )
        linked_detail, linked_totals = carino_link_attribution(
            result.totals,
            tolerance=context.spec.weight_tolerance,
        )
        return AnalyticsPluginResult(
            tables={
                "detail": result.detail.reset_index(),
                "totals": result.totals.reset_index(),
                "linked_detail": linked_detail,
                "linked_totals": linked_totals,
            },
            metadata={
                "period_count": int(
                    result.totals.index.nunique()
                )
            },
        )
