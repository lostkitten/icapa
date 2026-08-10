"""Registry of built-in analytics plugins."""
from __future__ import annotations
from .api import AnalyticsPlugin
from .attribution import _MultiPeriodAttributionPlugin
from .constituents import (_ConstituentChangePlugin, _SelectionReasonsPlugin, _TurnoverDecompositionPlugin, _WeightChangeContributorsPlugin)
from .core import _CoreAnalyticsPlugin, _LegacyParityPlugin
from .diagnostics import (_ConstraintDiagnosticsPlugin, _MethodologyDiagnosticsPlugin, _TargetAttainmentPlugin)
from .exposures import _FactorSignalExposurePlugin, _LiquidityCapacityCoveragePlugin
from .performance import (_CalendarPeriodPerformancePlugin, _DrawdownEpisodesPlugin, _RollingRiskPlugin)
from .quality import _DataCoveragePlugin, _DataFreshnessPlugin

def builtin_plugins() -> tuple[AnalyticsPlugin, ...]:
    return (
        _CoreAnalyticsPlugin(),
        _LegacyParityPlugin(),
        _ConstituentChangePlugin(),
        _SelectionReasonsPlugin(),
        _WeightChangeContributorsPlugin(),
        _TurnoverDecompositionPlugin(),
        _TargetAttainmentPlugin(),
        _ConstraintDiagnosticsPlugin(),
        _FactorSignalExposurePlugin(),
        _LiquidityCapacityCoveragePlugin(),
        _CalendarPeriodPerformancePlugin(),
        _RollingRiskPlugin(),
        _DrawdownEpisodesPlugin(),
        _DataCoveragePlugin(),
        _DataFreshnessPlugin(),
        _MultiPeriodAttributionPlugin(),
        _MethodologyDiagnosticsPlugin(),
    )


__all__ = ["builtin_plugins"]
