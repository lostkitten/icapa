"""Provider-neutral, side-effect-free analytics for ICAPA results.

The public API accepts already-produced backtest and simulation results. It
does not load data, call providers, mutate calculations, or write reports.
"""

from .attribution import calculate_brinson_attribution
from .comparison import (
    ComparisonEngine,
    ComparisonInput,
    ComparisonSpec,
    CompatibilityPolicy,
    DateAlignment,
    InstrumentAlignment,
    ResearchComparison,
    ReviewAlignment,
    compare_research_results,
)
from .contracts import (
    AnalyticsDiagnostic,
    AnalyticsResult,
    AnalyticsValidationError,
    BrinsonAttribution,
    BrinsonInput,
    ResearchAnalyticsInputs,
)
from .core import AnalyticsEngine, analyze_backtest
from .registry import (
    AnalyticsFeature,
    AnalyticsFeatureEngine,
    AnalyticsFeatureLoadError,
    AnalyticsFeatureNotFoundError,
    AnalyticsFeatureRegistry,
    default_analytics_registry,
)
from .plugins import (
    AnalyticsContext,
    AnalyticsPlugin,
    AnalyticsPluginResult,
    AnalyticsPluginRunner,
    AnalyticsPluginSpec,
    AnalyticsRunResult,
    AnalyticsSpec,
    MissingAnalyticsInput,
    MissingInputPolicy,
    ReturnSeries,
    run_analytics_plugins,
)
__all__ = [
    "AnalyticsContext",
    "AnalyticsDiagnostic",
    "AnalyticsEngine",
    "AnalyticsFeature",
    "AnalyticsFeatureEngine",
    "AnalyticsFeatureLoadError",
    "AnalyticsFeatureNotFoundError",
    "AnalyticsFeatureRegistry",
    "AnalyticsPlugin",
    "AnalyticsPluginResult",
    "AnalyticsPluginRunner",
    "AnalyticsPluginSpec",
    "AnalyticsResult",
    "AnalyticsRunResult",
    "AnalyticsSpec",
    "AnalyticsValidationError",
    "BrinsonAttribution",
    "BrinsonInput",
    "ComparisonEngine",
    "ComparisonInput",
    "ComparisonSpec",
    "CompatibilityPolicy",
    "DateAlignment",
    "InstrumentAlignment",
    "MissingAnalyticsInput",
    "MissingInputPolicy",
    "ResearchComparison",
    "ResearchAnalyticsInputs",
    "ReviewAlignment",
    "ReturnSeries",
    "analyze_backtest",
    "calculate_brinson_attribution",
    "compare_research_results",
    "default_analytics_registry",
    "run_analytics_plugins",
]
