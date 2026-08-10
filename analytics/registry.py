"""Lazy registry for discoverable analytics capabilities."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType
from typing import Any


class AnalyticsFeatureNotFoundError(KeyError):
    """Raised when an analytics feature is not registered."""


class AnalyticsFeatureLoadError(ImportError):
    """Raised when a registered analytics callable cannot be imported."""


@dataclass(frozen=True, slots=True)
class AnalyticsFeature:
    """Stable metadata and lazy import path for one analytics capability."""

    feature_id: str
    category: str
    description: str
    callable_path: str

    def __post_init__(self) -> None:
        for name in ("feature_id", "category", "description", "callable_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if ":" not in self.callable_path:
            raise ValueError("callable_path must use 'module:attribute' syntax")


class AnalyticsFeatureRegistry:
    """Explicit registry that imports implementations only when requested."""

    def __init__(self) -> None:
        self._features: dict[str, AnalyticsFeature] = {}

    def register(
        self,
        feature: AnalyticsFeature,
        *,
        replace: bool = False,
    ) -> AnalyticsFeature:
        """Register one feature without importing its implementation."""

        if not isinstance(feature, AnalyticsFeature):
            raise TypeError("feature must be an AnalyticsFeature")
        if feature.feature_id in self._features and not replace:
            raise KeyError(
                f"analytics feature is already registered: {feature.feature_id}"
            )
        self._features[feature.feature_id] = feature
        return feature

    def get(self, feature_id: str) -> AnalyticsFeature:
        """Return registered metadata without importing feature code."""

        try:
            return self._features[feature_id]
        except KeyError as exc:
            raise AnalyticsFeatureNotFoundError(
                f"analytics feature is not registered: {feature_id}"
            ) from exc

    def list(self) -> tuple[AnalyticsFeature, ...]:
        """Return stable feature metadata ordered by category and ID."""

        return tuple(
            sorted(
                self._features.values(),
                key=lambda item: (item.category, item.feature_id),
            )
        )

    def load(self, feature_id: str) -> Callable[..., Any]:
        """Import and return a registered callable with a clear failure."""

        feature = self.get(feature_id)
        module_name, attribute_name = feature.callable_path.split(":", 1)
        try:
            module = import_module(module_name)
            implementation = getattr(module, attribute_name)
        except (ImportError, AttributeError) as exc:
            raise AnalyticsFeatureLoadError(
                f"analytics feature {feature_id!r} could not load "
                f"{feature.callable_path!r}: {exc}"
            ) from exc
        if not callable(implementation):
            raise AnalyticsFeatureLoadError(
                f"analytics feature {feature_id!r} does not resolve to a callable"
            )
        return implementation

    def run(self, feature_id: str, /, *args: Any, **kwargs: Any) -> Any:
        """Load and execute one feature."""

        return self.load(feature_id)(*args, **kwargs)

    def items(self) -> Iterator[tuple[str, AnalyticsFeature]]:
        """Iterate over a read-only, stable view of registered entries."""

        return iter(
            MappingProxyType(dict(sorted(self._features.items()))).items()
        )


@dataclass(frozen=True, slots=True)
class AnalyticsFeatureEngine:
    """Small execution facade over an analytics feature registry."""

    registry: AnalyticsFeatureRegistry

    def available_features(self) -> tuple[AnalyticsFeature, ...]:
        """Return the features exposed by this engine."""

        return self.registry.list()

    def run(self, feature_id: str, /, *args: Any, **kwargs: Any) -> Any:
        """Execute a named analytics capability."""

        return self.registry.run(feature_id, *args, **kwargs)


def default_analytics_registry() -> AnalyticsFeatureRegistry:
    """Return a new registry containing the standard generic capabilities."""

    registry = AnalyticsFeatureRegistry()
    for feature in _BUILTIN_FEATURES:
        registry.register(feature)
    return registry


_BUILTIN_FEATURES = (
    AnalyticsFeature(
        "performance.summary",
        "performance",
        "Return, volatility, tracking-error, and drawdown statistics.",
        "icapa.analytics.performance:calculate_performance_metrics",
    ),
    AnalyticsFeature(
        "performance.rolling_risk",
        "performance",
        "Rolling volatility, tracking error, and correlation.",
        "icapa.analytics.performance:calculate_rolling_risk",
    ),
    AnalyticsFeature(
        "constituents.weight_change",
        "constituents",
        "Entrant, exit, and constituent weight-change diagnostics.",
        "icapa.analytics.constituents:explain_weight_change",
    ),
    AnalyticsFeature(
        "constituents.weight_explanation",
        "constituents",
        "Stepwise explanation of multiplicative methodology tilts.",
        "icapa.analytics.constituents:explain_weight_construction",
    ),
    AnalyticsFeature(
        "exposures.weighted",
        "exposures",
        "Portfolio, benchmark, and active weighted exposures.",
        "icapa.analytics.exposures:calculate_weighted_exposures",
    ),
    AnalyticsFeature(
        "attribution.brinson",
        "attribution",
        "Brinson-Fachler allocation, selection, and interaction attribution.",
        "icapa.analytics.attribution:calculate_brinson_attribution",
    ),
    AnalyticsFeature(
        "attribution.factor",
        "attribution",
        "Linear factor-return attribution with explicit inputs.",
        "icapa.analytics.attribution:calculate_factor_attribution",
    ),
    AnalyticsFeature(
        "risk.contributions",
        "risk",
        "Euler volatility and tracking-error contributions.",
        "icapa.analytics.risk:calculate_risk_contributions",
    ),
    AnalyticsFeature(
        "risk.concentration",
        "risk",
        "HHI, effective count, and top-weight concentration.",
        "icapa.analytics.risk:calculate_concentration",
    ),
    AnalyticsFeature(
        "risk.liquidity_capacity",
        "risk",
        "AUM, participation, and days-to-trade capacity diagnostics.",
        "icapa.analytics.risk:calculate_liquidity_capacity",
    ),
    AnalyticsFeature(
        "events.study",
        "events",
        "Business-day event windows and abnormal returns.",
        "icapa.analytics.events:run_event_study",
    ),
    AnalyticsFeature(
        "regimes.analysis",
        "regimes",
        "Regime performance, transitions, and optional distribution tests.",
        "icapa.analytics.regimes:analyze_regimes",
    ),
    AnalyticsFeature(
        "reconciliation.data_waterfall",
        "reconciliation",
        "Field-level differences across ordered calculation stages.",
        "icapa.analytics.reconciliation:compare_data_stages",
    ),
    AnalyticsFeature(
        "quality.coverage",
        "quality",
        "Row and weight coverage for required research fields.",
        "icapa.analytics.quality:calculate_data_coverage",
    ),
    AnalyticsFeature(
        "quality.freshness",
        "quality",
        "Point-in-time source freshness at a review cutoff.",
        "icapa.analytics.quality:calculate_data_freshness",
    ),
)


__all__ = [
    "AnalyticsFeature",
    "AnalyticsFeatureEngine",
    "AnalyticsFeatureLoadError",
    "AnalyticsFeatureNotFoundError",
    "AnalyticsFeatureRegistry",
    "default_analytics_registry",
]
