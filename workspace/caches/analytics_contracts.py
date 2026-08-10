"""Persistence constants and errors for analytics workspace caching."""


SCHEMA_VERSION = 1
COMMIT_BINDING = "analytics-result"
TABLE_BINDING_PREFIX = "analytics-table"
SERIES_COLUMN = "__icapa_analytics_series_value__"
MAX_COMMIT_BYTES = 1024 * 1024
LEGACY_FRAME_FIELDS = (
    "review_validation",
    "review_metrics",
    "country_exposures",
    "industry_exposures",
    "target_review_weight_change",
    "formal_turnover",
    "drawdowns",
)
TYPE_TAG = "__icapa_analytics_type__"


class AnalyticsWorkspaceCacheError(RuntimeError):
    """Base error for persistent analytics cache operations."""


class AnalyticsWorkspaceCacheIntegrityError(AnalyticsWorkspaceCacheError):
    """Raised when a committed analytics result is missing or corrupt."""


class AnalyticsWorkspaceCacheSerializationError(AnalyticsWorkspaceCacheError):
    """Raised when an analytics result cannot be persisted without coercion."""


class AnalyticsWorkspaceCacheCollisionError(AnalyticsWorkspaceCacheError):
    """Raised when one cache identity resolves to a different result."""


class AnalyticsWorkspaceCacheMissError(AnalyticsWorkspaceCacheError):
    """Raised when READ_ONLY mode cannot resolve a complete result."""


__all__ = [
    "AnalyticsWorkspaceCacheCollisionError",
    "AnalyticsWorkspaceCacheError",
    "AnalyticsWorkspaceCacheIntegrityError",
    "AnalyticsWorkspaceCacheMissError",
    "AnalyticsWorkspaceCacheSerializationError",
]
