"""Effective-date target-weight generation."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "ArtifactMetadata": ".cache_contracts",
    "CachePolicy": ".cache_contracts",
    "CacheSource": ".cache_contracts",
    "LoadedReview": ".cache_contracts",
    "ReviewArtifact": ".cache_contracts",
    "ReviewCacheMissError": ".cache_contracts",
    "ReviewCacheStore": ".cache_contracts",
    "register_review_store_factory": ".cache_contracts",
    "build_run_fingerprint": ".identity",
    "canonical_digest": ".identity",
    "BacktestMetadata": ".models",
    "BacktestResult": ".models",
    "ReviewResultMetadata": ".models",
    "Backtester": ".runner",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Load one review-domain symbol without importing unrelated domains."""

    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
