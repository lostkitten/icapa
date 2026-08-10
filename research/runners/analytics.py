"""Analytics execution and cache reuse for research runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...analytics import AnalyticsRunResult, run_analytics_plugins
from ...backtesting import BacktestResult, IndexSimulationResult
from ...workspace import CacheMode, IdentityError
from ...workspace.caches.analytics import AnalyticsWorkspaceCache
from .cache import _analytics_cache_identity
from .contracts import _CacheDecision
from ..models import ResearchSpec, UnsafeCacheReuseError


class _AnalyticsRunner:
    """Run or reuse analytics from deterministic research inputs."""

    def _run_analytics(
        self,
        spec: ResearchSpec,
        backtest: BacktestResult,
        simulation: IndexSimulationResult | None,
        decision: _CacheDecision,
        runner_identity: Mapping[str, Any],
    ) -> tuple[AnalyticsRunResult, tuple[Mapping[str, Any], ...]]:
        if spec.analytics is None:
            raise AssertionError("analytics specification is required")
        try:
            identity = _analytics_cache_identity(
                spec,
                backtest,
                simulation,
                runner_identity=runner_identity,
            )
        except (IdentityError, TypeError, ValueError) as exc:
            if decision.actual is not CacheMode.OFF:
                raise UnsafeCacheReuseError(
                    "analytics caching requires deterministic identities for "
                    "all review, simulation, plugin, and research inputs"
                ) from exc
            return (
                run_analytics_plugins(
                    backtest,
                    simulation,
                    spec=spec.analytics,
                    inputs=spec.analytics_inputs,
                ),
                (),
            )
        outcome = AnalyticsWorkspaceCache(
            self._workspace,
            mode=decision.actual,
        ).execute(
            identity,
            lambda: run_analytics_plugins(
                backtest,
                simulation,
                spec=spec.analytics,
                inputs=spec.analytics_inputs,
            ),
        )
        return (
            outcome.result,
            (
                {
                    "input_type": "analytics_calculation_inputs",
                    "content_digest": identity.cache_key,
                    "cache_source": outcome.source.value,
                },
            ),
        )


__all__ = ["_AnalyticsRunner"]
