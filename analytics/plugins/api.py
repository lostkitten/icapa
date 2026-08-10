"""Contracts and deterministic execution for analytics plugins."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from ..contracts import (
    AnalyticsDiagnostic, AnalyticsResult, AnalyticsValidationError,
    ResearchAnalyticsInputs,
)

class ReturnSeries(str, Enum):
    """Explicit return series used by performance analytics."""

    PRICE = "price"
    GROSS_TOTAL = "gross_total"
    NET_TOTAL = "net_total"

    @property
    def columns(self) -> tuple[str, str]:
        """Return the canonical index and benchmark columns."""

        return (
            f"index_{self.value}_return",
            f"benchmark_{self.value}_return",
        )


class MissingInputPolicy(str, Enum):
    """Control optional analytics when their inputs are unavailable."""

    WARN_AND_SKIP = "warn_and_skip"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class AnalyticsPluginSpec:
    """One versioned analytics plugin request."""

    plugin_id: str
    version: str = "1"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    required: bool = True

    def __post_init__(self) -> None:
        if not self.plugin_id or not self.plugin_id.strip():
            raise AnalyticsValidationError("plugin_id must not be empty")
        if not self.version or not self.version.strip():
            raise AnalyticsValidationError("plugin version must not be empty")
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(dict(self.parameters)),
        )


@dataclass(frozen=True, slots=True)
class AnalyticsSpec:
    """Deterministic analytics configuration."""

    profile: str
    plugins: tuple[AnalyticsPluginSpec, ...]
    return_series: ReturnSeries = ReturnSeries.NET_TOTAL
    annualization_factor: int = 252
    weight_tolerance: float = 1e-8
    missing_optional_input: MissingInputPolicy = MissingInputPolicy.WARN_AND_SKIP

    def __post_init__(self) -> None:
        object.__setattr__(self, "return_series", ReturnSeries(self.return_series))
        object.__setattr__(
            self,
            "missing_optional_input",
            MissingInputPolicy(self.missing_optional_input),
        )
        if not self.profile:
            raise AnalyticsValidationError("analytics profile must not be empty")
        if self.annualization_factor <= 0:
            raise AnalyticsValidationError("annualization_factor must be positive")
        if self.weight_tolerance <= 0:
            raise AnalyticsValidationError("weight_tolerance must be positive")
        identifiers = [item.plugin_id for item in self.plugins]
        if len(identifiers) != len(set(identifiers)):
            raise AnalyticsValidationError("analytics plugin IDs must be unique")

    @classmethod
    def legacy_parity(cls) -> "AnalyticsSpec":
        """Return the exact version-one analytics profile."""

        return cls(
            profile="legacy_parity_v1",
            plugins=(AnalyticsPluginSpec("legacy_parity"),),
        )

    @classmethod
    def standard_research(cls) -> "AnalyticsSpec":
        """Return the standard index-research review profile."""

        return cls(
            profile="standard_research_v1",
            plugins=(
                AnalyticsPluginSpec("core_analytics"),
                AnalyticsPluginSpec("constituent_change"),
                AnalyticsPluginSpec("selection_reasons", required=False),
                AnalyticsPluginSpec("weight_change_contributors"),
                AnalyticsPluginSpec("turnover_decomposition"),
                AnalyticsPluginSpec("target_attainment", required=False),
                AnalyticsPluginSpec("constraint_diagnostics", required=False),
                AnalyticsPluginSpec("factor_signal_exposure", required=False),
                AnalyticsPluginSpec(
                    "liquidity_capacity_coverage",
                    required=False,
                ),
                AnalyticsPluginSpec(
                    "calendar_period_performance",
                    required=False,
                ),
                AnalyticsPluginSpec(
                    "rolling_risk",
                    parameters={"windows": (21, 63, 252, 756)},
                    required=False,
                ),
                AnalyticsPluginSpec("drawdown_episodes", required=False),
                AnalyticsPluginSpec("data_coverage"),
                AnalyticsPluginSpec("data_freshness", required=False),
                AnalyticsPluginSpec(
                    "multi_period_attribution",
                    required=False,
                ),
                AnalyticsPluginSpec("methodology_diagnostics", required=False),
            ),
        )


@dataclass(frozen=True, slots=True)
class AnalyticsContext:
    """Read-only inputs supplied to one analytics plugin."""

    backtest_result: object
    simulation_result: object | None
    spec: AnalyticsSpec
    available_tables: Mapping[str, pd.DataFrame]
    research_inputs: ResearchAnalyticsInputs = field(
        default_factory=ResearchAnalyticsInputs
    )


@dataclass(frozen=True, slots=True)
class AnalyticsPluginResult:
    """Structured output from one analytics plugin."""

    metrics: Mapping[str, float | int | str] = field(default_factory=dict)
    tables: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    diagnostics: tuple[AnalyticsDiagnostic, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(
            self,
            "tables",
            MappingProxyType(
                {
                    name: frame.copy(deep=True)
                    for name, frame in self.tables.items()
                }
            ),
        )
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@runtime_checkable
class AnalyticsPlugin(Protocol):
    """Protocol implemented by analytics extensions."""

    plugin_id: str
    version: str
    requires: frozenset[str]
    provides: frozenset[str]

    def run(
        self,
        context: AnalyticsContext,
        parameters: Mapping[str, Any],
    ) -> AnalyticsPluginResult: ...


@dataclass(frozen=True, slots=True)
class AnalyticsRunResult:
    """All plugin outputs for one analytics specification."""

    spec: AnalyticsSpec
    plugin_results: Mapping[str, AnalyticsPluginResult]
    legacy_result: AnalyticsResult | None
    diagnostics: tuple[AnalyticsDiagnostic, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plugin_results",
            MappingProxyType(dict(self.plugin_results)),
        )
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return all plugin tables with stable, namespaced keys."""

        result: dict[str, pd.DataFrame] = {}
        for plugin_id, plugin_result in self.plugin_results.items():
            for table_name, frame in plugin_result.tables.items():
                result[f"{plugin_id}.{table_name}"] = frame.copy(deep=True)
        primary_plugin_id = next(
            (
                plugin_id
                for plugin_id in ("core_analytics", "legacy_parity")
                if plugin_id in self.plugin_results
            ),
            None,
        )
        canonical_turnover = (
            f"{primary_plugin_id}.one_way_turnover"
            if primary_plugin_id is not None
            else ""
        )
        if (
            self.legacy_result is not None
            and primary_plugin_id is not None
            and canonical_turnover not in result
        ):
            result[canonical_turnover] = self.legacy_result.one_way_turnover
        return result


class MissingAnalyticsInput(AnalyticsValidationError):
    """Raised by a plugin when a declared optional input is unavailable."""


class AnalyticsPluginRunner:
    """Resolve and run a deterministic analytics plugin graph."""

    def __init__(
        self,
        plugins: Sequence[AnalyticsPlugin] | None = None,
    ) -> None:
        if plugins is None:
            from .builtins import builtin_plugins

            selected = tuple(builtin_plugins())
        else:
            selected = tuple(plugins)
        registry: dict[tuple[str, str], AnalyticsPlugin] = {}
        provided: dict[str, str] = {}
        for plugin in selected:
            key = (plugin.plugin_id, plugin.version)
            if key in registry:
                raise AnalyticsValidationError(
                    f"duplicate analytics plugin registration: {key}"
                )
            for artifact in plugin.provides:
                previous = provided.get(artifact)
                if previous is not None:
                    raise AnalyticsValidationError(
                        f"analytics artifact {artifact!r} is provided by both "
                        f"{previous!r} and {plugin.plugin_id!r}"
                    )
                provided[artifact] = plugin.plugin_id
            registry[key] = plugin
        self._registry = registry

    def run(
        self,
        backtest_result: object,
        simulation_result: object | None = None,
        *,
        spec: AnalyticsSpec | None = None,
        inputs: ResearchAnalyticsInputs | None = None,
    ) -> AnalyticsRunResult:
        """Run the requested plugin graph without mutating its inputs."""

        selected_spec = spec or AnalyticsSpec.standard_research()
        selected_inputs = inputs or ResearchAnalyticsInputs()
        if not isinstance(selected_inputs, ResearchAnalyticsInputs):
            raise AnalyticsValidationError(
                "inputs must be a ResearchAnalyticsInputs instance"
            )
        outputs: dict[str, AnalyticsPluginResult] = {}
        available: dict[str, pd.DataFrame] = {}
        diagnostics: list[AnalyticsDiagnostic] = []
        legacy_result: AnalyticsResult | None = None
        pending = list(selected_spec.plugins)

        while pending:
            progressed = False
            for request in list(pending):
                plugin = self._registry.get((request.plugin_id, request.version))
                if plugin is None:
                    raise AnalyticsValidationError(
                        f"analytics plugin is not registered: "
                        f"{request.plugin_id}@{request.version}"
                    )
                if not plugin.requires.issubset(available):
                    continue
                context = AnalyticsContext(
                    backtest_result=backtest_result,
                    simulation_result=simulation_result,
                    spec=selected_spec,
                    available_tables=MappingProxyType(
                        {
                            name: frame.copy(deep=True)
                            for name, frame in available.items()
                        }
                    ),
                    research_inputs=selected_inputs,
                )
                try:
                    result = plugin.run(context, request.parameters)
                except MissingAnalyticsInput as error:
                    if (
                        request.required
                        or selected_spec.missing_optional_input
                        is MissingInputPolicy.FAIL
                    ):
                        raise
                    diagnostics.append(
                        AnalyticsDiagnostic(
                            level="warning",
                            code=f"{request.plugin_id}_skipped",
                            message=str(error),
                        )
                    )
                    result = AnalyticsPluginResult()
                outputs[request.plugin_id] = result
                diagnostics.extend(result.diagnostics)
                for name, frame in result.tables.items():
                    qualified = f"{request.plugin_id}.{name}"
                    available[qualified] = frame.copy(deep=True)
                if request.plugin_id in {"core_analytics", "legacy_parity"}:
                    legacy_result = result.metadata.get("legacy_result")
                pending.remove(request)
                progressed = True
            if not progressed:
                unresolved = ", ".join(item.plugin_id for item in pending)
                raise AnalyticsValidationError(
                    f"analytics plugin dependencies cannot be resolved: {unresolved}"
                )

        return AnalyticsRunResult(
            spec=selected_spec,
            plugin_results=outputs,
            legacy_result=legacy_result,
            diagnostics=tuple(diagnostics),
        )


def run_analytics_plugins(
    backtest_result: object,
    simulation_result: object | None = None,
    *,
    spec: AnalyticsSpec | None = None,
    plugins: Sequence[AnalyticsPlugin] | None = None,
    inputs: ResearchAnalyticsInputs | None = None,
) -> AnalyticsRunResult:
    """Convenience entry point for the plugin runner."""

    return AnalyticsPluginRunner(plugins).run(
        backtest_result,
        simulation_result,
        spec=spec,
        inputs=inputs,
    )


__all__ = [
    "ReturnSeries",
    "MissingInputPolicy",
    "AnalyticsPluginSpec",
    "AnalyticsSpec",
    "AnalyticsContext",
    "AnalyticsPluginResult",
    "AnalyticsPlugin",
    "AnalyticsRunResult",
    "MissingAnalyticsInput",
    "AnalyticsPluginRunner",
    "run_analytics_plugins",
 ]
