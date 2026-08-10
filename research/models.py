"""Public value types for high-level index-research workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd

from ..analytics import (
    AnalyticsRunResult,
    AnalyticsSpec,
    ComparisonInput,
    ResearchAnalyticsInputs,
)
from ..backtesting import (
    BacktestResult,
    Calendar,
    IndexSimulationResult,
    RebalanceFrequency,
    SimulationMaterialization,
    SimulationParams,
    WeightSnapshotMode,
)
from ..portfolio_construction import IndexRecipe, RecipeWeightProducer
from ..reporting import ReportBundle, ReportBundleSpec
from ..workspace import (
    CacheOptions,
    RunManifest,
    RunManifestRef,
)
from ..workspace.caches.source import UnsafeCacheReuseError


class ResearchWorkflowError(RuntimeError):
    """Base error for the high-level research workflow."""


class ResearchStatus(str, Enum):
    """Research-governance state for an immutable execution."""

    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class IndexDefinition:
    """A business index identifier paired with executable methodology logic.

    Calculation fingerprints, source digests, and configuration digests are
    collected by :class:`ResearchWorkspace`; researchers do not supply them.
    """

    index_id: str
    methodology: object
    name: str | None = None
    base_currency: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.CUSTOM
    recipe: IndexRecipe | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.index_id, str) or not self.index_id.strip():
            raise ValueError("index_id must not be empty")
        methodology = self.methodology
        if isinstance(methodology, IndexRecipe):
            object.__setattr__(self, "recipe", methodology)
            methodology = RecipeWeightProducer(methodology)
            object.__setattr__(self, "methodology", methodology)
        if not callable(getattr(methodology, "execute", None)):
            raise TypeError("methodology must implement execute(data_context)")
        if self.name is not None and not self.name.strip():
            raise ValueError("name must not be empty when supplied")
        if self.base_currency is not None and not self.base_currency.strip():
            raise ValueError("base_currency must not be empty when supplied")
        object.__setattr__(
            self,
            "attributes",
            MappingProxyType(dict(self.attributes)),
        )
        object.__setattr__(
            self,
            "rebalance_frequency",
            RebalanceFrequency(self.rebalance_frequency),
        )

    @property
    def display_name(self) -> str:
        """Return the report-facing name without inventing a second ID."""

        return self.name or self.index_id


def _default_simulation_params() -> SimulationParams:
    return SimulationParams(
        materialization=SimulationMaterialization(
            weight_snapshots=WeightSnapshotMode.NONE,
            include_asset_returns=False,
        )
    )


@dataclass(frozen=True, slots=True)
class ResearchSimulationSpec:
    """Daily index simulation inputs.

    Large constituent snapshots are disabled by default. Rebalance snapshots
    can be requested through ``params.materialization`` when a review needs
    them.
    """

    market_data_provider_name: str
    start_date: object
    end_date: object
    provider_parameters: Mapping[str, Any] = field(default_factory=dict)
    params: SimulationParams = field(default_factory=_default_simulation_params)
    segmented_cache: bool = True
    streaming: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.market_data_provider_name, str)
            or not self.market_data_provider_name.strip()
        ):
            raise ValueError("market_data_provider_name is required")
        start = pd.Timestamp(self.start_date).normalize()
        end = pd.Timestamp(self.end_date).normalize()
        if pd.isna(start) or pd.isna(end):
            raise ValueError("simulation dates must not be null")
        if start > end:
            raise ValueError("start_date must not be after end_date")
        if not isinstance(self.params, SimulationParams):
            raise TypeError("params must be SimulationParams")
        if not isinstance(self.segmented_cache, bool):
            raise TypeError("segmented_cache must be a bool")
        if not isinstance(self.streaming, bool):
            raise TypeError("streaming must be a bool")
        object.__setattr__(self, "start_date", start)
        object.__setattr__(self, "end_date", end)
        object.__setattr__(
            self,
            "provider_parameters",
            MappingProxyType(dict(self.provider_parameters)),
        )


@dataclass(frozen=True, slots=True)
class RecipeProviderBinding:
    """Explicit provider selection for one recipe capability."""

    provider_name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.provider_name, str)
            or not self.provider_name.strip()
        ):
            raise ValueError("provider_name must be a non-empty string")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        object.__setattr__(self, "provider_name", self.provider_name.strip())
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(dict(self.parameters)),
        )


@dataclass(frozen=True, slots=True)
class ResearchSpec:
    """One complete index-research request."""

    definition: IndexDefinition
    calendar: Calendar
    simulation: ResearchSimulationSpec | None = None
    analytics: AnalyticsSpec | None = field(
        default_factory=AnalyticsSpec.standard_research
    )
    cache: CacheOptions = field(default_factory=CacheOptions.off)
    report: ReportBundleSpec | None = None
    label: str | None = None
    tags: tuple[str, ...] = ()
    status: ResearchStatus = ResearchStatus.DRAFT
    analytics_inputs: ResearchAnalyticsInputs = field(
        default_factory=ResearchAnalyticsInputs
    )
    recipe_providers: Mapping[
        str,
        RecipeProviderBinding | str | Mapping[str, Any],
    ] = field(default_factory=dict)
    random_seed: int | None = None
    allow_empty_recipe_initial_state: bool = False
    allow_additional_reviews: bool = False
    allow_frequency_gaps: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.definition, IndexDefinition):
            raise TypeError("definition must be IndexDefinition")
        if not isinstance(self.calendar, Calendar):
            raise TypeError("calendar must be Calendar")
        if self.calendar.dates.empty:
            raise ValueError("calendar must contain at least one review")
        for name in (
            "allow_empty_recipe_initial_state",
            "allow_additional_reviews",
            "allow_frequency_gaps",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if (
            self.random_seed is not None
            and (
                not isinstance(self.random_seed, int)
                or isinstance(self.random_seed, bool)
                or self.random_seed < 0
            )
        ):
            raise ValueError(
                "random_seed must be a non-negative integer or None"
            )
        self.calendar.validate_frequency(
            self.definition.rebalance_frequency,
            allow_additional_reviews=self.allow_additional_reviews,
            allow_gaps=self.allow_frequency_gaps,
        )
        recipe_providers = _normalize_recipe_provider_bindings(
            self.recipe_providers
        )
        recipe = self.definition.recipe
        required_capabilities = (
            set()
            if recipe is None
            else {
                capability
                for node in recipe.nodes
                for capability in (
                    node.stage.requirements.provider_capabilities
                )
            }
        )
        missing_capabilities = required_capabilities.difference(
            recipe_providers
        )
        if missing_capabilities:
            raise ValueError(
                "recipe provider bindings are missing capabilities: "
                + ", ".join(sorted(missing_capabilities))
            )
        object.__setattr__(
            self,
            "recipe_providers",
            MappingProxyType(recipe_providers),
        )
        if self.simulation is not None and not isinstance(
            self.simulation,
            ResearchSimulationSpec,
        ):
            raise TypeError("simulation must be ResearchSimulationSpec")
        if self.analytics is not None and not isinstance(
            self.analytics,
            AnalyticsSpec,
        ):
            raise TypeError("analytics must be AnalyticsSpec or None")
        if not isinstance(self.analytics_inputs, ResearchAnalyticsInputs):
            raise TypeError(
                "analytics_inputs must be ResearchAnalyticsInputs"
            )
        if not isinstance(self.cache, CacheOptions):
            raise TypeError("cache must be CacheOptions")
        if self.report is not None and not isinstance(
            self.report,
            ReportBundleSpec,
        ):
            raise TypeError("report must be ReportBundleSpec or None")
        if self.label is not None and not self.label.strip():
            raise ValueError("label must not be empty when supplied")
        tags = tuple(sorted({str(tag).strip() for tag in self.tags}))
        if any(not tag for tag in tags):
            raise ValueError("tags must not contain empty values")
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "status", ResearchStatus(self.status))


@dataclass(frozen=True, slots=True)
class ResearchRun:
    """Outputs and automatic lineage for one completed research request."""

    definition: IndexDefinition
    backtest: BacktestResult
    simulation: IndexSimulationResult | None
    analytics: AnalyticsRunResult | None
    manifest: RunManifest
    manifest_ref: RunManifestRef
    report: ReportBundle | None = None
    cache_diagnostics: tuple[str, ...] = ()
    label: str | None = None
    tags: tuple[str, ...] = ()

    @property
    def review_status(self) -> pd.DataFrame:
        """Return one row of execution/cache status per effective date."""

        metadata = getattr(self.backtest, "metadata", None)
        review_metadata = getattr(metadata, "reviews", {}) or {}
        rows: list[dict[str, Any]] = []
        for effective_date, context in sorted(self.backtest.reviews.items()):
            item = review_metadata.get(effective_date)
            source = getattr(item, "cache_source", None)
            rows.append(
                {
                    "reference_date": pd.Timestamp(
                        context.reference_date
                    ).normalize(),
                    "effective_date": pd.Timestamp(effective_date).normalize(),
                    "status": "complete",
                    "cache_source": getattr(source, "value", source)
                    or "computed",
                    "constituent_count": int(len(context.cons)),
                }
            )
        return pd.DataFrame.from_records(rows)

    def comparison_input(self, name: str | None = None) -> ComparisonInput:
        """Return the existing comparison contract without recalculating."""

        return ComparisonInput(
            name=name or self.label or self.definition.display_name,
            backtest=self.backtest,
            simulation=self.simulation,
            analytics=self.analytics,
            manifest=self.manifest,
        )


@dataclass(frozen=True, slots=True)
class WorkspaceVerification:
    """Non-mutating integrity-check result."""

    ok: bool
    manifests_checked: int
    artifacts_checked: int
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CatalogRebuildResult:
    """Result of rebuilding manifests, artifacts, and reusable bindings."""

    discovered: int
    registered: int
    errors: tuple[str, ...] = ()
    artifacts: int = 0
    bindings: int = 0


@dataclass(frozen=True, slots=True)
class PruneResult:
    """Dry-run plan or result for explicitly removing orphan artifacts."""

    dry_run: bool
    candidates: tuple[Path, ...]
    removed: tuple[Path, ...]
    candidate_bytes: int


def _normalize_recipe_provider_bindings(
    bindings: Mapping[
        str,
        RecipeProviderBinding | str | Mapping[str, Any],
    ],
) -> dict[str, RecipeProviderBinding]:
    if not isinstance(bindings, Mapping):
        raise TypeError("recipe_providers must be a mapping")
    result: dict[str, RecipeProviderBinding] = {}
    for raw_capability, value in bindings.items():
        capability = str(raw_capability).strip()
        if not capability:
            raise ValueError(
                "recipe provider capability names must not be empty"
            )
        if isinstance(value, RecipeProviderBinding):
            binding = value
        elif isinstance(value, str):
            binding = RecipeProviderBinding(value)
        elif isinstance(value, Mapping):
            unknown = set(value).difference(
                {"provider_name", "parameters"}
            )
            if unknown:
                raise ValueError(
                    "recipe provider binding contains unknown fields: "
                    + ", ".join(sorted(map(str, unknown)))
                )
            binding = RecipeProviderBinding(
                provider_name=value.get("provider_name", ""),
                parameters=value.get("parameters", {}),
            )
        else:
            raise TypeError(
                "recipe provider bindings must be provider names, "
                "RecipeProviderBinding objects, or mappings"
            )
        result[capability] = binding
    return result


__all__ = [
    "CatalogRebuildResult",
    "IndexDefinition",
    "PruneResult",
    "RecipeProviderBinding",
    "ResearchRun",
    "ResearchSimulationSpec",
    "ResearchSpec",
    "ResearchStatus",
    "ResearchWorkflowError",
    "UnsafeCacheReuseError",
    "WorkspaceVerification",
]
