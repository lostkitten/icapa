"""High-level orchestration for reproducible index-research workflows.

The v1 backtesting, simulation, analytics, and reporting APIs remain
available independently.  This module composes them behind a small research
API while keeping cache reuse opt-in and evidence based.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..analytics import (
    ComparisonSpec,
    ResearchComparison,
    compare_research_results,
)
from ..reporting import ReportBundle, ReportBundleSpec, write_report_bundle
from ..workspace import WorkspaceRepository
from .runners.identity import (
    _methodology_report_name,
    _methodology_report_parameters,
    _report_data_sources,
)
from .runners.contracts import _RunReportWorkspace
from .runners.io import _report_label
from .catalog.workspace_runs import _ResearchRunCatalog
from .models import (
    CatalogRebuildResult,
    IndexDefinition,
    PruneResult,
    RecipeProviderBinding,
    ResearchRun,
    ResearchSimulationSpec,
    ResearchSpec,
    ResearchStatus,
    ResearchWorkflowError,
    UnsafeCacheReuseError,
    WorkspaceVerification,
)
from .runners.single import _ResearchExecution
from .runners.preflight import _ResearchPreflight


class ResearchWorkspace(
    _ResearchRunCatalog,
    _ResearchPreflight,
    _ResearchExecution,
):
    """Compose construction, simulation, analytics, and reporting.

    A workspace name always resolves beneath the deployment-controlled ICAPA
    workspace root.  Cache reuse is disabled by default.  When enabled, this
    API only delegates reuse to calculation engines after executable code and
    data identities are verified. Daily-data providers preferably implement
    ``describe_snapshot(capability=..., request=...)``. Providers may also
    expose ``research_data_identity`` when their identity service is separate
    from data retrieval. Snapshot payloads are reduced to secret-safe digests
    and are never persisted directly.
    """

    def __init__(self, workspace_name: str) -> None:
        self._workspace = WorkspaceRepository.open(workspace_name)

    @classmethod
    def open(cls, workspace_name: str) -> "ResearchWorkspace":
        """Open or create a high-level research workspace."""

        return cls(workspace_name)

    @property
    def workspace_name(self) -> str:
        return self._workspace.workspace_name

    @property
    def name(self) -> str:
        return self.workspace_name

    @property
    def workspace_path(self) -> Path:
        return self._workspace.workspace_path

    @property
    def reports_path(self) -> Path:
        path = self.workspace_path.joinpath("reports")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def compare(
        self,
        baseline: ResearchRun,
        candidates: Sequence[ResearchRun],
        *,
        spec: ComparisonSpec | None = None,
        baseline_name: str = "baseline",
        candidate_names: Sequence[str] | None = None,
    ) -> ResearchComparison:
        """Compare existing results without rerunning construction or simulation."""

        selected = tuple(candidates)
        names = (
            tuple(candidate_names)
            if candidate_names is not None
            else tuple(f"candidate_{offset}" for offset in range(1, len(selected) + 1))
        )
        if len(names) != len(selected):
            raise ValueError("candidate_names must match candidates")
        return compare_research_results(
            baseline.comparison_input(baseline_name),
            [
                run.comparison_input(name)
                for run, name in zip(selected, names, strict=True)
            ],
            spec=spec,
        )

    def write_report(
        self,
        run: ResearchRun,
        *,
        spec: ReportBundleSpec | None = None,
        comparison: ResearchComparison | None = None,
    ) -> ReportBundle:
        """Write a report bundle from already-computed research results."""

        if run.manifest.workspace_name != self.workspace_name:
            raise ResearchWorkflowError(
                "the research run belongs to a different workspace"
            )
        definition_fingerprint = run.manifest.definition_fingerprint
        if len(definition_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in definition_fingerprint
        ):
            raise ResearchWorkflowError(
                "the research run definition fingerprint is invalid"
            )
        reports_path = self.workspace_path.joinpath(
            "runs",
            definition_fingerprint,
            "reports",
        ).resolve()
        if not reports_path.is_relative_to(self.workspace_path.resolve()):
            raise ResearchWorkflowError(
                "the research report path escapes its workspace"
            )
        report_workspace = _RunReportWorkspace(
            workspace_name=self.workspace_name,
            reports_path=reports_path,
        )
        return write_report_bundle(
            report_workspace,
            run.backtest,
            simulation=run.simulation,
            analytics=run.analytics,
            comparison=comparison,
            run_manifest=run.manifest,
            index_name=_report_label(
                run.definition.display_name,
                fallback="Research Index",
            ),
            methodology_name=_report_label(
                _methodology_report_name(run.definition.methodology),
                fallback="Methodology",
            ),
            methodology_parameters=_methodology_report_parameters(
                run.definition.methodology
            ),
            data_sources=_report_data_sources(run.manifest.providers),
            spec=spec,
        )


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
    "ResearchWorkspace",
    "UnsafeCacheReuseError",
    "WorkspaceVerification",
]
