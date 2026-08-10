"""Single-run orchestration for reproducible index research."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from ...analytics import AnalyticsRunResult
from ...backtesting import IndexSimulationResult
from ...workspace import CacheStage
from .analytics import _AnalyticsRunner
from .artifacts import _ArtifactWriter
from .cache import _decision_for, _input_digest_records
from .identity import (
    _calendar_semantics,
    _definition_payload,
    _request_identity_payload,
)
from .reviews import _ReviewRunner
from .simulation import _SimulationRunner
from ..models import ResearchRun, ResearchSpec, UnsafeCacheReuseError


class _ResearchExecution(
    _ReviewRunner,
    _SimulationRunner,
    _AnalyticsRunner,
    _ArtifactWriter,
):
    """Coordinate one complete research calculation."""

    def run(self, spec: ResearchSpec) -> ResearchRun:
        """Run the configured research pipeline and persist immutable outputs."""

        if not isinstance(spec, ResearchSpec):
            raise TypeError("spec must be ResearchSpec")
        diagnostics: list[str] = []
        try:
            prepared = self._prepare_run(spec, diagnostics)
            component = prepared.component
            workflow_components = prepared.workflow_components
            provider_evidence = prepared.provider_evidence
            construction_identity = prepared.construction_identity
            decisions = prepared.decisions
            request_payload = prepared.request_payload
            manifest = self._workspace.start_run(
                index_id=spec.definition.index_id,
                definition=_definition_payload(
                    spec.definition,
                    component,
                    workflow_components,
                    construction_identity,
                    (
                        *provider_evidence.review_records,
                        *provider_evidence.calendar_records,
                    ),
                ),
                request=request_payload,
                request_identity=_request_identity_payload(request_payload),
                providers=provider_evidence.records,
                calendar=_calendar_semantics(spec.calendar),
                cache=spec.cache,
            )
        except BaseException as error:
            self._record_preflight_failure(spec, error)
            raise
        manifest = replace(
            manifest,
            cache_decisions=tuple(item.as_dict() for item in decisions),
        )
        self._workspace.write_manifest(manifest)

        try:
            fatal = next((item for item in decisions if item.fatal), None)
            if fatal is not None:
                raise UnsafeCacheReuseError(fatal.reason)
            review_decision = _decision_for(decisions, CacheStage.REVIEWS)
            backtester = self._backtester(
                spec,
                review_decision,
                construction_identity,
                manifest.definition_fingerprint,
            )
            backtest = backtester.run()

            simulation: IndexSimulationResult | None = None
            source_records: tuple[Mapping[str, object], ...] = ()
            if spec.simulation is not None:
                simulation_decision = _decision_for(
                    decisions,
                    CacheStage.SIMULATION,
                )
                source_decision = _decision_for(
                    decisions,
                    CacheStage.SOURCE_DATA,
                )
                simulation, source_records = self._simulate(
                    spec,
                    backtest,
                    backtester,
                    simulation_decision,
                    source_decision,
                )

            analytics: AnalyticsRunResult | None = None
            analytics_records: tuple[Mapping[str, object], ...] = ()
            if spec.analytics is not None:
                analytics, analytics_records = self._run_analytics(
                    spec,
                    backtest,
                    simulation,
                    _decision_for(decisions, CacheStage.ANALYTICS),
                    workflow_components["analytics_runner"],
                )
            artifacts = self._save_result_artifacts(
                backtest,
                simulation,
                analytics,
            )
            completed = self._workspace.complete_run(
                manifest,
                artifacts=artifacts,
                input_digests=_input_digest_records(
                    backtest,
                    decisions,
                    source_records=source_records,
                    analytics_records=analytics_records,
                ),
            )
            self._write_research_status(
                completed.execution_id,
                spec.status,
            )
            result = ResearchRun(
                definition=spec.definition,
                backtest=backtest,
                simulation=simulation,
                analytics=analytics,
                manifest=completed,
                manifest_ref=self._workspace.manifest_ref(completed),
                cache_diagnostics=tuple(dict.fromkeys(diagnostics)),
                label=spec.label,
                tags=spec.tags,
            )
            if spec.report is not None:
                result = replace(
                    result,
                    report=self.write_report(result, spec=spec.report),
                )
            return result
        except BaseException as error:
            latest = self._workspace.open_manifest(
                self._workspace.manifest_ref(manifest)
            )
            if latest.status in {"running", "complete"}:
                self._workspace.fail_run(latest, error)
            raise


__all__ = ["_ResearchExecution"]
