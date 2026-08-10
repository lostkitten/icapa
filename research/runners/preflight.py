"""Automatic identity and cache-safety preflight for research runs."""

from __future__ import annotations

from dataclasses import replace

from ...workspace import CacheMode, CacheStage
from .cache import _cache_decisions
from .contracts import _PreparedRun
from .identity import (
    _methodology_component_identity,
    _request_payload,
    _workflow_component_identities,
)
from .providers import _construction_identity, _provider_identities
from .snapshots import _review_snapshot, _simulation_snapshot
from ..models import ResearchSpec


class _ResearchPreflight:
    """Collect safe automatic identities before calculation begins."""

    def _prepare_run(
        self,
        spec: ResearchSpec,
        diagnostics: list[str],
    ) -> _PreparedRun:
        """Collect automatic identities before creating the main manifest."""

        component, component_verified = _methodology_component_identity(spec.definition)
        workflow_components, workflow_verified = _workflow_component_identities(spec)
        provider_evidence = _provider_identities(spec)
        construction_identity = _construction_identity(
            spec,
            methodology_component=component,
            backtester_component=workflow_components["backtester"],
            calendar_component=workflow_components["calendar"],
            providers=(
                *provider_evidence.review_records,
                *provider_evidence.calendar_records,
            ),
        )
        review_source_verified = (
            component_verified
            and workflow_verified["backtester"]
            and workflow_verified["calendar"]
            and provider_evidence.review_verified
            and provider_evidence.calendar_verified
        )
        review_snapshot = (
            _review_snapshot(
                spec,
                workspace=self._workspace,
                requested=spec.cache.mode_for(CacheStage.REVIEWS),
                diagnostics=diagnostics,
                construction_identity=construction_identity,
            )
            if review_source_verified
            and spec.cache.mode_for(CacheStage.REVIEWS) is not CacheMode.OFF
            else None
        )
        source_mode = spec.cache.mode_for(CacheStage.SOURCE_DATA)
        simulation_mode = spec.cache.mode_for(CacheStage.SIMULATION)
        simulation_snapshot_mode = (
            source_mode if source_mode is not CacheMode.OFF else simulation_mode
        )
        simulation_snapshot = (
            _simulation_snapshot(
                spec,
                workspace=self._workspace,
                requested=simulation_snapshot_mode,
                diagnostics=diagnostics,
            )
            if simulation_snapshot_mode is not CacheMode.OFF
            else None
        )
        decisions = _cache_decisions(
            spec,
            review_source_verified=review_source_verified,
            simulation_source_verified=(
                workflow_verified.get("simulator", True)
                and provider_evidence.simulation_verified
            ),
            analytics_source_verified=workflow_verified.get(
                "analytics_runner",
                True,
            ),
            review_snapshot=review_snapshot,
            simulation_snapshot=simulation_snapshot,
        )
        executable_identity_verified = (
            component_verified
            and workflow_verified["backtester"]
            and workflow_verified["calendar"]
            and provider_evidence.review_verified
            and provider_evidence.calendar_verified
            and (
                spec.simulation is None
                or (
                    workflow_verified.get("simulator", True)
                    and provider_evidence.simulation_verified
                )
            )
            and (
                spec.analytics is None
                or workflow_verified.get("analytics_runner", True)
            )
        )
        if not executable_identity_verified:
            decisions = tuple(
                (
                    replace(
                        decision,
                        actual=CacheMode.OFF,
                        reason=(
                            "Any non-OFF cache mode requires stable "
                            "executable identity for every calculation "
                            "component used by this request."
                        ),
                        fatal=True,
                    )
                    if decision.requested is not CacheMode.OFF
                    else decision
                )
                for decision in decisions
            )
        diagnostics.extend(
            decision.reason
            for decision in decisions
            if decision.requested is not CacheMode.OFF
            and decision.actual is CacheMode.OFF
        )
        request_payload = _request_payload(
            spec,
            workflow_components=workflow_components,
            providers=provider_evidence.records,
        )
        return _PreparedRun(
            component=component,
            workflow_components=workflow_components,
            provider_evidence=provider_evidence,
            construction_identity=construction_identity,
            decisions=decisions,
            request_payload=request_payload,
        )

    def _record_preflight_failure(
        self,
        spec: ResearchSpec,
        error: BaseException,
    ) -> None:
        """Best-effort sanitized manifest for failures before normal identity."""

        methodology_type = type(spec.definition.methodology)
        try:
            fallback = self._workspace.start_run(
                index_id=spec.definition.index_id,
                definition={
                    "index_id": spec.definition.index_id,
                    "preflight_identity": "unavailable",
                    "methodology_type": (
                        f"{methodology_type.__module__}."
                        f"{methodology_type.__qualname__}"
                    ),
                },
                request={"phase": "automatic_preflight"},
                request_identity={"phase": "automatic_preflight"},
                providers=(),
                calendar={
                    "type": (
                        f"{type(spec.calendar).__module__}."
                        f"{type(spec.calendar).__qualname__}"
                    ),
                },
                cache=spec.cache,
            )
            self._workspace.fail_run(fallback, error)
        except BaseException:
            # The original failure must remain authoritative. This fallback
            # intentionally never masks an environment-level storage failure.
            return


__all__ = ["_ResearchPreflight"]
