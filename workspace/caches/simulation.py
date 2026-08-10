"""Workspace-backed identity adapter for simulation cache keys."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ..identity import (
    automatic_digest,
    automatic_runtime_identity,
    automatic_source_closure_identity,
    dataframe_content_digest,
    safe_parameter_identity,
)


class WorkspaceSimulationIdentityService:
    """Use the workspace's canonical identity implementation for simulations."""

    def digest(self, value: Any) -> str:
        return automatic_digest(value)

    def safe_parameters(
        self,
        parameters: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return safe_parameter_identity(parameters)

    def source_identity(self, component: object) -> dict[str, Any]:
        return automatic_source_closure_identity(component)

    def runtime_identity(self) -> tuple[dict[str, Any], ...]:
        return automatic_runtime_identity()

    def dataframe_digest(
        self,
        frame: pd.DataFrame,
        *,
        sort_by: Sequence[str] | None = None,
    ) -> str:
        return dataframe_content_digest(frame, sort_by=sort_by)


__all__ = ["WorkspaceSimulationIdentityService"]
