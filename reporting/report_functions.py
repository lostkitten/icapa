"""Public report helpers with workspace-confined output."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .excel_report import ExcelReportRenderer
from .report_data import ReportPayload


_REPORT_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,119}\.xlsx$")
_WORKSPACE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,79}$")
_WINDOWS_RESERVED = {
    "aux",
    "com1",
    "com2",
    "com3",
    "com4",
    "com5",
    "com6",
    "com7",
    "com8",
    "com9",
    "con",
    "lpt1",
    "lpt2",
    "lpt3",
    "lpt4",
    "lpt5",
    "lpt6",
    "lpt7",
    "lpt8",
    "lpt9",
    "nul",
    "prn",
}


@runtime_checkable
class WorkspaceLike(Protocol):
    """Minimal named-workspace interface required by report writing."""

    name: str
    path: Path
    reports_path: Path

    def report_path(self, filename: str) -> Path:
        """Return a path under this workspace's reports directory."""


def validate_report_filename(filename: str) -> str:
    """Validate and normalise one flat, portable report filename."""

    if not isinstance(filename, str):
        raise TypeError("filename must be a string")
    value = filename.strip()
    if not value:
        raise ValueError("filename must not be empty")
    if not value.casefold().endswith(".xlsx"):
        value = f"{value}.xlsx"
    if Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError("filename must not contain a directory")
    if not _REPORT_FILENAME.fullmatch(value):
        raise ValueError(
            "filename may contain only letters, numbers, spaces, dots, "
            "underscores, and hyphens"
        )
    stem = Path(value).stem.rstrip(" .").casefold()
    if stem in _WINDOWS_RESERVED:
        raise ValueError("filename uses a reserved system name")
    return value


def write_index_research_report(
    workspace: WorkspaceLike,
    backtest_result: object,
    *,
    filename: str,
    simulation: object | None = None,
    analytics: object | None = None,
    index_name: str | None = None,
    methodology_name: str | None = None,
    methodology_parameters: Mapping[str, Any] | None = None,
    data_sources: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    generated_at: object | None = None,
    overwrite: bool = False,
) -> Path:
    """Build and write one report under a named workspace's reports directory."""

    workspace_name, reports_path = _validate_workspace(workspace)
    safe_filename = validate_report_filename(filename)
    reports_path.mkdir(parents=True, exist_ok=True)
    destination = Path(workspace.report_path(safe_filename)).resolve()
    if destination.parent != reports_path:
        raise ValueError("workspace.report_path returned a path outside reports/")
    payload = ReportPayload.from_backtest_result(
        backtest_result,
        simulation=simulation,
        analytics=analytics,
        index_name=index_name,
        methodology_name=methodology_name,
        methodology_parameters=methodology_parameters,
        data_sources=data_sources,
        workspace_name=workspace_name,
        generated_at=generated_at,
    )
    return ExcelReportRenderer().render(
        payload,
        destination,
        overwrite=overwrite,
    )


def _validate_workspace(workspace: WorkspaceLike) -> tuple[str, Path]:
    if not callable(getattr(workspace, "report_path", None)):
        raise TypeError(
            "workspace must expose a named root, reports_path, and report_path()"
        )
    raw_name = getattr(workspace, "name", None)
    if raw_name is None:
        raw_name = getattr(workspace, "workspace_name", None)
    raw_root = getattr(workspace, "path", None)
    if raw_root is None:
        raw_root = getattr(workspace, "workspace_path", None)
    raw_reports = getattr(workspace, "reports_path", None)
    if raw_name is None or raw_root is None or raw_reports is None:
        raise TypeError(
            "workspace must expose a named root, reports_path, and report_path()"
        )
    name = str(raw_name).strip()
    if not _WORKSPACE_NAME.fullmatch(name):
        raise ValueError("workspace has an invalid name")
    root = Path(os.fspath(raw_root)).resolve()
    reports = Path(os.fspath(raw_reports)).resolve()
    if reports.parent != root or reports.name != "reports":
        raise ValueError("workspace reports_path must be its reports/ directory")
    return name, reports


__all__ = [
    "WorkspaceLike",
    "validate_report_filename",
    "write_index_research_report",
]
