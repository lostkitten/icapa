"""Atomic, multi-format report bundles for index research runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import os as _os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any

from icapa.analytics import AnalyticsRunResult, ResearchComparison
from .excel import ExcelReportRenderer
from .builders import ReportPayload
from .builders.bundle_constants import _SAFE_NAME
from .contracts import ReportBundleError

# Retain the module-level test seam used to exercise atomic publish rollback.
os = _os


class ReportFormat(str, Enum):
    """Supported report-bundle formats."""

    XLSX = "xlsx"
    JSON = "json"
    PARQUET = "parquet"


class LargeTablePolicy(str, Enum):
    """Excel handling for tables beyond one worksheet."""

    SPLIT = "split"


@dataclass(frozen=True, slots=True)
class ReportBundleSpec:
    """Configuration for a deterministic research deliverable."""

    name: str = "index_research"
    contract_version: str = "2.0"
    formats: tuple[ReportFormat, ...] = (
        ReportFormat.XLSX,
        ReportFormat.JSON,
        ReportFormat.PARQUET,
    )
    include_raw_tables: bool = True
    include_comparison: bool = True
    overwrite: bool = False
    excel_large_table_policy: LargeTablePolicy = LargeTablePolicy.SPLIT
    include_sensitive_values: bool = False

    def __post_init__(self) -> None:
        if not _SAFE_NAME.fullmatch(self.name):
            raise ValueError(
                "report bundle name must use letters, numbers, dots, "
                "underscores, or hyphens"
            )
        if not self.contract_version:
            raise ValueError("report contract_version must not be empty")
        formats = tuple(ReportFormat(item) for item in self.formats)
        if not formats or len(formats) != len(set(formats)):
            raise ValueError("report formats must be non-empty and unique")
        object.__setattr__(self, "formats", formats)
        object.__setattr__(
            self,
            "excel_large_table_policy",
            LargeTablePolicy(self.excel_large_table_policy),
        )
        if self.include_sensitive_values:
            raise ValueError(
                "report bundles never include sensitive configuration values"
            )


@dataclass(frozen=True, slots=True)
class ReportBundle:
    """Paths and checksums for one completed report bundle."""

    bundle_id: str
    path: Path
    files: Mapping[str, Path]
    checksums: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))
        object.__setattr__(
            self,
            "checksums",
            MappingProxyType(dict(self.checksums)),
        )


from .builders.bundle_tables import (
    _append_research_sheets, _collect_tables, _report_table_identity,
    _split_v1_payload_for_v2, _summary_payload,
)
from .builders.bundle_io import (
    _file_checksum, _parquet_table_file_names, _publish_report_bundle,
    _write_json, _write_parquet,
)
from .builders.security import (
    _manifest_timestamp, _sanitize_output_value, _sanitize_report_payload,
    _workspace_name, _workspace_reports_path,
)

def write_report_bundle(
    workspace: object,
    backtest_result: object,
    *,
    simulation: object | None = None,
    analytics: AnalyticsRunResult | object | None = None,
    comparison: ResearchComparison | None = None,
    run_manifest: Mapping[str, Any] | object | None = None,
    index_name: str | None = None,
    methodology_name: str | None = None,
    methodology_parameters: Mapping[str, Any] | None = None,
    data_sources: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    spec: ReportBundleSpec | None = None,
) -> ReportBundle:
    """Create an atomic Excel, JSON, and Parquet research bundle."""

    selected = spec or ReportBundleSpec()
    reports_path = _workspace_reports_path(workspace)
    manifest_payload = _sanitize_output_value(run_manifest or {})
    safe_methodology_parameters = _sanitize_output_value(
        methodology_parameters or {}
    )
    safe_data_sources = _sanitize_output_value(
        () if data_sources is None else data_sources
    )
    generated_at = _manifest_timestamp(run_manifest)
    v1_analytics = (
        analytics.legacy_result
        if isinstance(analytics, AnalyticsRunResult)
        else analytics
    )
    payload = _sanitize_report_payload(
        ReportPayload.from_backtest_result(
            backtest_result,
            simulation=simulation,
            analytics=v1_analytics,
            index_name=index_name,
            methodology_name=methodology_name,
            methodology_parameters=safe_methodology_parameters,
            data_sources=safe_data_sources,
            workspace_name=_workspace_name(workspace),
            generated_at=generated_at,
        )
    )
    tables = _collect_tables(
        payload,
        analytics,
        comparison if selected.include_comparison else None,
        manifest_payload,
    )
    identity = {
        "contract_version": selected.contract_version,
        "index_id": payload.index_id,
        "manifest": manifest_payload,
        "report_specification": {
            "formats": [item.value for item in selected.formats],
            "include_raw_tables": selected.include_raw_tables,
            "include_comparison": selected.include_comparison,
            "excel_large_table_policy": (
                selected.excel_large_table_policy.value
            ),
        },
        "tables": {
            name: _report_table_identity(name, frame)
            for name, frame in sorted(tables.items())
        },
    }
    digest = sha256(
        json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    bundle_id = f"{selected.name}-{digest[:12]}"
    destination = reports_path.joinpath(bundle_id)
    if destination.exists() and not selected.overwrite:
        raise FileExistsError(f"report bundle already exists: {bundle_id}")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{bundle_id}.", dir=reports_path)
    )
    try:
        files: dict[str, Path] = {}
        if ReportFormat.XLSX in selected.formats:
            workbook_path = temporary.joinpath("report.xlsx")
            excel_payload, v1_overflow = _split_v1_payload_for_v2(payload)
            ExcelReportRenderer().render(excel_payload, workbook_path)
            _append_research_sheets(
                workbook_path,
                analytics,
                comparison if selected.include_comparison else None,
                manifest_payload,
                v1_overflow=v1_overflow,
            )
            files["report.xlsx"] = workbook_path

        summary = _summary_payload(
            payload,
            analytics,
            comparison if selected.include_comparison else None,
            manifest_payload,
            contract_version=selected.contract_version,
        )
        if ReportFormat.JSON in selected.formats:
            summary_path = temporary.joinpath("summary.json")
            _write_json(summary_path, summary)
            files["summary.json"] = summary_path
            manifest_path = temporary.joinpath("manifest.json")
            _write_json(manifest_path, manifest_payload)
            files["manifest.json"] = manifest_path

        if (
            ReportFormat.PARQUET in selected.formats
            and selected.include_raw_tables
        ):
            tables_path = temporary.joinpath("tables")
            tables_path.mkdir()
            table_file_names = _parquet_table_file_names(tables)
            for table_name, frame in sorted(tables.items()):
                file_name = table_file_names[table_name]
                path = tables_path.joinpath(file_name)
                _write_parquet(path, frame)
                files[f"tables/{file_name}"] = path

        checksums = {
            relative: _file_checksum(path)
            for relative, path in sorted(files.items())
        }
        checksums_path = temporary.joinpath("checksums.json")
        _write_json(checksums_path, checksums)
        files["checksums.json"] = checksums_path

        _publish_report_bundle(
            temporary,
            destination,
            overwrite=selected.overwrite,
        )
        completed_files = {
            relative: destination.joinpath(relative)
            for relative in files
        }
        completed_checksums = {
            relative: _file_checksum(path)
            for relative, path in completed_files.items()
            if relative != "checksums.json"
        }
        return ReportBundle(
            bundle_id=bundle_id,
            path=destination,
            files=completed_files,
            checksums=completed_checksums,
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise



__all__ = [
    "LargeTablePolicy", "ReportBundle", "ReportBundleError",
    "ReportBundleSpec", "ReportFormat", "write_report_bundle",
]
