"""Safe Excel reporting for ICAPA index research outputs."""

from .bundle import (
    LargeTablePolicy,
    ReportBundle,
    ReportBundleError,
    ReportBundleSpec,
    ReportFormat,
    write_report_bundle,
)
from .excel import (
    ExcelReportRenderer,
    ReportIntegrityError,
    ReportTemplateError,
    TEMPLATE_VERSION,
)
from .builders import (
    REPORT_CONTRACT,
    ReportDataError,
    ReportPayload,
    SHEET_COLUMNS,
)
from .excel import (
    WorkspaceLike,
    validate_report_filename,
    write_index_research_report,
)

__all__ = [
    "ExcelReportRenderer",
    "LargeTablePolicy",
    "REPORT_CONTRACT",
    "ReportBundle",
    "ReportBundleError",
    "ReportBundleSpec",
    "ReportDataError",
    "ReportFormat",
    "ReportIntegrityError",
    "ReportPayload",
    "ReportTemplateError",
    "SHEET_COLUMNS",
    "TEMPLATE_VERSION",
    "WorkspaceLike",
    "validate_report_filename",
    "write_index_research_report",
    "write_report_bundle",
]
