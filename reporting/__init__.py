"""Safe Excel reporting for ICAPA index research outputs."""

from .excel_report import (
    ExcelReportRenderer,
    ReportIntegrityError,
    ReportTemplateError,
    TEMPLATE_VERSION,
)
from .report_data import (
    REPORT_CONTRACT,
    ReportDataError,
    ReportPayload,
    SHEET_COLUMNS,
)
from .report_functions import (
    WorkspaceLike,
    validate_report_filename,
    write_index_research_report,
)

__all__ = [
    "ExcelReportRenderer",
    "REPORT_CONTRACT",
    "ReportDataError",
    "ReportIntegrityError",
    "ReportPayload",
    "ReportTemplateError",
    "SHEET_COLUMNS",
    "TEMPLATE_VERSION",
    "WorkspaceLike",
    "validate_report_filename",
    "write_index_research_report",
]
