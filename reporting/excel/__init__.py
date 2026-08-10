"""Excel report rendering and workspace-confined report services."""

from .renderer import (
    ExcelReportRenderer,
    ReportIntegrityError,
    ReportTemplateError,
    TEMPLATE_VERSION,
)
from .service import (
    WorkspaceLike,
    validate_report_filename,
    write_index_research_report,
)

__all__ = [
    "ExcelReportRenderer",
    "ReportIntegrityError",
    "ReportTemplateError",
    "TEMPLATE_VERSION",
    "WorkspaceLike",
    "validate_report_filename",
    "write_index_research_report",
]
