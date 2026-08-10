"""Safe, provider-neutral data adapter for index research reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import re
from typing import Any

import pandas as pd


REPORT_CONTRACT = "icapa-index-research-report"

OVERVIEW_COLUMNS = ("field", "value")
REVIEW_SCHEDULE_COLUMNS = (
    "reference_date",
    "effective_date",
    "index_id",
    "universe_size",
)
LATEST_HOLDINGS_COLUMNS = (
    "instrument_id",
    "name",
    "country",
    "industry",
    "shares",
    "free_float",
    "price",
    "currency",
    "base_currency",
    "fx_rate",
    "market_cap",
    "benchmark_weight",
    "reference_date",
    "effective_date",
    "index_weight",
    "excluded",
    "exclusion_reason",
)
ALL_REVIEW_WEIGHTS_COLUMNS = (
    "effective_date",
    "instrument_id",
    "index_weight",
)
PERFORMANCE_COLUMNS = (
    "business_date",
    "index_price_return",
    "index_gross_total_return",
    "index_net_total_return",
    "benchmark_price_return",
    "benchmark_gross_total_return",
    "benchmark_net_total_return",
    "active_price_return",
    "active_gross_total_return",
    "active_net_total_return",
    "index_price_level",
    "index_gross_total_level",
    "index_net_total_level",
    "benchmark_price_level",
    "benchmark_gross_total_level",
    "benchmark_net_total_level",
)
EXPOSURE_COLUMNS = (
    "effective_date",
    "exposure_type",
    "exposure_name",
    "portfolio_exposure",
    "benchmark_exposure",
    "active_exposure",
)
TURNOVER_COLUMNS = (
    "effective_date",
    "one_way_turnover",
    "two_way_turnover",
)
ATTRIBUTION_COLUMNS = ("business_date", "component", "contribution")
METHODOLOGY_PARAMETER_COLUMNS = ("parameter", "value")
DATA_SOURCE_COLUMNS = ("capability", "provider_name", "data_type", "fields")
VALIDATION_COLUMNS = ("effective_date", "check", "status", "value", "message")

SHEET_COLUMNS: dict[str, tuple[str, ...]] = {
    "Overview": OVERVIEW_COLUMNS,
    "Review Schedule": REVIEW_SCHEDULE_COLUMNS,
    "Latest Holdings": LATEST_HOLDINGS_COLUMNS,
    "All Review Weights": ALL_REVIEW_WEIGHTS_COLUMNS,
    "Performance": PERFORMANCE_COLUMNS,
    "Exposures": EXPOSURE_COLUMNS,
    "Turnover": TURNOVER_COLUMNS,
    "Attribution": ATTRIBUTION_COLUMNS,
    "Methodology Parameters": METHODOLOGY_PARAMETER_COLUMNS,
    "Data Sources": DATA_SOURCE_COLUMNS,
    "Validation": VALIDATION_COLUMNS,
}

_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._&(),/+-]{0,127}$")
_SAFE_FIELD = re.compile(r"^[A-Za-z][A-Za-z0-9_.]{0,127}$")
_SENSITIVE_KEY_PARTS = {
    "account",
    "connection",
    "credential",
    "database",
    "dsn",
    "executor",
    "host",
    "oauth",
    "password",
    "path",
    "privatekey",
    "providerparameter",
    "query",
    "role",
    "schema",
    "secret",
    "server",
    "sql",
    "token",
    "url",
    "user",
    "warehouse",
}


class ReportDataError(ValueError):
    """Raised when an object cannot be adapted to the safe report contract."""


class ReportBundleError(RuntimeError):
    """Raised when a report bundle cannot be produced safely."""


@dataclass(frozen=True, slots=True)
class ReportPayload:
    """Whitelisted tables ready for deterministic workbook rendering."""

    index_id: str
    overview: pd.DataFrame
    review_schedule: pd.DataFrame
    latest_holdings: pd.DataFrame
    all_review_weights: pd.DataFrame
    performance: pd.DataFrame
    exposures: pd.DataFrame
    turnover: pd.DataFrame
    attribution: pd.DataFrame
    methodology_parameters: pd.DataFrame
    data_sources: pd.DataFrame
    validation: pd.DataFrame

    def __post_init__(self) -> None:
        if not isinstance(self.index_id, str) or not self.index_id.strip():
            raise ReportDataError("index_id must be a non-empty string")
        expected = {
            item.name: SHEET_COLUMNS[_sheet_name_for_field(item.name)]
            for item in fields(self)
            if item.name != "index_id"
        }
        for field_name, columns in expected.items():
            frame = getattr(self, field_name)
            if not isinstance(frame, pd.DataFrame):
                raise ReportDataError(f"{field_name} must be a pandas DataFrame")
            if tuple(frame.columns) != columns:
                raise ReportDataError(
                    f"{field_name} columns must be exactly {list(columns)}"
                )
            object.__setattr__(self, field_name, frame.copy(deep=True))

    def sheet_frames(self) -> dict[str, pd.DataFrame]:
        """Return defensive copies keyed by the fixed workbook sheet names."""

        return {
            sheet_name: getattr(self, _field_name_for_sheet(sheet_name)).copy(deep=True)
            for sheet_name in SHEET_COLUMNS
        }

    @classmethod
    def from_backtest_result(
        cls,
        backtest_result: object,
        *,
        simulation: object | None = None,
        analytics: object | None = None,
        index_name: str | None = None,
        methodology_name: str | None = None,
        methodology_parameters: Mapping[str, Any] | None = None,
        data_sources: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
        workspace_name: str | None = None,
        generated_at: object | None = None,
    ) -> "ReportPayload":
        """Adapt explicit research outputs without traversing private state."""

        from .builders.payload import build_report_payload

        return build_report_payload(
            cls,
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



def _field_name_for_sheet(sheet_name: str) -> str:
    return {
        "Overview": "overview",
        "Review Schedule": "review_schedule",
        "Latest Holdings": "latest_holdings",
        "All Review Weights": "all_review_weights",
        "Performance": "performance",
        "Exposures": "exposures",
        "Turnover": "turnover",
        "Attribution": "attribution",
        "Methodology Parameters": "methodology_parameters",
        "Data Sources": "data_sources",
        "Validation": "validation",
    }[sheet_name]


def _sheet_name_for_field(field_name: str) -> str:
    return {
        value: key
        for key, value in (
            (sheet_name, _field_name_for_sheet(sheet_name))
            for sheet_name in SHEET_COLUMNS
        )
    }[field_name]


__all__ = [
    "REPORT_CONTRACT",
    "ReportBundleError",
    "ReportDataError",
    "ReportPayload",
    "SHEET_COLUMNS",
]
