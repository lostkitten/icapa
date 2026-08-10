"""Self-contained risk dashboard rendering."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class RiskDashboardData:
    """Tables required by the provider-neutral risk dashboard."""

    summary: pd.DataFrame
    contributions: pd.DataFrame
    diagnostics: pd.DataFrame

    def __post_init__(self) -> None:
        for name in ("summary", "contributions", "diagnostics"):
            frame = getattr(self, name)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame")
            object.__setattr__(self, name, frame.copy(deep=True))


class RiskDashboardRenderer:
    """Render auditable risk tables into a self-contained HTML document."""

    def render(
        self,
        data: RiskDashboardData,
        output_path: str | Path,
        *,
        title: str = "Index Research Risk Dashboard",
        overwrite: bool = False,
    ) -> Path:
        if not isinstance(data, RiskDashboardData):
            raise TypeError("data must be RiskDashboardData")
        destination = Path(output_path)
        if destination.suffix.casefold() != ".html":
            raise ValueError("risk dashboard output must use the .html extension")
        if destination.exists() and not overwrite:
            raise FileExistsError(f"dashboard already exists: {destination.name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            _html_document(data, title=title),
            encoding="utf-8",
        )
        return destination


def _html_document(data: RiskDashboardData, *, title: str) -> str:
    sections = (
        ("Risk Summary", data.summary),
        ("Risk Contributions", data.contributions),
        ("Diagnostics", data.diagnostics),
    )
    body = "\n".join(
        f"<section><h2>{escape(label)}</h2>{_safe_table(frame)}</section>"
        for label, frame in sections
    )
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{escape(title)}</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:2rem;color:#1f2937}"
        "h1,h2{color:#17365d}section{margin:2rem 0}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d1d5db;padding:.45rem;text-align:right}"
        "th{background:#1f4e78;color:white}th:first-child,td:first-child{text-align:left}"
        "</style></head><body>"
        f"<h1>{escape(title)}</h1>{body}</body></html>"
    )


def _safe_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>No data available.</p>"
    columns = [escape(str(column)) for column in frame.columns]
    header = "".join(f"<th>{column}</th>" for column in columns)
    rows: list[str] = []
    for values in frame.itertuples(index=False, name=None):
        cells = "".join(f"<td>{escape(_display(value))}</td>" for value in values)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _display(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


__all__ = ["RiskDashboardData", "RiskDashboardRenderer"]
