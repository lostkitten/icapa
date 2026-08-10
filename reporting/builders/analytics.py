"""Safe, provider-neutral data adapter for index research reports."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..contracts import (
    ATTRIBUTION_COLUMNS,
    EXPOSURE_COLUMNS,
    TURNOVER_COLUMNS,
    ReportDataError,
)
from .validation import (
    _empty,
    _first_present,
    _require_finite_numeric,
    _with_named_index_columns,
)

def _build_exposures(analytics: object | None) -> pd.DataFrame:
    if analytics is None:
        return _empty(EXPOSURE_COLUMNS)
    frames: list[pd.DataFrame] = []
    for attribute, exposure_type in (
        ("country_exposures", "country"),
        ("industry_exposures", "industry"),
    ):
        raw = getattr(analytics, attribute, None)
        if raw is None:
            continue
        if not isinstance(raw, pd.DataFrame):
            raise ReportDataError(f"analytics.{attribute} must be a DataFrame")
        if not raw.empty:
            frames.append(_adapt_exposure(raw, exposure_type))
    if not frames:
        generic = getattr(analytics, "exposures", None)
        if generic is not None:
            if not isinstance(generic, pd.DataFrame):
                raise ReportDataError("analytics.exposures must be a DataFrame")
            if not generic.empty:
                frames.append(_adapt_generic_exposure(generic))
    if not frames:
        return _empty(EXPOSURE_COLUMNS)
    return pd.concat(frames, ignore_index=True).loc[:, EXPOSURE_COLUMNS]


def _adapt_exposure(raw: pd.DataFrame, exposure_type: str) -> pd.DataFrame:
    frame = _with_named_index_columns(raw)
    date_column = _first_present(
        frame,
        ("effective_date", "reference_date", "period", "business_date"),
        f"{exposure_type} exposure date",
    )
    name_column = _first_present(
        frame,
        (exposure_type, "exposure_name", "classification", "group"),
        f"{exposure_type} exposure name",
    )
    portfolio_column = _first_present(
        frame,
        ("portfolio_exposure", "index_weight", "portfolio_weight"),
        f"{exposure_type} portfolio exposure",
    )
    benchmark_column = _first_present(
        frame,
        ("benchmark_exposure", "benchmark_weight"),
        f"{exposure_type} benchmark exposure",
    )
    active_column = next(
        (
            column
            for column in ("active_exposure", "active_weight")
            if column in frame
        ),
        None,
    )
    result = pd.DataFrame(
        {
            "effective_date": frame[date_column],
            "exposure_type": exposure_type,
            "exposure_name": frame[name_column],
            "portfolio_exposure": frame[portfolio_column],
            "benchmark_exposure": frame[benchmark_column],
        }
    )
    result["active_exposure"] = (
        frame[active_column]
        if active_column
        else result["portfolio_exposure"] - result["benchmark_exposure"]
    )
    return _normalize_exposure_result(result)


def _adapt_generic_exposure(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _with_named_index_columns(raw)
    required = set(EXPOSURE_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ReportDataError(f"analytics.exposures is missing columns: {missing}")
    return _normalize_exposure_result(frame.loc[:, EXPOSURE_COLUMNS].copy())


def _normalize_exposure_result(frame: pd.DataFrame) -> pd.DataFrame:
    frame["effective_date"] = pd.to_datetime(
        frame["effective_date"], errors="raise"
    ).dt.normalize()
    _require_finite_numeric(
        frame,
        ("portfolio_exposure", "benchmark_exposure", "active_exposure"),
    )
    return frame.loc[:, EXPOSURE_COLUMNS].sort_values(
        ["effective_date", "exposure_type", "exposure_name"],
        kind="stable",
    )


def _build_turnover(
    analytics: object | None,
    simulation: object | None,
) -> pd.DataFrame:
    raw = getattr(analytics, "formal_turnover", None) if analytics else None
    if raw is None and analytics is not None:
        raw = getattr(analytics, "turnover", None)
    if (raw is None or getattr(raw, "empty", False)) and simulation is not None:
        raw = getattr(simulation, "rebalances", None)
    if raw is None:
        return _empty(TURNOVER_COLUMNS)
    if not isinstance(raw, pd.DataFrame):
        raise ReportDataError("turnover data must be a pandas DataFrame")
    if raw.empty:
        return _empty(TURNOVER_COLUMNS)
    frame = _with_named_index_columns(raw)
    date_column = _first_present(
        frame,
        ("effective_date", "applied_business_date", "business_date"),
        "turnover date",
    )
    one_way_column = _first_present(
        frame,
        (
            "one_way_turnover",
            "formal_one_way_turnover",
            "index_turnover",
            "turnover",
        ),
        "one-way turnover",
    )
    result = pd.DataFrame(
        {
            "effective_date": frame[date_column],
            "one_way_turnover": frame[one_way_column],
        }
    )
    result["two_way_turnover"] = (
        frame["two_way_turnover"]
        if "two_way_turnover" in frame
        else 2.0 * result["one_way_turnover"]
    )
    result["effective_date"] = pd.to_datetime(
        result["effective_date"], errors="raise"
    ).dt.normalize()
    _require_finite_numeric(
        result,
        ("one_way_turnover", "two_way_turnover"),
        allow_missing=True,
    )
    return result.loc[:, TURNOVER_COLUMNS].sort_values(
        "effective_date", kind="stable"
    ).reset_index(drop=True)


def _build_attribution(analytics: object | None) -> pd.DataFrame:
    if analytics is None:
        return _empty(ATTRIBUTION_COLUMNS)
    generic = getattr(analytics, "attribution", None)
    if generic is not None:
        if not isinstance(generic, pd.DataFrame):
            raise ReportDataError("analytics.attribution must be a DataFrame")
        if generic.empty:
            return _empty(ATTRIBUTION_COLUMNS)
        frame = _with_named_index_columns(generic)
        missing = sorted(set(ATTRIBUTION_COLUMNS).difference(frame.columns))
        if missing:
            raise ReportDataError(f"analytics.attribution is missing columns: {missing}")
        result = frame.loc[:, ATTRIBUTION_COLUMNS].copy()
        _require_finite_numeric(result, ("contribution",))
        return result

    brinson = getattr(analytics, "brinson", None)
    detail = getattr(brinson, "detail", None) if brinson is not None else None
    if detail is None or getattr(detail, "empty", True):
        return _empty(ATTRIBUTION_COLUMNS)
    if not isinstance(detail, pd.DataFrame):
        raise ReportDataError("analytics.brinson.detail must be a DataFrame")
    frame = _with_named_index_columns(detail)
    date_column = _first_present(
        frame,
        ("business_date", "period", "effective_date"),
        "attribution date",
    )
    classification_column = next(
        (
            column
            for column in ("industry", "country", "classification", "group")
            if column in frame
        ),
        None,
    )
    components = [
        column
        for column in ("allocation", "selection", "interaction", "total_attribution")
        if column in frame
    ]
    if not components:
        raise ReportDataError("Brinson detail contains no attribution components")
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        label = (
            str(record[classification_column])
            if classification_column is not None
            else "total"
        )
        for component in components:
            rows.append(
                {
                    "business_date": record[date_column],
                    "component": f"{label}.{component}",
                    "contribution": record[component],
                }
            )
    result = pd.DataFrame.from_records(rows, columns=ATTRIBUTION_COLUMNS)
    _require_finite_numeric(result, ("contribution",))
    return result



__all__ = [name for name in globals() if name.startswith("_")]
