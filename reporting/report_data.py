"""Safe, provider-neutral data adapter for index research reports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import date, datetime
from enum import Enum
import math
import re
from typing import Any

import numpy as np
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

        reviews = _extract_reviews(backtest_result)
        all_weights = _extract_weights(backtest_result)
        index_id = _extract_index_id(reviews)
        schedule = _build_review_schedule(reviews, index_id)
        _validate_review_weight_dates(all_weights, schedule)
        latest_holdings = _build_latest_holdings(reviews)
        performance = _build_performance(simulation)
        exposures = _build_exposures(analytics)
        turnover = _build_turnover(analytics, simulation)
        attribution = _build_attribution(analytics)
        parameters = _build_methodology_parameters(methodology_parameters)
        source_table = _build_data_sources(data_sources)
        validation = _build_validation(reviews, analytics)
        overview = _build_overview(
            index_id=index_id,
            index_name=index_name,
            methodology_name=methodology_name,
            workspace_name=workspace_name,
            generated_at=generated_at,
            schedule=schedule,
            latest_holdings=latest_holdings,
            analytics=analytics,
        )
        return cls(
            index_id=index_id,
            overview=overview,
            review_schedule=schedule,
            latest_holdings=latest_holdings,
            all_review_weights=all_weights,
            performance=performance,
            exposures=exposures,
            turnover=turnover,
            attribution=attribution,
            methodology_parameters=parameters,
            data_sources=source_table,
            validation=validation,
        )


def _extract_reviews(backtest_result: object) -> dict[pd.Timestamp, object]:
    reviews = getattr(backtest_result, "reviews", None)
    if not isinstance(reviews, Mapping) or not reviews:
        raise ReportDataError("backtest_result.reviews must be a non-empty mapping")
    result: dict[pd.Timestamp, object] = {}
    for raw_date, context in reviews.items():
        effective_date = _normalise_date(raw_date, "review effective date")
        if effective_date in result:
            raise ReportDataError("backtest_result contains duplicate review dates")
        context_date = getattr(context, "effective_date", None)
        if context_date is None:
            raise ReportDataError("each review context must define effective_date")
        if _normalise_date(context_date, "context effective_date") != effective_date:
            raise ReportDataError(
                "review mapping key does not match the context effective_date"
            )
        result[effective_date] = context
    return dict(sorted(result.items()))


def _extract_weights(backtest_result: object) -> pd.DataFrame:
    frame = getattr(backtest_result, "weights", None)
    if not isinstance(frame, pd.DataFrame):
        raise ReportDataError("backtest_result.weights must be a pandas DataFrame")
    frame = _with_named_index_columns(frame)
    required = set(ALL_REVIEW_WEIGHTS_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ReportDataError(f"backtest weights are missing columns: {missing}")
    result = frame.loc[:, ALL_REVIEW_WEIGHTS_COLUMNS].copy()
    result["effective_date"] = pd.to_datetime(
        result["effective_date"], errors="raise"
    ).dt.normalize()
    result["index_weight"] = pd.to_numeric(
        result["index_weight"], errors="raise"
    )
    if result.duplicated(["effective_date", "instrument_id"]).any():
        raise ReportDataError(
            "backtest weights contain duplicate effective_date/instrument_id rows"
        )
    if not np.isfinite(result["index_weight"].to_numpy(dtype=float)).all():
        raise ReportDataError("backtest weights contain non-finite values")
    return result.sort_values(
        ["effective_date", "instrument_id"], kind="stable"
    ).reset_index(drop=True)


def _extract_index_id(reviews: Mapping[pd.Timestamp, object]) -> str:
    values = {
        str(getattr(context, "index_id", "")).strip()
        for context in reviews.values()
    }
    if "" in values or len(values) != 1:
        raise ReportDataError(
            "all review contexts must contain the same non-empty index_id"
        )
    return values.pop()


def _build_review_schedule(
    reviews: Mapping[pd.Timestamp, object],
    index_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for effective_date, context in reviews.items():
        reference_date = _normalise_date(
            getattr(context, "reference_date", None),
            "reference_date",
        )
        if reference_date > effective_date:
            raise ReportDataError("reference_date must not be after effective_date")
        constituents = getattr(context, "cons", None)
        if not isinstance(constituents, pd.DataFrame):
            raise ReportDataError("each review context must expose a constituent frame")
        rows.append(
            {
                "reference_date": reference_date,
                "effective_date": effective_date,
                "index_id": index_id,
                "universe_size": int(len(constituents)),
            }
        )
    return pd.DataFrame.from_records(rows, columns=REVIEW_SCHEDULE_COLUMNS)


def _validate_review_weight_dates(
    weights: pd.DataFrame,
    schedule: pd.DataFrame,
) -> None:
    actual = set(weights["effective_date"])
    expected = set(schedule["effective_date"])
    if actual != expected:
        raise ReportDataError(
            "backtest weights and review contexts contain different effective dates"
        )


def _build_latest_holdings(
    reviews: Mapping[pd.Timestamp, object],
) -> pd.DataFrame:
    effective_date = max(reviews)
    context = reviews[effective_date]
    raw = getattr(context, "cons", None)
    if not isinstance(raw, pd.DataFrame):
        raise ReportDataError("latest review context has no constituent frame")
    frame = _with_named_index_columns(raw)
    if "instrument_id" not in frame:
        raise ReportDataError(
            "latest holdings must use instrument_id as a column or named index"
        )
    if frame["instrument_id"].duplicated().any():
        raise ReportDataError("latest holdings contain duplicate instrument_id values")
    frame["reference_date"] = _normalise_date(
        getattr(context, "reference_date", None),
        "reference_date",
    )
    frame["effective_date"] = effective_date
    result = pd.DataFrame(index=frame.index)
    for column in LATEST_HOLDINGS_COLUMNS:
        result[column] = frame[column] if column in frame else pd.NA
    return result.reset_index(drop=True)


def _build_performance(simulation: object | None) -> pd.DataFrame:
    if simulation is None:
        return _empty(PERFORMANCE_COLUMNS)
    raw = getattr(simulation, "daily", None)
    if not isinstance(raw, pd.DataFrame):
        raise ReportDataError("simulation.daily must be a pandas DataFrame")
    if raw.empty:
        return _empty(PERFORMANCE_COLUMNS)
    frame = _with_named_index_columns(raw)
    if "business_date" not in frame:
        raise ReportDataError(
            "simulation.daily must use business_date as a column or named index"
        )
    core = {
        "index_price_return",
        "index_gross_total_return",
        "index_net_total_return",
        "benchmark_price_return",
        "benchmark_gross_total_return",
        "benchmark_net_total_return",
    }
    missing = sorted(core.difference(frame.columns))
    if missing:
        raise ReportDataError(f"simulation.daily is missing columns: {missing}")
    frame["business_date"] = pd.to_datetime(
        frame["business_date"], errors="raise"
    ).dt.normalize()
    for variant in ("price", "gross_total", "net_total"):
        active = f"active_{variant}_return"
        if active not in frame:
            frame[active] = (
                frame[f"index_{variant}_return"]
                - frame[f"benchmark_{variant}_return"]
            )
    result = pd.DataFrame(index=frame.index)
    for column in PERFORMANCE_COLUMNS:
        result[column] = frame[column] if column in frame else pd.NA
    _require_finite_numeric(
        result,
        [column for column in PERFORMANCE_COLUMNS if column != "business_date"],
        allow_missing=True,
    )
    return result.sort_values("business_date", kind="stable").reset_index(drop=True)


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
    return _normalise_exposure_result(result)


def _adapt_generic_exposure(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _with_named_index_columns(raw)
    required = set(EXPOSURE_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ReportDataError(f"analytics.exposures is missing columns: {missing}")
    return _normalise_exposure_result(frame.loc[:, EXPOSURE_COLUMNS].copy())


def _normalise_exposure_result(frame: pd.DataFrame) -> pd.DataFrame:
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


def _build_methodology_parameters(
    values: Mapping[str, Any] | None,
) -> pd.DataFrame:
    if values is None:
        return _empty(METHODOLOGY_PARAMETER_COLUMNS)
    if not isinstance(values, Mapping):
        raise ReportDataError("methodology_parameters must be a mapping")
    rows: list[dict[str, Any]] = []
    _flatten_parameters(values, rows=rows)
    return pd.DataFrame.from_records(rows, columns=METHODOLOGY_PARAMETER_COLUMNS)


def _flatten_parameters(
    values: Mapping[str, Any],
    *,
    rows: list[dict[str, Any]],
    prefix: str = "",
) -> None:
    for raw_key, value in values.items():
        key = str(raw_key).strip()
        if not _SAFE_FIELD.fullmatch(key):
            raise ReportDataError(f"unsafe methodology parameter name: {key!r}")
        full_key = f"{prefix}.{key}" if prefix else key
        _reject_sensitive_key(full_key)
        if isinstance(value, Mapping):
            _flatten_parameters(value, rows=rows, prefix=full_key)
        else:
            rows.append(
                {
                    "parameter": full_key,
                    "value": _safe_parameter_value(value),
                }
            )


def _build_data_sources(
    values: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> pd.DataFrame:
    if values is None:
        return _empty(DATA_SOURCE_COLUMNS)
    records: list[Mapping[str, Any]]
    if isinstance(values, Mapping):
        if "capability" in values or "provider_name" in values:
            records = [values]
        else:
            records = []
            for capability, specification in values.items():
                if isinstance(specification, str):
                    records.append(
                        {
                            "capability": capability,
                            "provider_name": specification,
                            "data_type": "",
                            "fields": (),
                        }
                    )
                elif isinstance(specification, Mapping):
                    records.append({**specification, "capability": capability})
                else:
                    raise ReportDataError(
                        "data source mappings must contain labels or mappings"
                    )
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray)
    ):
        records = list(values)
    else:
        raise ReportDataError("data_sources must be a mapping or sequence of mappings")

    rows: list[dict[str, Any]] = []
    allowed = set(DATA_SOURCE_COLUMNS)
    for record in records:
        if not isinstance(record, Mapping):
            raise ReportDataError("each data source must be a mapping")
        unknown = set(map(str, record)).difference(allowed)
        if unknown:
            raise ReportDataError(
                f"data source contains unsupported fields: {sorted(unknown)}"
            )
        capability = _safe_label_value(record.get("capability"), "capability")
        provider_name = _safe_label_value(
            record.get("provider_name"), "provider_name"
        )
        data_type_raw = record.get("data_type", "")
        data_type = (
            ""
            if data_type_raw in (None, "")
            else _safe_label_value(data_type_raw, "data_type")
        )
        fields_value = record.get("fields", ())
        if isinstance(fields_value, str):
            field_names = [fields_value]
        elif isinstance(fields_value, Iterable):
            field_names = list(fields_value)
        else:
            raise ReportDataError("data source fields must be a string or iterable")
        clean_fields: list[str] = []
        for raw_field in field_names:
            field_name = str(raw_field).strip()
            if not _SAFE_FIELD.fullmatch(field_name):
                raise ReportDataError(f"unsafe data source field: {field_name!r}")
            _reject_sensitive_key(field_name)
            clean_fields.append(field_name)
        rows.append(
            {
                "capability": capability,
                "provider_name": provider_name,
                "data_type": data_type,
                "fields": ", ".join(clean_fields),
            }
        )
    return pd.DataFrame.from_records(rows, columns=DATA_SOURCE_COLUMNS)


def _build_validation(
    reviews: Mapping[pd.Timestamp, object],
    analytics: object | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for effective_date, context in reviews.items():
        constituents = getattr(context, "cons")
        if "index_weight" not in constituents:
            rows.append(
                _validation_row(
                    effective_date,
                    "index_weight_present",
                    False,
                    "",
                    "The review does not contain index_weight.",
                )
            )
            continue
        weights = pd.to_numeric(constituents["index_weight"], errors="coerce")
        finite = bool(np.isfinite(weights.to_numpy(dtype=float)).all())
        non_negative = bool(finite and (weights >= 0).all())
        total = float(weights.sum()) if finite else math.nan
        sums_to_one = bool(finite and math.isclose(total, 1.0, abs_tol=1e-8))
        rows.extend(
            (
                _validation_row(
                    effective_date,
                    "finite_index_weights",
                    finite,
                    finite,
                    "",
                ),
                _validation_row(
                    effective_date,
                    "non_negative_index_weights",
                    non_negative,
                    float(weights.min()) if finite and len(weights) else "",
                    "",
                ),
                _validation_row(
                    effective_date,
                    "index_weights_sum_to_one",
                    sums_to_one,
                    total,
                    "",
                ),
            )
        )
    if analytics is not None:
        rows.extend(_adapt_analytics_validation(analytics))
    return pd.DataFrame.from_records(rows, columns=VALIDATION_COLUMNS)


def _adapt_analytics_validation(analytics: object) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw = getattr(analytics, "review_validation", None)
    if raw is not None:
        if not isinstance(raw, pd.DataFrame):
            raise ReportDataError("analytics.review_validation must be a DataFrame")
        if not raw.empty:
            frame = _with_named_index_columns(raw)
            if {"check", "status"}.issubset(frame.columns):
                for record in frame.to_dict(orient="records"):
                    rows.append(
                        {
                            "effective_date": record.get("effective_date", ""),
                            "check": str(record["check"]),
                            "status": str(record["status"]).upper(),
                            "value": _safe_parameter_value(record.get("value", "")),
                            "message": str(record.get("message", "")),
                        }
                    )
            else:
                date_column = next(
                    (
                        column
                        for column in ("effective_date", "reference_date")
                        if column in frame
                    ),
                    None,
                )
                for record in frame.to_dict(orient="records"):
                    for key, value in record.items():
                        if key == date_column:
                            continue
                        _reject_sensitive_key(str(key))
                        passed = bool(value) if isinstance(value, (bool, np.bool_)) else True
                        rows.append(
                            _validation_row(
                                record.get(date_column, ""),
                                f"analytics.{key}",
                                passed,
                                _safe_parameter_value(value),
                                "",
                            )
                        )
    diagnostics = getattr(analytics, "diagnostics", ()) or ()
    for diagnostic in diagnostics:
        level = str(getattr(diagnostic, "level", "info")).lower()
        code = str(getattr(diagnostic, "code", "analytics_diagnostic"))
        message = str(getattr(diagnostic, "message", ""))
        _reject_sensitive_key(code)
        rows.append(
            {
                "effective_date": "",
                "check": code,
                "status": "WARNING" if level == "warning" else "INFO",
                "value": "",
                "message": message,
            }
        )
    return rows


def _build_overview(
    *,
    index_id: str,
    index_name: str | None,
    methodology_name: str | None,
    workspace_name: str | None,
    generated_at: object | None,
    schedule: pd.DataFrame,
    latest_holdings: pd.DataFrame,
    analytics: object | None,
) -> pd.DataFrame:
    timestamp = (
        pd.Timestamp.now(tz="UTC")
        if generated_at is None
        else pd.Timestamp(generated_at)
    )
    rows: list[dict[str, Any]] = [
        {"field": "report_contract", "value": REPORT_CONTRACT},
        {"field": "index_id", "value": index_id},
        {"field": "generated_at", "value": timestamp},
        {"field": "review_count", "value": int(len(schedule))},
        {
            "field": "first_reference_date",
            "value": schedule["reference_date"].min(),
        },
        {
            "field": "first_effective_date",
            "value": schedule["effective_date"].min(),
        },
        {
            "field": "latest_effective_date",
            "value": schedule["effective_date"].max(),
        },
        {
            "field": "latest_constituent_count",
            "value": int(len(latest_holdings)),
        },
    ]
    for field_name, value in (
        ("index_name", index_name),
        ("methodology_name", methodology_name),
        ("workspace_name", workspace_name),
    ):
        if value is not None:
            rows.append(
                {
                    "field": field_name,
                    "value": _safe_label_value(value, field_name),
                }
            )
    performance = getattr(analytics, "performance", None) if analytics else None
    if isinstance(performance, pd.Series):
        for name, value in performance.items():
            key = str(name).strip()
            if not _SAFE_FIELD.fullmatch(key):
                raise ReportDataError(f"unsafe analytics performance name: {key!r}")
            _reject_sensitive_key(key)
            rows.append(
                {
                    "field": f"performance.{key}",
                    "value": _safe_parameter_value(value),
                }
            )
    return pd.DataFrame.from_records(rows, columns=OVERVIEW_COLUMNS)


def _validation_row(
    effective_date: object,
    check: str,
    passed: bool,
    value: Any,
    message: str,
) -> dict[str, Any]:
    return {
        "effective_date": effective_date,
        "check": check,
        "status": "PASS" if passed else "FAIL",
        "value": value,
        "message": message,
    }


def _normalise_date(value: object, label: str) -> pd.Timestamp:
    if value is None:
        raise ReportDataError(f"{label} is required")
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ReportDataError(f"{label} is not a valid date") from exc
    if pd.isna(result):
        raise ReportDataError(f"{label} is not a valid date")
    if result.tzinfo is not None:
        result = result.tz_convert("UTC").tz_localize(None)
    return result.normalize()


def _with_named_index_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(deep=True)
    index_names = [
        name
        for name in result.index.names
        if name is not None and name not in result.columns
    ]
    if index_names:
        result = result.reset_index()
    if result.columns.duplicated().any():
        raise ReportDataError("report input contains duplicate column names")
    for column in result.columns:
        _reject_sensitive_key(str(column))
    return result


def _first_present(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    label: str,
) -> str:
    for candidate in candidates:
        if candidate in frame:
            return candidate
    raise ReportDataError(f"could not identify {label}")


def _require_finite_numeric(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    allow_missing: bool = False,
) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        present = values.notna()
        if not allow_missing and not present.all():
            raise ReportDataError(f"{column} contains missing or non-numeric values")
        if present.any() and not np.isfinite(values[present].to_numpy(dtype=float)).all():
            raise ReportDataError(f"{column} contains non-finite values")


def _safe_label_value(value: object, label: str) -> str:
    if value is None:
        raise ReportDataError(f"{label} is required")
    result = str(value).strip()
    if (
        not _SAFE_LABEL.fullmatch(result)
        or ".." in result
        or "//" in result
    ):
        raise ReportDataError(f"unsafe {label}: {result!r}")
    return result


def _safe_parameter_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, Enum):
        return _safe_parameter_value(value.value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value) if isinstance(value, (float, np.floating)) else int(value)
        if isinstance(number, float) and not math.isfinite(number):
            raise ReportDataError("report parameters must be finite")
        return number
    if isinstance(value, str):
        if len(value) > 1_000:
            raise ReportDataError("report parameter strings are limited to 1,000 characters")
        return value
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return ", ".join(str(_safe_parameter_value(item)) for item in value)
    raise ReportDataError(
        f"unsupported report parameter value type: {type(value).__name__}"
    )


def _reject_sensitive_key(value: str) -> None:
    compact = re.sub(r"[^a-z0-9]", "", value.casefold())
    if any(part in compact for part in _SENSITIVE_KEY_PARTS):
        raise ReportDataError(f"sensitive field is not permitted in reports: {value!r}")


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


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
    "ALL_REVIEW_WEIGHTS_COLUMNS",
    "ATTRIBUTION_COLUMNS",
    "DATA_SOURCE_COLUMNS",
    "EXPOSURE_COLUMNS",
    "LATEST_HOLDINGS_COLUMNS",
    "METHODOLOGY_PARAMETER_COLUMNS",
    "OVERVIEW_COLUMNS",
    "PERFORMANCE_COLUMNS",
    "REPORT_CONTRACT",
    "REVIEW_SCHEDULE_COLUMNS",
    "ReportDataError",
    "ReportPayload",
    "SHEET_COLUMNS",
    "TURNOVER_COLUMNS",
    "VALIDATION_COLUMNS",
]
