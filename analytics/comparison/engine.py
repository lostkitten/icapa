"""Baseline-to-candidate comparisons for index research runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import AnalyticsResult, AnalyticsValidationError
from ..plugins import AnalyticsRunResult


class ReviewAlignment(str, Enum):
    """How review dates are aligned across compared runs."""

    INTERSECTION = "intersection"


class DateAlignment(str, Enum):
    """How daily simulation dates are aligned."""

    INTERSECTION = "intersection"


class InstrumentAlignment(str, Enum):
    """How constituent universes are aligned."""

    UNION_FILL_ZERO = "union_fill_zero"


class CompatibilityPolicy(str, Enum):
    """How non-fatal lineage differences are handled."""

    REPORT_DIFFERENCES = "report_differences"
    REQUIRE_EQUAL = "require_equal"


@dataclass(frozen=True, slots=True)
class ComparisonSpec:
    """Deterministic alignment and validation choices."""

    review_alignment: ReviewAlignment = ReviewAlignment.INTERSECTION
    business_date_alignment: DateAlignment = DateAlignment.INTERSECTION
    instrument_alignment: InstrumentAlignment = InstrumentAlignment.UNION_FILL_ZERO
    classifications: tuple[str, ...] = ("country", "industry")
    weight_tolerance: float = 1e-8
    compatibility_policy: CompatibilityPolicy = CompatibilityPolicy.REPORT_DIFFERENCES

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "review_alignment",
            ReviewAlignment(self.review_alignment),
        )
        object.__setattr__(
            self,
            "business_date_alignment",
            DateAlignment(self.business_date_alignment),
        )
        object.__setattr__(
            self,
            "instrument_alignment",
            InstrumentAlignment(self.instrument_alignment),
        )
        object.__setattr__(
            self,
            "compatibility_policy",
            CompatibilityPolicy(self.compatibility_policy),
        )
        if self.weight_tolerance <= 0:
            raise AnalyticsValidationError("weight_tolerance must be positive")


@dataclass(frozen=True, slots=True)
class ComparisonInput:
    """One named run supplied to the comparison engine."""

    name: str
    backtest: object
    simulation: object | None = None
    analytics: AnalyticsResult | AnalyticsRunResult | None = None
    manifest: Mapping[str, Any] | object | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise AnalyticsValidationError("comparison input name must not be empty")


@dataclass(frozen=True, slots=True)
class ResearchComparison:
    """Complete baseline-to-candidate comparison tables."""

    baseline_name: str
    candidate_names: tuple[str, ...]
    compatibility: pd.DataFrame
    overview: pd.DataFrame
    parameter_differences: pd.DataFrame
    review_coverage: pd.DataFrame
    constituent_changes: pd.DataFrame
    weight_differences: pd.DataFrame
    exposure_differences: pd.DataFrame
    performance_differences: pd.DataFrame
    turnover_differences: pd.DataFrame
    validation_differences: pd.DataFrame
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "compatibility",
            "overview",
            "parameter_differences",
            "review_coverage",
            "constituent_changes",
            "weight_differences",
            "exposure_differences",
            "performance_differences",
            "turnover_differences",
            "validation_differences",
        ):
            object.__setattr__(self, name, getattr(self, name).copy(deep=True))
        object.__setattr__(self, "candidate_names", tuple(self.candidate_names))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    def tables(self) -> Mapping[str, pd.DataFrame]:
        """Return defensive copies for report and artifact writers."""

        return MappingProxyType(
            {
                name: getattr(self, name).copy(deep=True)
                for name in (
                    "compatibility",
                    "overview",
                    "parameter_differences",
                    "review_coverage",
                    "constituent_changes",
                    "weight_differences",
                    "exposure_differences",
                    "performance_differences",
                    "turnover_differences",
                    "validation_differences",
                )
            }
        )


class ComparisonEngine:
    """Compare one baseline with one or more candidate research results."""

    def compare(
        self,
        baseline: ComparisonInput,
        candidates: Sequence[ComparisonInput],
        *,
        spec: ComparisonSpec | None = None,
    ) -> ResearchComparison:
        """Build deterministic, zero-filled constituent comparisons."""

        selected_spec = spec or ComparisonSpec()
        candidate_inputs = tuple(candidates)
        if not candidate_inputs:
            raise AnalyticsValidationError(
                "at least one comparison candidate is required"
            )
        names = [item.name for item in candidate_inputs]
        if len(names) != len(set(names)) or baseline.name in names:
            raise AnalyticsValidationError("comparison input names must be unique")

        baseline_weights = _weights(baseline.backtest)
        compatibility_rows: list[dict[str, Any]] = []
        overview_rows: list[dict[str, Any]] = []
        parameter_rows: list[dict[str, Any]] = []
        coverage_rows: list[dict[str, Any]] = []
        change_frames: list[pd.DataFrame] = []
        weight_frames: list[pd.DataFrame] = []
        exposure_frames: list[pd.DataFrame] = []
        performance_frames: list[pd.DataFrame] = []
        turnover_frames: list[pd.DataFrame] = []
        validation_frames: list[pd.DataFrame] = []
        diagnostics: list[str] = []

        for candidate in candidate_inputs:
            candidate_weights = _weights(candidate.backtest)
            compatibility = _compatibility_rows(baseline, candidate)
            compatibility_rows.extend(compatibility)
            fatal = [item for item in compatibility if item["severity"] == "error"]
            differences = [
                item for item in compatibility if item["status"] == "different"
            ]
            if fatal:
                fields = ", ".join(item["field"] for item in fatal)
                raise AnalyticsValidationError(
                    f"comparison inputs are incompatible for {candidate.name}: {fields}"
                )
            if (
                selected_spec.compatibility_policy is CompatibilityPolicy.REQUIRE_EQUAL
                and differences
            ):
                fields = ", ".join(item["field"] for item in differences)
                raise AnalyticsValidationError(
                    f"comparison requires equal lineage for {candidate.name}: {fields}"
                )

            common_dates = (
                baseline_weights.index.get_level_values("effective_date")
                .unique()
                .intersection(
                    candidate_weights.index.get_level_values("effective_date").unique()
                )
            )
            if common_dates.empty:
                raise AnalyticsValidationError(
                    f"baseline and {candidate.name} have no common review dates"
                )
            coverage_rows.append(
                {
                    "candidate": candidate.name,
                    "baseline_review_count": baseline_weights.index.get_level_values(
                        "effective_date"
                    ).nunique(),
                    "candidate_review_count": candidate_weights.index.get_level_values(
                        "effective_date"
                    ).nunique(),
                    "common_review_count": len(common_dates),
                    "first_common_effective_date": common_dates.min(),
                    "last_common_effective_date": common_dates.max(),
                }
            )
            weights, changes, summary = _weight_comparison(
                baseline_weights,
                candidate_weights,
                common_dates,
                candidate.name,
                selected_spec.weight_tolerance,
            )
            weight_frames.append(weights)
            change_frames.append(changes)
            overview_rows.extend(summary)
            parameter_rows.extend(_parameter_difference_rows(baseline, candidate))
            exposure = _exposure_comparison(
                baseline,
                candidate,
                candidate.name,
            )
            if not exposure.empty:
                exposure_frames.append(exposure)
            performance = _performance_comparison(
                baseline,
                candidate,
                candidate.name,
            )
            if not performance.empty:
                performance_frames.append(performance)
            turnover = _turnover_comparison(
                baseline,
                candidate,
                candidate.name,
            )
            if not turnover.empty:
                turnover_frames.append(turnover)
            validation = _validation_comparison(
                baseline,
                candidate,
                candidate.name,
            )
            if not validation.empty:
                validation_frames.append(validation)
            if differences:
                diagnostics.append(
                    f"{candidate.name} has {len(differences)} reported lineage "
                    "or configuration differences."
                )

        return ResearchComparison(
            baseline_name=baseline.name,
            candidate_names=tuple(names),
            compatibility=pd.DataFrame(compatibility_rows),
            overview=pd.DataFrame(overview_rows),
            parameter_differences=pd.DataFrame(parameter_rows),
            review_coverage=pd.DataFrame(coverage_rows),
            constituent_changes=_concat(change_frames),
            weight_differences=_concat(weight_frames),
            exposure_differences=_concat(exposure_frames),
            performance_differences=_concat(performance_frames),
            turnover_differences=_concat(turnover_frames),
            validation_differences=_concat(validation_frames),
            diagnostics=tuple(diagnostics),
        )


def compare_research_results(
    baseline: ComparisonInput,
    candidates: Sequence[ComparisonInput],
    *,
    spec: ComparisonSpec | None = None,
) -> ResearchComparison:
    """Convenience wrapper around :class:`ComparisonEngine`."""

    return ComparisonEngine().compare(baseline, candidates, spec=spec)


def _weights(backtest: object) -> pd.Series:
    frame = getattr(backtest, "weights", None)
    if not isinstance(frame, pd.DataFrame) or "index_weight" not in frame:
        raise AnalyticsValidationError(
            "comparison backtest weights must contain index_weight"
        )
    result = frame.copy(deep=True)
    if not isinstance(result.index, pd.MultiIndex):
        required = {"effective_date", "instrument_id", "index_weight"}
        if not required.issubset(result.reset_index().columns):
            raise AnalyticsValidationError(
                "comparison weights require effective_date and instrument_id"
            )
        result = result.reset_index().set_index(
            ["effective_date", "instrument_id"],
            verify_integrity=True,
        )
    if result.index.names != ["effective_date", "instrument_id"]:
        result = result.reorder_levels(["effective_date", "instrument_id"]).sort_index()
    values = pd.to_numeric(result["index_weight"], errors="raise").astype(float)
    dates = pd.to_datetime(values.index.get_level_values("effective_date")).normalize()
    values.index = pd.MultiIndex.from_arrays(
        [dates, values.index.get_level_values("instrument_id")],
        names=["effective_date", "instrument_id"],
    )
    return values.sort_index()


def _compatibility_rows(
    baseline: ComparisonInput,
    candidate: ComparisonInput,
) -> list[dict[str, Any]]:
    baseline_values = _comparison_metadata(baseline)
    candidate_values = _comparison_metadata(candidate)
    fields = sorted(set(baseline_values) | set(candidate_values))
    rows = []
    for field_name in fields:
        baseline_value = baseline_values.get(field_name)
        candidate_value = candidate_values.get(field_name)
        equal = baseline_value == candidate_value
        severity = (
            "error"
            if field_name in {"base_currency", "calendar_semantics"}
            and baseline_value is not None
            and candidate_value is not None
            and not equal
            else "info"
        )
        rows.append(
            {
                "candidate": candidate.name,
                "field": field_name,
                "baseline_value": baseline_value,
                "candidate_value": candidate_value,
                "status": "equal" if equal else "different",
                "severity": severity,
            }
        )
    return rows


def _comparison_metadata(item: ComparisonInput) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "index_id": _index_id(item.backtest),
    }
    manifest = item.manifest
    for name in (
        "base_currency",
        "calendar_semantics",
        "definition_fingerprint",
        "result_fingerprint",
        "data_snapshot",
    ):
        value = _manifest_value(manifest, name)
        if value is not None:
            metadata[name] = value
    return metadata


def _manifest_value(manifest: Mapping[str, Any] | object | None, name: str) -> Any:
    if manifest is None:
        return None
    direct = (
        manifest.get(name)
        if isinstance(manifest, Mapping)
        else getattr(manifest, name, None)
    )
    if direct is not None:
        return direct

    request = (
        manifest.get("request")
        if isinstance(manifest, Mapping)
        else getattr(manifest, "request", None)
    )
    definition = request.get("definition") if isinstance(request, Mapping) else None
    if name == "base_currency" and isinstance(definition, Mapping):
        return definition.get("base_currency")
    if name == "parameters" and isinstance(definition, Mapping):
        return definition.get("methodology_parameters")
    if name == "calendar_semantics":
        calendar = (
            manifest.get("calendar")
            if isinstance(manifest, Mapping)
            else getattr(manifest, "calendar", None)
        )
        if isinstance(calendar, Mapping) and calendar:
            return dict(calendar)
    return None


def _index_id(backtest: object) -> str | None:
    reviews = getattr(backtest, "reviews", None)
    if isinstance(reviews, Mapping) and reviews:
        return getattr(next(iter(reviews.values())), "index_id", None)
    return None


def _weight_comparison(
    baseline: pd.Series,
    candidate: pd.Series,
    dates: pd.DatetimeIndex,
    candidate_name: str,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    weights: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for effective_date in dates.sort_values():
        left = baseline.xs(effective_date, level="effective_date")
        right = candidate.xs(effective_date, level="effective_date")
        left, right = left.align(right, join="outer", fill_value=0.0)
        delta = right - left
        one_way = 0.5 * float(delta.abs().sum())
        summaries.append(
            {
                "candidate": candidate_name,
                "effective_date": effective_date,
                "one_way_weight_difference": one_way,
                "maximum_absolute_weight_difference": float(delta.abs().max()),
                "baseline_constituent_count": int((left > tolerance).sum()),
                "candidate_constituent_count": int((right > tolerance).sum()),
            }
        )
        for instrument_id in left.index:
            baseline_weight = float(left.loc[instrument_id])
            candidate_weight = float(right.loc[instrument_id])
            if baseline_weight <= tolerance < candidate_weight:
                status = "entrant"
            elif candidate_weight <= tolerance < baseline_weight:
                status = "exit"
            elif candidate_weight > baseline_weight + tolerance:
                status = "weight_increase"
            elif candidate_weight < baseline_weight - tolerance:
                status = "weight_decrease"
            else:
                status = "unchanged"
            row = {
                "candidate": candidate_name,
                "effective_date": effective_date,
                "instrument_id": instrument_id,
                "baseline_weight": baseline_weight,
                "candidate_weight": candidate_weight,
                "weight_difference": candidate_weight - baseline_weight,
                "status": status,
            }
            weights.append(row)
            if status != "unchanged":
                changes.append(row)
    return pd.DataFrame(weights), pd.DataFrame(changes), summaries


def _parameter_difference_rows(
    baseline: ComparisonInput,
    candidate: ComparisonInput,
) -> list[dict[str, Any]]:
    baseline_parameters = _manifest_value(baseline.manifest, "parameters")
    candidate_parameters = _manifest_value(candidate.manifest, "parameters")
    if not isinstance(baseline_parameters, Mapping):
        baseline_parameters = {}
    if not isinstance(candidate_parameters, Mapping):
        candidate_parameters = {}
    rows = []
    for name in sorted(set(baseline_parameters) | set(candidate_parameters)):
        left = baseline_parameters.get(name)
        right = candidate_parameters.get(name)
        if left != right:
            rows.append(
                {
                    "candidate": candidate.name,
                    "parameter": name,
                    "baseline_value": left,
                    "candidate_value": right,
                }
            )
    return rows


def _legacy_analytics(item: ComparisonInput) -> AnalyticsResult | None:
    if isinstance(item.analytics, AnalyticsResult):
        return item.analytics
    if isinstance(item.analytics, AnalyticsRunResult):
        return item.analytics.legacy_result
    return None


def _exposure_comparison(
    baseline: ComparisonInput,
    candidate: ComparisonInput,
    candidate_name: str,
) -> pd.DataFrame:
    left = _legacy_analytics(baseline)
    right = _legacy_analytics(candidate)
    frames: list[pd.DataFrame] = []
    if left is not None and right is not None:
        for classification, left_frame, right_frame in (
            ("country", left.country_exposures, right.country_exposures),
            ("industry", left.industry_exposures, right.industry_exposures),
        ):
            left_values = left_frame["portfolio_weight"].rename("baseline_exposure")
            right_values = right_frame["portfolio_weight"].rename("candidate_exposure")
            combined = pd.concat(
                [left_values, right_values],
                axis=1,
                join="inner",
            )
            combined["exposure_difference"] = (
                combined["candidate_exposure"] - combined["baseline_exposure"]
            )
            combined = combined.reset_index()
            combined.insert(0, "classification", classification)
            combined.insert(0, "candidate", candidate_name)
            frames.append(combined)

    left_research = _research_table(
        baseline,
        "factor_signal_exposure.exposures",
    )
    right_research = _research_table(
        candidate,
        "factor_signal_exposure.exposures",
    )
    if left_research is not None and right_research is not None:
        keys = ["effective_date", "exposure_type", "field"]
        required = set(keys + ["index_exposure"])
        if required.issubset(left_research) and required.issubset(right_research):
            left_values = left_research.loc[
                :,
                keys + ["index_exposure"],
            ].rename(columns={"index_exposure": "baseline_exposure"})
            right_values = right_research.loc[
                :,
                keys + ["index_exposure"],
            ].rename(columns={"index_exposure": "candidate_exposure"})
            combined = left_values.merge(
                right_values,
                on=keys,
                how="inner",
                validate="one_to_one",
            )
            combined["exposure_difference"] = (
                combined["candidate_exposure"] - combined["baseline_exposure"]
            )
            combined.insert(0, "classification", "factor_signal")
            combined.insert(0, "candidate", candidate_name)
            frames.append(combined)
    return _concat(frames)


def _performance_comparison(
    baseline: ComparisonInput,
    candidate: ComparisonInput,
    candidate_name: str,
) -> pd.DataFrame:
    left = _legacy_analytics(baseline)
    right = _legacy_analytics(candidate)
    if left is None or right is None:
        return pd.DataFrame()
    names = left.performance.index.intersection(right.performance.index)
    rows = [
        {
            "candidate": candidate_name,
            "metric": name,
            "baseline_value": left.performance.loc[name],
            "candidate_value": right.performance.loc[name],
            "difference": right.performance.loc[name] - left.performance.loc[name],
        }
        for name in names
        if np.isscalar(left.performance.loc[name])
        and np.isscalar(right.performance.loc[name])
    ]
    return pd.DataFrame(rows)


def _turnover_comparison(
    baseline: ComparisonInput,
    candidate: ComparisonInput,
    candidate_name: str,
) -> pd.DataFrame:
    left = _legacy_analytics(baseline)
    right = _legacy_analytics(candidate)
    if left is None or right is None:
        return pd.DataFrame()
    left_frame = _dated_value_frame(left.one_way_turnover, "one_way_turnover")
    right_frame = _dated_value_frame(right.one_way_turnover, "one_way_turnover")
    if left_frame.empty or right_frame.empty:
        return pd.DataFrame()
    combined = pd.concat(
        [
            left_frame.rename("baseline_turnover"),
            right_frame.rename("candidate_turnover"),
        ],
        axis=1,
        join="inner",
    ).dropna(how="all")
    combined["turnover_difference"] = (
        combined["candidate_turnover"] - combined["baseline_turnover"]
    )
    combined = combined.reset_index()
    combined.insert(0, "candidate", candidate_name)
    return combined


def _validation_comparison(
    baseline: ComparisonInput,
    candidate: ComparisonInput,
    candidate_name: str,
) -> pd.DataFrame:
    left = _legacy_analytics(baseline)
    right = _legacy_analytics(candidate)
    frames: list[pd.DataFrame] = []
    if left is not None and right is not None:
        left_frame = left.review_validation.add_prefix("baseline_")
        right_frame = right.review_validation.add_prefix("candidate_")
        combined = pd.concat(
            [left_frame, right_frame],
            axis=1,
            join="inner",
        )
        combined = combined.reset_index()
        combined.insert(0, "diagnostic_type", "review_validation")
        combined.insert(0, "candidate", candidate_name)
        frames.append(combined)

    targets = _diagnostic_comparison(
        baseline,
        candidate,
        candidate_name,
        table_name="target_attainment.detail",
        name_column="target_name",
        value_columns=(
            "requested_value",
            "achieved_value",
            "within_bounds",
        ),
        diagnostic_type="target_attainment",
    )
    if not targets.empty:
        frames.append(targets)
    constraints = _diagnostic_comparison(
        baseline,
        candidate,
        candidate_name,
        table_name="constraint_diagnostics.detail",
        name_column="constraint_name",
        value_columns=(
            "achieved_value",
            "violation",
            "binding",
            "violated",
        ),
        diagnostic_type="constraint",
    )
    if not constraints.empty:
        frames.append(constraints)
    return _concat(frames)


def _research_table(
    item: ComparisonInput,
    name: str,
) -> pd.DataFrame | None:
    if not isinstance(item.analytics, AnalyticsRunResult):
        return None
    table = item.analytics.tables().get(name)
    return table.copy(deep=True) if table is not None else None


def _diagnostic_comparison(
    baseline: ComparisonInput,
    candidate: ComparisonInput,
    candidate_name: str,
    *,
    table_name: str,
    name_column: str,
    value_columns: Sequence[str],
    diagnostic_type: str,
) -> pd.DataFrame:
    left = _research_table(baseline, table_name)
    right = _research_table(candidate, table_name)
    if left is None or right is None:
        return pd.DataFrame()
    keys = ["effective_date", name_column]
    required = set(keys + list(value_columns))
    if not required.issubset(left) or not required.issubset(right):
        return pd.DataFrame()
    left_values = left.loc[:, keys + list(value_columns)].rename(
        columns={column: f"baseline_{column}" for column in value_columns}
    )
    right_values = right.loc[:, keys + list(value_columns)].rename(
        columns={column: f"candidate_{column}" for column in value_columns}
    )
    result = left_values.merge(
        right_values,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if result.empty:
        return result
    result.insert(0, "diagnostic_type", diagnostic_type)
    result.insert(0, "candidate", candidate_name)
    return result


def _dated_value_frame(frame: pd.DataFrame, value_column: str) -> pd.Series:
    if not isinstance(frame, pd.DataFrame) or frame.empty or value_column not in frame:
        return pd.Series(dtype=float)
    for column in (
        "effective_date",
        "scheduled_effective_date",
        "applied_business_date",
    ):
        if column in frame:
            result = frame[[column, value_column]].copy()
            result[column] = pd.to_datetime(result[column]).dt.normalize()
            return result.set_index(column)[value_column]
    if frame.index.name:
        result = frame[value_column].copy()
        result.index = pd.to_datetime(result.index).normalize()
        return result
    return pd.Series(dtype=float)


def _concat(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if isinstance(frame, pd.DataFrame)]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True, sort=False)


__all__ = [
    "ComparisonEngine",
    "ComparisonInput",
    "ComparisonSpec",
    "CompatibilityPolicy",
    "DateAlignment",
    "InstrumentAlignment",
    "ResearchComparison",
    "ReviewAlignment",
    "compare_research_results",
]
