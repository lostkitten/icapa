"""Build the canonical report payload from completed research results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from ..contracts import ReportPayload
from .analytics import (
    _build_attribution, _build_exposures, _build_turnover,
)
from .metadata import (
    _build_data_sources, _build_methodology_parameters, _build_validation,
)
from .overview import _build_overview
from .reviews import (
    _build_latest_holdings, _build_performance, _build_review_schedule,
    _extract_index_id,
    _extract_reviews, _extract_weights, _validate_review_weight_dates,
)

PayloadT = TypeVar("PayloadT", bound=ReportPayload)

def build_report_payload(
    payload_type: type[PayloadT],
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
) -> PayloadT:
    """Adapt explicit outputs without traversing private implementation state."""

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
        index_id=index_id, index_name=index_name,
        methodology_name=methodology_name, workspace_name=workspace_name,
        generated_at=generated_at, schedule=schedule,
        latest_holdings=latest_holdings, analytics=analytics,
    )
    return payload_type(
        index_id=index_id, overview=overview, review_schedule=schedule,
        latest_holdings=latest_holdings, all_review_weights=all_weights,
        performance=performance, exposures=exposures, turnover=turnover,
        attribution=attribution, methodology_parameters=parameters,
        data_sources=source_table, validation=validation,
    )

__all__ = ["build_report_payload"]
