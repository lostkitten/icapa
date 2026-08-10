"""Side-effect-free core analytics over completed research results."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from ..attribution.brinson import calculate_brinson_attribution
from ..contracts import (AnalyticsDiagnostic, AnalyticsResult, AnalyticsValidationError, BrinsonInput)
from .validation import (
    annualized_return as _annualized_return,
    annualized_volatility as _annualized_volatility,
    empty_performance as _empty_performance,
    level_and_drawdown as _level_and_drawdown,
    normalize_date as _normalise_date,
    normalize_result_weights as _normalise_result_weights,
    use_business_date_index as _use_business_date_index,
    use_instrument_index as _use_instrument_index,
    validated_weight_series as _validated_weight_series,
)

_UNCLASSIFIED = "Unclassified"
_CANONICAL_RETURN_COLUMN_PAIRS = (
    ("index_net_total_return", "benchmark_net_total_return"),
    ("index_gross_total_return", "benchmark_gross_total_return"),
    ("index_price_return", "benchmark_price_return"),
)
_COMPATIBILITY_RETURN_COLUMN_PAIRS = (("index_return", "benchmark_return"),)
_TURNOVER_COLUMNS = (
    "one_way_turnover", "formal_one_way_turnover", "index_one_way_turnover",
    "index_turnover", "formal_turnover", "turnover",
)

@dataclass(frozen=True, slots=True)
class AnalyticsEngine:
    """Calculate generic analytics without loading data or changing results."""

    annualization_factor: int = 252
    weight_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if self.annualization_factor <= 0:
            raise AnalyticsValidationError("annualization_factor must be positive")
        if self.weight_tolerance <= 0:
            raise AnalyticsValidationError("weight_tolerance must be positive")

    def analyze(
        self,
        backtest_result: object,
        simulation_result: object | None = None,
        *,
        daily_returns: pd.DataFrame | None = None,
        one_way_turnover: pd.DataFrame | pd.Series | None = None,
        formal_turnover: pd.DataFrame | pd.Series | None = None,
        return_columns: tuple[str, str] | None = None,
        brinson_input: BrinsonInput | None = None,
    ) -> AnalyticsResult:
        """Analyze an already-completed backtest and optional daily simulation.

        ``backtest_result`` is consumed through its ``weights`` and ``reviews``
        attributes. ``simulation_result`` is consumed through its ``daily`` and
        ``rebalances`` attributes. This keeps analytics independent of the
        calculation classes while retaining strict tabular contracts.
        """

        diagnostics: list[AnalyticsDiagnostic] = []
        review_frame = self._validated_review_frame(backtest_result)
        review_validation, review_metrics = self._review_statistics(review_frame)
        country_exposures = self._exposures(
            review_frame, "country", diagnostics
        )
        industry_exposures = self._exposures(
            review_frame, "industry", diagnostics
        )
        target_weight_change = self._target_weight_change(
            review_frame, diagnostics
        )
        if one_way_turnover is not None and formal_turnover is not None:
            raise AnalyticsValidationError(
                "supply one_way_turnover or formal_turnover, not both"
            )
        supplied_turnover = (
            one_way_turnover
            if one_way_turnover is not None
            else formal_turnover
        )
        formal_turnover_frame = self._formal_turnover(
            supplied_turnover,
            simulation_result,
            diagnostics,
            supplied_name=(
                "explicit one_way_turnover"
                if one_way_turnover is not None
                else "explicit formal_turnover"
            ),
        )
        performance, drawdowns = self._performance(
            daily_returns,
            simulation_result,
            return_columns,
            diagnostics,
        )

        brinson = None
        if brinson_input is not None:
            brinson = calculate_brinson_attribution(
                brinson_input,
                weight_tolerance=self.weight_tolerance,
            )
            diagnostics.append(
                AnalyticsDiagnostic(
                    level="info",
                    code="brinson_calculated",
                    message="Brinson attribution used explicitly supplied, pre-aligned inputs.",
                )
            )

        return AnalyticsResult(
            review_validation=review_validation,
            review_metrics=review_metrics,
            country_exposures=country_exposures,
            industry_exposures=industry_exposures,
            target_review_weight_change=target_weight_change,
            formal_turnover=formal_turnover_frame,
            performance=performance,
            drawdowns=drawdowns,
            brinson=brinson,
            diagnostics=tuple(diagnostics),
        )

    def _validated_review_frame(self, result: object) -> pd.DataFrame:
        reviews = getattr(result, "reviews", None)
        result_weights = getattr(result, "weights", None)
        if not isinstance(reviews, Mapping):
            raise AnalyticsValidationError(
                "backtest_result.reviews must be a mapping by effective_date"
            )
        if not isinstance(result_weights, pd.DataFrame):
            raise AnalyticsValidationError(
                "backtest_result.weights must be a pandas DataFrame"
            )
        if not reviews:
            raise AnalyticsValidationError(
                "backtest_result.reviews must contain at least one review"
            )

        dated_contexts: list[tuple[pd.Timestamp, object]] = []
        seen_dates: set[pd.Timestamp] = set()
        for effective_date, context in reviews.items():
            date = _normalise_date(effective_date, "review effective_date")
            if date in seen_dates:
                raise AnalyticsValidationError(
                    f"multiple reviews resolve to effective_date {date.date()}"
                )
            seen_dates.add(date)
            dated_contexts.append((date, context))
        dated_contexts.sort(key=lambda item: item[0])

        review_frames: list[pd.DataFrame] = []
        for effective_date, context in dated_contexts:
            context_date = getattr(context, "effective_date", None)
            if context_date is not None and (
                _normalise_date(context_date, "context effective_date")
                != effective_date
            ):
                raise AnalyticsValidationError(
                    "review mapping date does not match context.effective_date"
                )

            constituents = getattr(context, "cons", None)
            if not isinstance(constituents, pd.DataFrame):
                raise AnalyticsValidationError(
                    "every review context must expose a pandas DataFrame as cons"
                )
            frame = constituents.copy(deep=True)
            if frame.empty:
                raise AnalyticsValidationError(
                    f"review {effective_date.date()} contains no instruments"
                )
            frame = _use_instrument_index(frame, "review constituent data")

            if "effective_date" in frame.columns:
                constituent_dates = pd.to_datetime(
                    frame["effective_date"], errors="coerce"
                )
                if constituent_dates.isna().any():
                    raise AnalyticsValidationError(
                        "constituent effective_date contains missing or invalid values"
                    )
                constituent_dates = constituent_dates.map(
                    lambda value: pd.Timestamp(value).normalize()
                )
                if not constituent_dates.eq(effective_date).all():
                    raise AnalyticsValidationError(
                        "constituent effective_date does not match its review"
                    )
                frame = frame.drop(columns="effective_date")

            required = {"index_weight", "benchmark_weight"}
            missing = sorted(required.difference(frame.columns))
            if missing:
                raise AnalyticsValidationError(
                    f"review {effective_date.date()} is missing columns: {missing}"
                )
            for column in ("index_weight", "benchmark_weight"):
                frame[column] = _validated_weight_series(
                    frame[column],
                    f"{column} for review {effective_date.date()}",
                )

            frame.insert(0, "effective_date", effective_date)
            frame = frame.reset_index().set_index(
                ["effective_date", "instrument_id"], verify_integrity=True
            )
            review_frames.append(frame)

        combined = pd.concat(review_frames, axis=0)
        supplied_weights = _normalise_result_weights(result_weights)
        comparison = pd.concat(
            [
                combined["index_weight"].rename("context_index_weight"),
                supplied_weights.rename("result_index_weight"),
            ],
            axis=1,
        )
        if comparison.isna().any().any():
            raise AnalyticsValidationError(
                "backtest_result.weights and review contexts contain different rows"
            )
        if not np.allclose(
            comparison["context_index_weight"].to_numpy(dtype=float),
            comparison["result_index_weight"].to_numpy(dtype=float),
            atol=self.weight_tolerance,
            rtol=0.0,
        ):
            raise AnalyticsValidationError(
                "backtest_result.weights does not match review context index_weight"
            )

        for effective_date, group in combined.groupby(
            level="effective_date", sort=True
        ):
            for column in ("index_weight", "benchmark_weight"):
                total = float(group[column].sum())
                if not np.isclose(
                    total, 1.0, atol=self.weight_tolerance, rtol=0.0
                ):
                    raise AnalyticsValidationError(
                        f"{column} sums to {total:.12g} for "
                        f"{pd.Timestamp(effective_date).date()}, not one"
                    )
        return combined

    def _review_statistics(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        validations: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        for effective_date, group in frame.groupby(
            level="effective_date", sort=True
        ):
            portfolio = group["index_weight"]
            benchmark = group["benchmark_weight"]
            portfolio_count = int((portfolio > self.weight_tolerance).sum())
            benchmark_count = int((benchmark > self.weight_tolerance).sum())
            hhi = float(portfolio.pow(2).sum())

            validations.append(
                {
                    "effective_date": effective_date,
                    "instrument_count": int(len(group)),
                    "portfolio_constituent_count": portfolio_count,
                    "benchmark_constituent_count": benchmark_count,
                    "portfolio_weight_sum": float(portfolio.sum()),
                    "benchmark_weight_sum": float(benchmark.sum()),
                    "minimum_portfolio_weight": float(portfolio.min()),
                    "maximum_portfolio_weight": float(portfolio.max()),
                    "is_valid": True,
                }
            )
            metrics.append(
                {
                    "effective_date": effective_date,
                    "constituent_count": portfolio_count,
                    "max_weight": float(portfolio.max()),
                    "top_10_weight": float(portfolio.nlargest(10).sum()),
                    "hhi": hhi,
                    "effective_n": float(1.0 / hhi) if hhi > 0 else np.nan,
                    "active_share": float(
                        0.5 * (portfolio - benchmark).abs().sum()
                    ),
                }
            )

        validation_frame = pd.DataFrame(validations).set_index("effective_date")
        metric_frame = pd.DataFrame(metrics).set_index("effective_date")
        validation_frame.index.name = "effective_date"
        metric_frame.index.name = "effective_date"
        return validation_frame, metric_frame

    def _exposures(
        self,
        frame: pd.DataFrame,
        classification: str,
        diagnostics: list[AnalyticsDiagnostic],
    ) -> pd.DataFrame:
        working = frame[
            ["index_weight", "benchmark_weight"]
        ].copy(deep=True)
        if classification not in frame.columns:
            values = pd.Series(
                _UNCLASSIFIED, index=frame.index, dtype="object"
            )
            missing_count = len(frame)
        else:
            values = frame[classification].astype("object").copy()
            missing = values.isna() | values.astype(str).str.strip().eq("")
            missing_count = int(missing.sum())
            values.loc[missing] = _UNCLASSIFIED
        working[classification] = values

        if missing_count:
            diagnostics.append(
                AnalyticsDiagnostic(
                    level="warning",
                    code=f"unclassified_{classification}",
                    message=(
                        f"{missing_count} review rows had no {classification}; "
                        f"they were grouped as {_UNCLASSIFIED!r}."
                    ),
                )
            )

        grouped = (
            working.reset_index()
            .groupby(
                ["effective_date", classification],
                dropna=False,
                sort=True,
            )[["index_weight", "benchmark_weight"]]
            .sum()
            .rename(
                columns={
                    "index_weight": "portfolio_weight",
                    "benchmark_weight": "benchmark_weight",
                }
            )
        )
        grouped["active_weight"] = (
            grouped["portfolio_weight"] - grouped["benchmark_weight"]
        )
        return grouped

    def _target_weight_change(
        self,
        frame: pd.DataFrame,
        diagnostics: list[AnalyticsDiagnostic],
    ) -> pd.DataFrame:
        dates = list(
            frame.index.get_level_values("effective_date").unique().sort_values()
        )
        columns = [
            "previous_effective_date",
            "gross_target_weight_change",
            "one_way_target_weight_change",
        ]
        if len(dates) < 2:
            diagnostics.append(
                AnalyticsDiagnostic(
                    level="info",
                    code="no_adjacent_target_reviews",
                    message=(
                        "At least two reviews are required to calculate target "
                        "review weight change."
                    ),
                )
            )
            return pd.DataFrame(columns=columns).rename_axis("effective_date")

        rows: list[dict[str, Any]] = []
        for previous_date, effective_date in zip(dates, dates[1:]):
            previous = frame.xs(
                previous_date, level="effective_date"
            )["index_weight"]
            current = frame.xs(
                effective_date, level="effective_date"
            )["index_weight"]
            previous, current = previous.align(
                current, join="outer", fill_value=0.0
            )
            gross_change = float((current - previous).abs().sum())
            rows.append(
                {
                    "effective_date": effective_date,
                    "previous_effective_date": previous_date,
                    "gross_target_weight_change": gross_change,
                    "one_way_target_weight_change": 0.5 * gross_change,
                }
            )
        return pd.DataFrame(rows).set_index("effective_date")[columns]

    def _formal_turnover(
        self,
        supplied: pd.DataFrame | pd.Series | None,
        simulation_result: object | None,
        diagnostics: list[AnalyticsDiagnostic],
        *,
        supplied_name: str,
    ) -> pd.DataFrame:
        source_name = supplied_name
        source: object | None = supplied
        if source is None and simulation_result is not None:
            # Keep the v1 diagnostic source stable while the rebalance frame
            # exposes canonical turnover aliases.
            source = getattr(simulation_result, "rebalances", None)
            source_name = "simulation_result.rebalances"

        if source is None:
            diagnostics.append(
                AnalyticsDiagnostic(
                    level="warning",
                    code="formal_turnover_unavailable",
                    message=(
                        "Formal simulation turnover is unavailable. Target review "
                        "weight change remains available as a separate measure."
                    ),
                )
            )
            return pd.DataFrame(
                columns=["one_way_turnover", "formal_one_way_turnover"]
            )
        if isinstance(source, pd.Series):
            source = source.to_frame()
        if not isinstance(source, pd.DataFrame):
            raise AnalyticsValidationError(
                f"{source_name} must be a pandas DataFrame or Series"
            )
        frame = source.copy(deep=True)
        if frame.empty:
            diagnostics.append(
                AnalyticsDiagnostic(
                    level="warning",
                    code="formal_turnover_unavailable",
                    message=f"{source_name} contains no rebalance rows.",
                )
            )
            if "one_way_turnover" not in frame.columns:
                frame["one_way_turnover"] = pd.Series(dtype=float)
            if "formal_one_way_turnover" not in frame.columns:
                frame["formal_one_way_turnover"] = frame[
                    "one_way_turnover"
                ]
            return frame

        casefolded = {
            str(column).casefold(): column for column in frame.columns
        }
        selected = next(
            (
                casefolded[candidate.casefold()]
                for candidate in _TURNOVER_COLUMNS
                if candidate.casefold() in casefolded
            ),
            None,
        )
        if selected is None:
            raise AnalyticsValidationError(
                f"{source_name} must contain a one-way turnover column"
            )
        if selected != "one_way_turnover":
            frame["one_way_turnover"] = frame[selected]

        original_turnover = frame["one_way_turnover"]
        turnover = pd.to_numeric(original_turnover, errors="coerce")
        if (original_turnover.notna() & turnover.isna()).any():
            raise AnalyticsValidationError(
                f"{source_name} contains non-numeric turnover"
            )
        finite_turnover = turnover.dropna().to_numpy(dtype=float)
        if not np.isfinite(finite_turnover).all():
            raise AnalyticsValidationError(
                f"{source_name} contains non-finite turnover"
            )
        if (turnover.dropna() < -self.weight_tolerance).any():
            raise AnalyticsValidationError(
                f"{source_name} contains negative turnover"
            )
        frame["one_way_turnover"] = turnover.clip(lower=0.0)
        frame["formal_one_way_turnover"] = frame["one_way_turnover"]

        for column in (
            "effective_date",
            "scheduled_effective_date",
            "applied_date",
            "applied_business_date",
            "business_date",
        ):
            if column in frame.columns:
                converted = pd.to_datetime(frame[column], errors="coerce")
                if converted.isna().any():
                    raise AnalyticsValidationError(
                        f"{source_name}.{column} contains invalid dates"
                    )
                frame[column] = converted.map(
                    lambda value: pd.Timestamp(value).normalize()
                )

        diagnostics.append(
            AnalyticsDiagnostic(
                level="info",
                code="formal_turnover_loaded",
                message=f"Formal one-way turnover was read from {source_name}.",
            )
        )
        return frame

    def _performance(
        self,
        supplied: pd.DataFrame | None,
        simulation_result: object | None,
        return_columns: tuple[str, str] | None,
        diagnostics: list[AnalyticsDiagnostic],
    ) -> tuple[pd.Series, pd.DataFrame]:
        source_name = "explicit daily_returns"
        source: object | None = supplied
        if source is None and simulation_result is not None:
            for attribute in (
                "daily",
                "daily_returns",
                "index_returns",
                "index_series",
            ):
                candidate = getattr(simulation_result, attribute, None)
                if candidate is not None:
                    source = candidate
                    source_name = f"simulation_result.{attribute}"
                    break

        if source is None:
            diagnostics.append(
                AnalyticsDiagnostic(
                    level="warning",
                    code="daily_returns_unavailable",
                    message="Daily index and benchmark returns are unavailable.",
                )
            )
            return _empty_performance()
        if not isinstance(source, pd.DataFrame):
            raise AnalyticsValidationError(
                f"{source_name} must be a pandas DataFrame"
            )
        daily = source.copy(deep=True)
        if daily.empty:
            diagnostics.append(
                AnalyticsDiagnostic(
                    level="warning",
                    code="daily_returns_unavailable",
                    message=f"{source_name} contains no daily rows.",
                )
            )
            return _empty_performance()

        daily = _use_business_date_index(daily, source_name)
        if return_columns is not None:
            if (
                not isinstance(return_columns, tuple)
                or len(return_columns) != 2
                or any(
                    not isinstance(column, str) or not column
                    for column in return_columns
                )
            ):
                raise AnalyticsValidationError(
                    "return_columns must be a two-item tuple of column names"
                )
            selected_pair = return_columns
            missing = [
                column for column in selected_pair if column not in daily.columns
            ]
            if missing:
                raise AnalyticsValidationError(
                    f"{source_name} is missing selected return columns: {missing}"
                )
        else:
            selected_pair = next(
                (
                    pair
                    for pair in (
                        _CANONICAL_RETURN_COLUMN_PAIRS
                        + _COMPATIBILITY_RETURN_COLUMN_PAIRS
                    )
                    if pair[0] in daily.columns
                    and pair[1] in daily.columns
                ),
                None,
            )
            if selected_pair is None:
                diagnostics.append(
                    AnalyticsDiagnostic(
                        level="warning",
                        code="paired_returns_unavailable",
                        message=(
                            f"{source_name} has no supported index and benchmark "
                            "return-column pair."
                        ),
                    )
                )
                return _empty_performance()

        index_column, benchmark_column = selected_pair
        selected = daily[[index_column, benchmark_column]].copy()
        for column in selected.columns:
            original = selected[column]
            converted = pd.to_numeric(original, errors="coerce")
            if (original.notna() & converted.isna()).any():
                raise AnalyticsValidationError(
                    f"{source_name}.{column} contains non-numeric values"
                )
            finite = converted.dropna().to_numpy(dtype=float)
            if not np.isfinite(finite).all():
                raise AnalyticsValidationError(
                    f"{source_name}.{column} contains non-finite values"
                )
            selected[column] = converted

        complete = selected.dropna(how="any")
        dropped = len(selected) - len(complete)
        if dropped:
            diagnostics.append(
                AnalyticsDiagnostic(
                    level="warning",
                    code="incomplete_daily_returns_dropped",
                    message=(
                        f"{dropped} daily rows without a complete index/benchmark "
                        "return pair were excluded."
                    ),
                )
            )
        if complete.empty:
            diagnostics.append(
                AnalyticsDiagnostic(
                    level="warning",
                    code="paired_returns_unavailable",
                    message=f"{source_name} contains no complete return pairs.",
                )
            )
            return _empty_performance()
        if (complete < -1.0 - self.weight_tolerance).any().any():
            raise AnalyticsValidationError("daily returns cannot be less than -100%")
        complete = complete.clip(lower=-1.0)

        index_returns = complete[index_column].to_numpy(dtype=float)
        benchmark_returns = complete[benchmark_column].to_numpy(dtype=float)
        active_returns = index_returns - benchmark_returns
        observations = len(complete)
        index_total = float(np.prod(1.0 + index_returns) - 1.0)
        benchmark_total = float(np.prod(1.0 + benchmark_returns) - 1.0)
        index_annualized = _annualized_return(
            index_total, observations, self.annualization_factor
        )
        benchmark_annualized = _annualized_return(
            benchmark_total, observations, self.annualization_factor
        )
        index_volatility = _annualized_volatility(
            index_returns, self.annualization_factor
        )
        benchmark_volatility = _annualized_volatility(
            benchmark_returns, self.annualization_factor
        )
        tracking_error = _annualized_volatility(
            active_returns, self.annualization_factor
        )
        annualized_excess = index_annualized - benchmark_annualized
        information_ratio = (
            annualized_excess / tracking_error
            if np.isfinite(tracking_error) and tracking_error > 0
            else np.nan
        )

        index_levels, index_drawdowns = _level_and_drawdown(index_returns)
        benchmark_levels, benchmark_drawdowns = _level_and_drawdown(
            benchmark_returns
        )
        drawdowns = pd.DataFrame(
            {
                "index_level": index_levels,
                "benchmark_level": benchmark_levels,
                "index_drawdown": index_drawdowns,
                "benchmark_drawdown": benchmark_drawdowns,
                "active_return": active_returns,
            },
            index=complete.index.copy(),
        )
        drawdowns.index.name = "business_date"
        performance = pd.Series(
            {
                "observations": float(observations),
                "annualization_factor": float(self.annualization_factor),
                "total_return": index_total,
                "benchmark_total_return": benchmark_total,
                "annualized_return": index_annualized,
                "benchmark_annualized_return": benchmark_annualized,
                "annualized_volatility": index_volatility,
                "benchmark_annualized_volatility": benchmark_volatility,
                "annualized_excess_return": annualized_excess,
                "tracking_error": tracking_error,
                "information_ratio": information_ratio,
                "maximum_drawdown": float(index_drawdowns.min()),
                "benchmark_maximum_drawdown": float(
                    benchmark_drawdowns.min()
                ),
            },
            dtype=float,
            name="value",
        )
        diagnostics.append(
            AnalyticsDiagnostic(
                level="info",
                code="return_columns_selected",
                message=(
                    f"Performance used {index_column!r} and "
                    f"{benchmark_column!r} from {source_name}."
                ),
            )
        )
        return performance, drawdowns

def analyze_backtest(
    backtest_result: object,
    simulation_result: object | None = None,
    *,
    daily_returns: pd.DataFrame | None = None,
    one_way_turnover: pd.DataFrame | pd.Series | None = None,
    formal_turnover: pd.DataFrame | pd.Series | None = None,
    return_columns: tuple[str, str] | None = None,
    brinson_input: BrinsonInput | None = None,
    annualization_factor: int = 252,
    weight_tolerance: float = 1e-8,
) -> AnalyticsResult:
    """Convenience wrapper around :class:`AnalyticsEngine`."""

    return AnalyticsEngine(
        annualization_factor=annualization_factor,
        weight_tolerance=weight_tolerance,
    ).analyze(
        backtest_result,
        simulation_result,
        daily_returns=daily_returns,
        one_way_turnover=one_way_turnover,
        formal_turnover=formal_turnover,
        return_columns=return_columns,
        brinson_input=brinson_input,
    )

__all__ = ["AnalyticsEngine", "analyze_backtest"]
