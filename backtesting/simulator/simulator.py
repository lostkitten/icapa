"""Stateful daily simulation for review-based index research."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any

import numpy as np
import pandas as pd

from icapa.backtesting.backtester import BacktestResult
from icapa.backtesting.simulation_params import (
    DividendTreatment,
    RebalanceTiming,
    SimulationParams,
    WeightDrift,
)
from icapa.data_sources.contracts import validate_daily_market_data
from icapa.data_sources.registry import registry
from icapa.workspace import CacheMissError, CachePolicy


@dataclass(frozen=True)
class IndexSimulationResult:
    """Daily index series, holdings, rebalances, and underlying observations."""

    daily: pd.DataFrame
    holdings: pd.DataFrame
    rebalances: pd.DataFrame
    asset_returns: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def formal_turnover(self) -> pd.DataFrame:
        """Return formal pre-trade-to-target one-way turnover."""

        columns = [
            column
            for column in ("index_turnover", "benchmark_turnover")
            if column in self.rebalances
        ]
        return self.rebalances.loc[:, columns].copy()


@dataclass
class IndexSimulator:
    """Simulate daily index levels from cached or freshly generated review weights."""

    backtest_result: BacktestResult
    market_data_provider_name: str
    start_date: object
    end_date: object
    provider_parameters: dict[str, Any] = field(default_factory=dict)
    params: SimulationParams = field(default_factory=SimulationParams)
    data_revision: str | None = None
    workspace: object | None = None
    cache_policy: CachePolicy | str = CachePolicy.REUSE

    def __post_init__(self) -> None:
        if not self.market_data_provider_name:
            raise ValueError("market_data_provider_name is required")
        self.start_date = pd.Timestamp(self.start_date).normalize()
        self.end_date = pd.Timestamp(self.end_date).normalize()
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        if not self.backtest_result.reviews:
            raise ValueError("backtest_result must contain at least one review")
        self.cache_policy = CachePolicy(self.cache_policy)
        if self.cache_policy is CachePolicy.READ_ONLY and self.workspace is None:
            raise ValueError("READ_ONLY simulation cache requires a workspace")

    def run(self) -> IndexSimulationResult:
        """Load market data and run a stateful daily index simulation."""

        cache_key = self._cache_key()
        cached = self._load_cached(cache_key)
        if cached is not None:
            return cached

        instrument_ids = sorted(
            {
                instrument_id
                for context in self.backtest_result.reviews.values()
                for instrument_id in context.cons.index
            },
            key=str,
        )
        provider = registry.resolve(
            "load_daily_market_data",
            self.market_data_provider_name,
        )
        raw = provider.load_daily_market_data(
            instrument_ids=instrument_ids,
            start_date=self.start_date,
            end_date=self.end_date,
            **self.provider_parameters,
        )
        market_data = validate_daily_market_data(raw)
        market_data = market_data.loc[
            market_data["business_date"].between(self.start_date, self.end_date)
        ].copy()
        if market_data.empty:
            raise ValueError("market data contains no observations in the requested range")
        result = self._simulate(market_data, cache_key)
        self._save_cached(cache_key, result)
        return result

    def _simulate(
        self,
        market_data: pd.DataFrame,
        cache_key: str,
    ) -> IndexSimulationResult:
        required = {
            "instrument_id",
            "business_date",
            "price_return",
            "gross_dividend",
            "net_dividend",
            "market_cap",
        }
        missing = sorted(required.difference(market_data.columns))
        if missing:
            raise ValueError(f"daily market data is missing columns: {missing}")
        numeric = ["price_return", "gross_dividend", "net_dividend", "market_cap"]
        for column in numeric:
            market_data[column] = pd.to_numeric(
                market_data[column],
                errors="raise",
            )
            if not np.isfinite(market_data[column]).all():
                raise ValueError(f"daily market data contains non-finite {column}")
        if (market_data["market_cap"] < 0).any():
            raise ValueError("daily market data contains negative market_cap")

        observations_by_date = {
            business_date: frame.set_index("instrument_id", verify_integrity=True)
            for business_date, frame in market_data.groupby(
                "business_date",
                sort=True,
            )
        }
        business_dates = pd.DatetimeIndex(sorted(observations_by_date))
        applications = self._review_applications(business_dates)
        first_date = business_dates[0]
        active_applications = [
            applied for applied in applications if applied["applied_business_date"] <= first_date
        ]
        if not active_applications:
            first_effective = min(
                item["scheduled_effective_date"] for item in applications
            )
            raise ValueError(
                "simulation starts before the first available review becomes effective; "
                f"first effective date is {first_effective.date()}"
            )

        current_index: pd.Series | None = None
        current_benchmark: pd.Series | None = None
        daily_rows: list[dict[str, Any]] = []
        holding_rows: list[dict[str, Any]] = []
        rebalance_rows: list[dict[str, Any]] = []
        levels = {
            "index_price_level": float(self.params.base_value),
            "index_gross_total_level": float(self.params.base_value),
            "index_net_total_level": float(self.params.base_value),
            "benchmark_price_level": float(self.params.base_value),
            "benchmark_gross_total_level": float(self.params.base_value),
            "benchmark_net_total_level": float(self.params.base_value),
        }
        applications_by_date: dict[pd.Timestamp, dict[str, Any]] = {
            item["applied_business_date"]: item for item in applications
        }

        for business_date in business_dates:
            observation = observations_by_date[business_date]
            application = applications_by_date.get(business_date)
            if application is not None:
                context = self.backtest_result.reviews[
                    application["scheduled_effective_date"]
                ]
                target_index = _normalised_weights(
                    context.cons["index_weight"],
                    "index_weight",
                )
                target_benchmark = _normalised_weights(
                    context.cons["benchmark_weight"],
                    "benchmark_weight",
                )
                index_turnover = (
                    np.nan
                    if current_index is None
                    else _one_way_turnover(current_index, target_index)
                )
                benchmark_turnover = (
                    np.nan
                    if current_benchmark is None
                    else _one_way_turnover(current_benchmark, target_benchmark)
                )
                current_index = target_index
                current_benchmark = target_benchmark
                source = self._review_source(
                    application["scheduled_effective_date"]
                )
                rebalance_rows.append(
                    {
                        "scheduled_effective_date": application[
                            "scheduled_effective_date"
                        ],
                        "applied_business_date": business_date,
                        "reference_date": application["reference_date"],
                        "index_turnover": index_turnover,
                        "benchmark_turnover": benchmark_turnover,
                        "instrument_count": int(len(target_index)),
                        "review_source": source,
                    }
                )
            if current_index is None or current_benchmark is None:
                continue

            index_open = current_index.copy()
            benchmark_open = current_benchmark.copy()
            index_factors = _portfolio_factors(
                observation,
                index_open,
                self.params.dividend_treatment,
                business_date,
            )
            benchmark_factors = _portfolio_factors(
                observation,
                benchmark_open,
                self.params.dividend_treatment,
                business_date,
            )
            current_index = self._drift(index_open, observation, business_date)
            current_benchmark = self._drift(
                benchmark_open,
                observation,
                business_date,
            )
            returns = {
                "index_price_return": index_factors["price"] - 1.0,
                "index_gross_total_return": index_factors["gross"] - 1.0,
                "index_net_total_return": index_factors["net"] - 1.0,
                "benchmark_price_return": benchmark_factors["price"] - 1.0,
                "benchmark_gross_total_return": benchmark_factors["gross"] - 1.0,
                "benchmark_net_total_return": benchmark_factors["net"] - 1.0,
            }
            for prefix in ("index", "benchmark"):
                for return_name, level_name in (
                    ("price_return", "price_level"),
                    ("gross_total_return", "gross_total_level"),
                    ("net_total_return", "net_total_level"),
                ):
                    level_column = f"{prefix}_{level_name}"
                    levels[level_column] *= 1.0 + returns[
                        f"{prefix}_{return_name}"
                    ]
            daily_rows.append(
                {
                    "business_date": business_date,
                    **returns,
                    **levels,
                    "active_price_return": returns["index_price_return"]
                    - returns["benchmark_price_return"],
                    "active_gross_total_return": returns[
                        "index_gross_total_return"
                    ]
                    - returns["benchmark_gross_total_return"],
                    "active_net_total_return": returns["index_net_total_return"]
                    - returns["benchmark_net_total_return"],
                }
            )

            all_ids = index_open.index.union(benchmark_open.index)
            for instrument_id in all_ids:
                holding_rows.append(
                    {
                        "business_date": business_date,
                        "instrument_id": instrument_id,
                        "index_opening_weight": float(index_open.get(instrument_id, 0.0)),
                        "index_closing_weight": float(
                            current_index.get(instrument_id, 0.0)
                        ),
                        "benchmark_opening_weight": float(
                            benchmark_open.get(instrument_id, 0.0)
                        ),
                        "benchmark_closing_weight": float(
                            current_benchmark.get(instrument_id, 0.0)
                        ),
                    }
                )

        if not daily_rows:
            raise ValueError("simulation produced no daily index observations")
        daily = pd.DataFrame.from_records(daily_rows).set_index("business_date")
        holdings = pd.DataFrame.from_records(holding_rows).set_index(
            ["business_date", "instrument_id"]
        )
        rebalances = pd.DataFrame.from_records(rebalance_rows)
        assets = market_data.set_index(["business_date", "instrument_id"]).sort_index()
        return IndexSimulationResult(
            daily=daily,
            holdings=holdings,
            rebalances=rebalances,
            asset_returns=assets,
            metadata={
                "cache_key": cache_key,
                "cache_source": "computed",
                "market_data_provider_name": self.market_data_provider_name,
                "data_revision": self.data_revision,
                "start_date": str(self.start_date.date()),
                "end_date": str(self.end_date.date()),
                "simulation_params": {
                    key: value.value if hasattr(value, "value") else value
                    for key, value in asdict(self.params).items()
                },
            },
        )

    def _review_applications(
        self,
        business_dates: pd.DatetimeIndex,
    ) -> list[dict[str, Any]]:
        review_items = sorted(
            (
                pd.Timestamp(effective_date).normalize(),
                context,
            )
            for effective_date, context in self.backtest_result.reviews.items()
        )
        first_business_date = business_dates[0]
        latest_prior = [
            item for item in review_items if item[0] <= first_business_date
        ][-1:]
        future = [
            item
            for item in review_items
            if first_business_date < item[0] <= business_dates[-1]
        ]
        applications: list[dict[str, Any]] = []
        for scheduled, context in [*latest_prior, *future]:
            if self.params.rebalance_timing is RebalanceTiming.EXACT_DATE:
                if scheduled not in business_dates:
                    raise ValueError(
                        f"effective date is not a market-data business day: {scheduled.date()}"
                    )
                applied = scheduled
            else:
                position = int(business_dates.searchsorted(scheduled, side="left"))
                if position >= len(business_dates):
                    continue
                applied = business_dates[position]
            applications.append(
                {
                    "scheduled_effective_date": scheduled,
                    "applied_business_date": applied,
                    "reference_date": pd.Timestamp(context.reference_date).normalize(),
                }
            )
        applied_dates = [item["applied_business_date"] for item in applications]
        if len(applied_dates) != len(set(applied_dates)):
            raise ValueError("multiple reviews resolve to the same applied business date")
        return applications

    def _drift(
        self,
        opening_weights: pd.Series,
        observations: pd.DataFrame,
        business_date: pd.Timestamp,
    ) -> pd.Series:
        selected = _select_observations(
            observations,
            opening_weights.index,
            business_date,
        )
        if self.params.weight_drift is WeightDrift.PRICE_RETURN:
            values = opening_weights * (1.0 + selected["price_return"])
        else:
            values = selected["market_cap"].where(opening_weights > 0, 0.0)
        if not np.isfinite(values).all() or float(values.sum()) <= 0:
            raise ValueError(
                f"weight drift is invalid on {business_date.date()}"
            )
        result = values / float(values.sum())
        result.index.name = "instrument_id"
        return result

    def _cache_key(self) -> str:
        result_metadata = getattr(self.backtest_result, "metadata", None)
        fingerprint = getattr(result_metadata, "fingerprint", None)
        if fingerprint:
            review_metadata = getattr(result_metadata, "reviews", {})
            weight_identity: Any = {
                "backtest_fingerprint": fingerprint,
                "review_artifacts": [
                    {
                        "effective_date": str(
                            pd.Timestamp(effective_date).date()
                        ),
                        "checksum": getattr(review, "artifact_checksum", None),
                    }
                    for effective_date, review in sorted(review_metadata.items())
                ],
            }
        else:
            weight_frame = self.backtest_result.weights.reset_index().copy()
            weight_frame["effective_date"] = pd.to_datetime(
                weight_frame["effective_date"]
            ).dt.strftime("%Y-%m-%d")
            weight_frame = weight_frame.sort_values(
                ["effective_date", "instrument_id"],
                key=lambda values: values.map(str),
            )
            weight_identity = weight_frame.to_dict(orient="records")
        payload = {
            "schema": 1,
            "weights": weight_identity,
            "provider": self.market_data_provider_name,
            "provider_parameters": self.provider_parameters,
            "data_revision": self.data_revision,
            "start_date": str(self.start_date.date()),
            "end_date": str(self.end_date.date()),
            "params": {
                key: value.value if hasattr(value, "value") else value
                for key, value in asdict(self.params).items()
            },
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def _load_cached(self, cache_key: str) -> IndexSimulationResult | None:
        if (
            self.workspace is None
            or self.data_revision is None
            or self.cache_policy is CachePolicy.REFRESH
        ):
            return None
        load_frame = getattr(self.workspace, "load_frame", None)
        load_json = getattr(self.workspace, "load_json", None)
        if not callable(load_frame) or not callable(load_json):
            return None
        try:
            daily = load_frame("simulation", cache_key, "daily")
            holdings = load_frame("simulation", cache_key, "holdings")
            rebalances = load_frame("simulation", cache_key, "rebalances")
            asset_returns = load_frame("simulation", cache_key, "asset_returns")
            metadata = load_json("simulation", cache_key, "metadata")
        except CacheMissError:
            if self.cache_policy is CachePolicy.READ_ONLY:
                raise
            return None
        return IndexSimulationResult(
            daily=daily,
            holdings=holdings,
            rebalances=rebalances,
            asset_returns=asset_returns,
            metadata={**metadata, "cache_source": "workspace"},
        )

    def _save_cached(
        self,
        cache_key: str,
        result: IndexSimulationResult,
    ) -> None:
        if (
            self.workspace is None
            or self.data_revision is None
            or self.cache_policy is CachePolicy.READ_ONLY
        ):
            return
        save_frame = getattr(self.workspace, "save_frame", None)
        save_json = getattr(self.workspace, "save_json", None)
        if not callable(save_frame) or not callable(save_json):
            return
        save_frame("simulation", cache_key, "daily", result.daily)
        save_frame("simulation", cache_key, "holdings", result.holdings)
        save_frame("simulation", cache_key, "rebalances", result.rebalances)
        save_frame("simulation", cache_key, "asset_returns", result.asset_returns)
        save_json(
            "simulation",
            cache_key,
            "metadata",
            {**result.metadata, "cache_source": "computed"},
        )

    def _review_source(self, effective_date: pd.Timestamp) -> str:
        metadata = getattr(self.backtest_result, "metadata", None)
        reviews = getattr(metadata, "reviews", {}) if metadata is not None else {}
        review = reviews.get(effective_date)
        source = getattr(review, "cache_source", None)
        return getattr(source, "value", source) or "computed"


def _normalised_weights(weights: pd.Series, label: str) -> pd.Series:
    result = pd.to_numeric(weights, errors="raise").astype(float)
    if result.index.has_duplicates:
        raise ValueError(f"{label} contains duplicate instrument_id values")
    if not np.isfinite(result).all() or (result < 0).any() or float(result.sum()) <= 0:
        raise ValueError(f"{label} must be finite, non-negative, and non-zero")
    result = result / float(result.sum())
    result.index.name = "instrument_id"
    return result


def _select_observations(
    observations: pd.DataFrame,
    instrument_ids: pd.Index,
    business_date: pd.Timestamp,
) -> pd.DataFrame:
    missing_ids = instrument_ids.difference(observations.index)
    if len(missing_ids):
        raise ValueError(
            f"daily market data is missing held instruments on {business_date.date()}: "
            f"{list(missing_ids)}"
        )
    return observations.loc[instrument_ids]


def _portfolio_factors(
    observations: pd.DataFrame,
    weights: pd.Series,
    treatment: DividendTreatment,
    business_date: pd.Timestamp,
) -> dict[str, float]:
    selected = _select_observations(observations, weights.index, business_date)
    price_factor = float((weights * (1.0 + selected["price_return"])).sum())
    gross_dividend = float((weights * selected["gross_dividend"]).sum())
    net_dividend = float((weights * selected["net_dividend"]).sum())
    if treatment is DividendTreatment.NYSE:
        if gross_dividend >= 1.0 or net_dividend >= 1.0:
            raise ValueError("weighted dividend must be less than one")
        gross_factor = price_factor / (1.0 - gross_dividend)
        net_factor = price_factor / (1.0 - net_dividend)
    else:
        gross_factor = price_factor + gross_dividend
        net_factor = price_factor + net_dividend
    factors = {
        "price": price_factor,
        "gross": gross_factor,
        "net": net_factor,
    }
    if not np.isfinite(list(factors.values())).all() or min(factors.values()) <= 0:
        raise ValueError(
            f"daily portfolio factor is invalid on {business_date.date()}"
        )
    return factors


def _one_way_turnover(previous: pd.Series, target: pd.Series) -> float:
    instruments = previous.index.union(target.index)
    aligned_previous = previous.reindex(instruments, fill_value=0.0)
    aligned_target = target.reindex(instruments, fill_value=0.0)
    return 0.5 * float((aligned_target - aligned_previous).abs().sum())


__all__ = ["IndexSimulationResult", "IndexSimulator"]
