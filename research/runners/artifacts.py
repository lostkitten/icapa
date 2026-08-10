"""Immutable output persistence for completed research calculations."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ...analytics import AnalyticsRunResult
from ...backtesting import BacktestResult, IndexSimulationResult
from ...workspace import ArtifactRef
from .persistence import (
    _analytics_run_metadata_frame,
    _persisted_review_constituents,
    _result_artifact_sort_columns,
    _review_context_metadata_frame,
)


class _ArtifactWriter:
    """Write canonical result tables to the workspace object store."""

    def _save_result_artifacts(
        self,
        backtest: BacktestResult,
        simulation: IndexSimulationResult | None,
        analytics: AnalyticsRunResult | None,
    ) -> tuple[ArtifactRef, ...]:
        """Persist immutable run outputs; this does not bind them as cache hits."""

        frames: list[tuple[str, pd.DataFrame, Sequence[str] | None]] = [
            (
                "review_weights",
                backtest.weights,
                ("effective_date", "instrument_id"),
            )
        ]
        for effective_date, context in sorted(backtest.reviews.items()):
            date_token = pd.Timestamp(effective_date).strftime("%Y%m%d")
            frames.append(
                (
                    f"review_constituents.{date_token}",
                    _persisted_review_constituents(context.cons),
                    ("instrument_id",),
                )
            )
            if context.daily is not None:
                frames.append(
                    (
                        f"review_daily.{date_token}",
                        context.daily,
                        _result_artifact_sort_columns(context.daily),
                    )
                )
            frames.append(
                (
                    f"review_context_metadata.{date_token}",
                    _review_context_metadata_frame(context),
                    None,
                )
            )
        if simulation is not None:
            frames.extend(
                (
                    (
                        "simulation_daily",
                        simulation.daily,
                        _result_artifact_sort_columns(simulation.daily),
                    ),
                    (
                        "simulation_rebalances",
                        simulation.rebalances,
                        _result_artifact_sort_columns(simulation.rebalances),
                    ),
                )
            )
            if not simulation.holdings.empty:
                frames.append(
                    (
                        "simulation_weight_snapshots",
                        simulation.holdings,
                        _result_artifact_sort_columns(simulation.holdings),
                    )
                )
            if not simulation.asset_returns.empty:
                frames.append(
                    (
                        "simulation_asset_returns",
                        simulation.asset_returns,
                        _result_artifact_sort_columns(simulation.asset_returns),
                    )
                )
            if not simulation.weight_snapshots.empty:
                frames.append(
                    (
                        "simulation_rebalance_weight_snapshots",
                        simulation.weight_snapshots,
                        _result_artifact_sort_columns(simulation.weight_snapshots),
                    )
                )
        if analytics is not None:
            frames.append(
                (
                    "analytics.metadata",
                    _analytics_run_metadata_frame(analytics),
                    None,
                )
            )
            frames.extend(
                (
                    f"analytics.{name}",
                    frame,
                    _result_artifact_sort_columns(frame),
                )
                for name, frame in sorted(analytics.tables().items())
                if isinstance(frame, pd.DataFrame)
            )
        saved: list[ArtifactRef] = []
        for artifact_type, frame, sort_by in frames:
            saved.append(
                self._workspace.save_frame(
                    artifact_type,
                    frame,
                    sort_by=sort_by,
                )
            )
        return tuple(saved)


__all__ = ["_ArtifactWriter"]
