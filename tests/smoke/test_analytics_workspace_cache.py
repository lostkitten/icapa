"""Focused tests for immutable analytics plugin result caching."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icapa.analytics import (
    AnalyticsDiagnostic,
    AnalyticsPluginResult,
    AnalyticsPluginSpec,
    AnalyticsResult,
    AnalyticsRunResult,
    AnalyticsSpec,
    BrinsonAttribution,
)
from icapa.workspace import CacheMode, WorkspaceRepository
from icapa.workspace.caches.analytics import (
    AnalyticsCacheIdentity,
    AnalyticsCacheSource,
    AnalyticsWorkspaceCache,
    AnalyticsWorkspaceCacheCollisionError,
    AnalyticsWorkspaceCacheIntegrityError,
    AnalyticsWorkspaceCacheMissError,
    AnalyticsWorkspaceCacheSerializationError,
)


def _analytics_result(marker: float = 1.0) -> AnalyticsRunResult:
    dates = pd.DatetimeIndex(
        ["2026-03-23", "2026-06-22"],
        name="effective_date",
    )
    review_validation = pd.DataFrame(
        {
            "portfolio_weight_sum": [1.0, 1.0],
            "is_valid": [True, True],
        },
        index=dates,
    )
    review_metrics = pd.DataFrame(
        {
            "active_share": [0.1 * marker, 0.2 * marker],
            "constituent_count": [3, 4],
        },
        index=dates,
    )
    exposure_index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-03-23"), "US"),
            (pd.Timestamp("2026-06-22"), "US"),
        ],
        names=["effective_date", "country"],
    )
    country_exposures = pd.DataFrame(
        {
            "portfolio_weight": [0.7, 0.6],
            "benchmark_weight": [0.6, 0.6],
            "active_weight": [0.1, 0.0],
        },
        index=exposure_index,
    )
    industry_exposures = country_exposures.copy(deep=True)
    industry_exposures.index = industry_exposures.index.set_names(
        ["effective_date", "industry"]
    )
    target_review_weight_change = pd.DataFrame(
        {"one_way_target_weight_change": [0.15 * marker]},
        index=pd.DatetimeIndex(["2026-06-22"], name="effective_date"),
    )
    formal_turnover = pd.DataFrame(
        {
            "scheduled_effective_date": [pd.Timestamp("2026-06-22")],
            "formal_one_way_turnover": [0.14 * marker],
        }
    )
    performance = pd.Series(
        {
            "observations": 20.0,
            "total_return": 0.03 * marker,
            "information_ratio": np.nan,
        },
        name="value",
    )
    business_dates = pd.DatetimeIndex(
        ["2026-06-22", "2026-06-23"],
        name="business_date",
    )
    drawdowns = pd.DataFrame(
        {
            "index_level": [1.0, 0.99],
            "benchmark_level": [1.0, 0.995],
            "index_drawdown": [0.0, -0.01],
            "benchmark_drawdown": [0.0, -0.005],
            "active_return": [0.0, -0.005],
        },
        index=business_dates,
    )
    brinson = BrinsonAttribution(
        detail=pd.DataFrame(
            {
                "allocation": [0.01 * marker],
                "selection": [0.02 * marker],
            },
            index=pd.Index(["Technology"], name="industry"),
        ),
        totals=pd.DataFrame(
            {"total_attribution": [0.03 * marker]},
            index=pd.Index(["Q2"], name="period"),
        ),
    )
    diagnostic = AnalyticsDiagnostic(
        level="info",
        code="calculated",
        message="Analytics completed.",
    )
    legacy = AnalyticsResult(
        review_validation=review_validation,
        review_metrics=review_metrics,
        country_exposures=country_exposures,
        industry_exposures=industry_exposures,
        target_review_weight_change=target_review_weight_change,
        formal_turnover=formal_turnover,
        performance=performance,
        drawdowns=drawdowns,
        brinson=brinson,
        diagnostics=(diagnostic,),
    )
    plugin_table = pd.DataFrame(
        {
            "effective_date": dates,
            "entrant_count": [0, int(marker)],
        }
    )
    spec = AnalyticsSpec(
        profile="cache_test",
        plugins=(
            AnalyticsPluginSpec(
                "legacy_parity",
                parameters={"windows": (21, 63)},
            ),
            AnalyticsPluginSpec("constituent_change"),
        ),
    )
    return AnalyticsRunResult(
        spec=spec,
        plugin_results={
            "legacy_parity": AnalyticsPluginResult(
                metrics={
                    "total_return": 0.03 * marker,
                    "information_ratio": np.nan,
                },
                tables={"review_metrics": review_metrics},
                diagnostics=(diagnostic,),
                metadata={"legacy_result": legacy},
            ),
            "constituent_change": AnalyticsPluginResult(
                metrics={"review_count": np.int64(2)},
                tables={"summary": plugin_table},
                metadata={
                    "windows": (21, 63),
                    "as_of": pd.Timestamp("2026-06-22"),
                },
            ),
        },
        legacy_result=legacy,
        diagnostics=(
            diagnostic,
            AnalyticsDiagnostic(
                level="warning",
                code="optional_input_skipped",
                message="An optional input was unavailable.",
            ),
        ),
    )


def _identity(snapshot: str = "snapshot-1") -> AnalyticsCacheIdentity:
    return AnalyticsCacheIdentity.from_inputs(
        definition_fingerprint="d" * 64,
        simulation_digest=snapshot,
        analytics_spec={
            "profile": "cache_test",
            "return_series": "net_total",
        },
    )


def _assert_result_equal(
    actual: AnalyticsRunResult,
    expected: AnalyticsRunResult,
) -> None:
    assert actual.spec == expected.spec
    assert actual.diagnostics == expected.diagnostics
    assert tuple(actual.plugin_results) == tuple(expected.plugin_results)
    for plugin_id in expected.plugin_results:
        actual_plugin = actual.plugin_results[plugin_id]
        expected_plugin = expected.plugin_results[plugin_id]
        assert tuple(actual_plugin.metrics) == tuple(expected_plugin.metrics)
        for key, expected_value in expected_plugin.metrics.items():
            actual_value = actual_plugin.metrics[key]
            if isinstance(expected_value, float) and np.isnan(expected_value):
                assert np.isnan(actual_value)
            else:
                assert actual_value == expected_value
        assert actual_plugin.diagnostics == expected_plugin.diagnostics
        for table_name, expected_table in expected_plugin.tables.items():
            pd.testing.assert_frame_equal(
                actual_plugin.tables[table_name],
                expected_table,
            )
    assert actual.legacy_result is not None
    assert expected.legacy_result is not None
    for name in (
        "review_validation",
        "review_metrics",
        "country_exposures",
        "industry_exposures",
        "target_review_weight_change",
        "formal_turnover",
        "drawdowns",
    ):
        pd.testing.assert_frame_equal(
            getattr(actual.legacy_result, name),
            getattr(expected.legacy_result, name),
        )
    pd.testing.assert_series_equal(
        actual.legacy_result.performance,
        expected.legacy_result.performance,
    )
    assert actual.legacy_result.brinson is not None
    assert expected.legacy_result.brinson is not None
    pd.testing.assert_frame_equal(
        actual.legacy_result.brinson.detail,
        expected.legacy_result.brinson.detail,
    )
    pd.testing.assert_frame_equal(
        actual.legacy_result.brinson.totals,
        expected.legacy_result.brinson.totals,
    )
    assert (
        actual.plugin_results["legacy_parity"].metadata["legacy_result"]
        is actual.legacy_result
    )
    assert actual.plugin_results["constituent_change"].metadata == {
        "windows": (21, 63),
        "as_of": pd.Timestamp("2026-06-22"),
    }


def test_analytics_result_roundtrips_through_immutable_parquet(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("analytics_roundtrip")
    expected = _analytics_result()

    saved = AnalyticsWorkspaceCache(workspace).save(_identity(), expected)
    loaded = AnalyticsWorkspaceCache(
        WorkspaceRepository.open("analytics_roundtrip"),
        mode=CacheMode.READ_ONLY,
    ).load(_identity())

    assert loaded is not None
    assert saved.source is AnalyticsCacheSource.EXECUTED
    assert loaded.source is AnalyticsCacheSource.WORKSPACE
    assert loaded.from_cache
    assert loaded.cache_key == _identity().cache_key
    assert saved.artifacts
    assert all(item.format == "parquet" for item in saved.artifacts)
    assert any(
        item.artifact_type == "analytics_commit"
        for item in saved.artifacts
    )
    _assert_result_equal(loaded.result, expected)


def test_reuse_and_read_only_control_analytics_call_counts(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("analytics_modes")
    calls = {"analytics": 0}

    def calculate():
        calls["analytics"] += 1
        return _analytics_result()

    first = AnalyticsWorkspaceCache(
        workspace,
        mode=CacheMode.REUSE,
    ).execute(_identity(), calculate)
    second = AnalyticsWorkspaceCache(
        WorkspaceRepository.open("analytics_modes"),
        mode=CacheMode.REUSE,
    ).execute(_identity(), calculate)
    third = AnalyticsWorkspaceCache(
        workspace,
        mode=CacheMode.READ_ONLY,
    ).execute(_identity(), calculate)

    assert calls["analytics"] == 1
    assert first.source is AnalyticsCacheSource.EXECUTED
    assert second.source is AnalyticsCacheSource.WORKSPACE
    assert third.source is AnalyticsCacheSource.WORKSPACE

    with pytest.raises(AnalyticsWorkspaceCacheMissError, match="READ_ONLY"):
        AnalyticsWorkspaceCache(
            workspace,
            mode=CacheMode.READ_ONLY,
        ).execute(_identity("missing"), calculate)
    assert calls["analytics"] == 1


def test_off_never_reads_or_writes_reusable_analytics_artifacts(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("analytics_off")
    calls = {"analytics": 0}

    def calculate():
        calls["analytics"] += 1
        return _analytics_result()

    cache = AnalyticsWorkspaceCache(workspace, mode=CacheMode.OFF)
    first = cache.execute(_identity(), calculate)
    second = cache.execute(_identity(), calculate)

    assert calls["analytics"] == 2
    assert first.artifacts == ()
    assert second.artifacts == ()
    assert tuple(workspace.workspace_path.rglob("*.parquet")) == ()


def test_refresh_recalculates_and_preserves_old_immutable_objects(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("analytics_refresh")
    calls = {"analytics": 0}

    def calculate():
        calls["analytics"] += 1
        return _analytics_result(float(calls["analytics"]))

    cache = AnalyticsWorkspaceCache(workspace, mode=CacheMode.REFRESH)
    first = cache.execute(_identity(), calculate)
    first_paths = tuple(
        workspace.workspace_path.joinpath(item.relative_path)
        for item in first.artifacts
    )
    second = cache.execute(_identity(), calculate)

    assert calls["analytics"] == 2
    assert all(path.is_file() for path in first_paths)
    assert {
        item.content_digest for item in first.artifacts
    } != {
        item.content_digest for item in second.artifacts
    }
    loaded = AnalyticsWorkspaceCache(
        workspace,
        mode=CacheMode.READ_ONLY,
    ).load(_identity())
    assert loaded is not None
    assert (
        loaded.result.plugin_results["legacy_parity"].metrics[
            "total_return"
        ]
        == pytest.approx(0.06)
    )


@pytest.mark.parametrize("damage", ("missing", "corrupt"))
def test_analytics_cache_detects_missing_or_corrupt_table_artifact(
    tmp_path,
    monkeypatch,
    damage,
):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open(f"analytics_{damage}")
    saved = AnalyticsWorkspaceCache(workspace).save(
        _identity(),
        _analytics_result(),
    )
    reference = next(
        item
        for item in saved.artifacts
        if item.artifact_type == "analytics_table"
    )
    path = workspace.workspace_path.joinpath(reference.relative_path)
    if damage == "missing":
        path.unlink()
    else:
        raw = path.read_bytes()
        path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))

    with pytest.raises(AnalyticsWorkspaceCacheIntegrityError):
        AnalyticsWorkspaceCache(
            WorkspaceRepository.open(f"analytics_{damage}"),
            mode=CacheMode.READ_ONLY,
        ).load(_identity())


def test_analytics_cache_identity_is_order_independent_and_input_sensitive():
    first = AnalyticsCacheIdentity(
        {
            "simulation_digest": "snapshot-1",
            "spec": {"profile": "standard", "window": 63},
        }
    )
    reordered = AnalyticsCacheIdentity(
        {
            "spec": {"window": 63, "profile": "standard"},
            "simulation_digest": "snapshot-1",
        }
    )
    changed = AnalyticsCacheIdentity(
        {
            "spec": {"window": 126, "profile": "standard"},
            "simulation_digest": "snapshot-1",
        }
    )

    assert first.cache_key == reordered.cache_key
    assert first.cache_key != changed.cache_key


def test_reuse_rejects_identity_collision_while_off_accepts_opaque_metadata(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("analytics_collision")
    AnalyticsWorkspaceCache(workspace).save(
        _identity(),
        _analytics_result(1.0),
    )

    with pytest.raises(AnalyticsWorkspaceCacheCollisionError):
        AnalyticsWorkspaceCache(workspace).save(
            _identity(),
            _analytics_result(2.0),
        )

    base = _analytics_result()
    opaque = AnalyticsRunResult(
        spec=base.spec,
        plugin_results={
            "custom": AnalyticsPluginResult(metadata={"opaque": object()})
        },
        legacy_result=None,
        diagnostics=(),
    )
    outcome = AnalyticsWorkspaceCache(
        workspace,
        mode=CacheMode.OFF,
    ).execute(_identity("opaque"), lambda: opaque)
    assert outcome.result is opaque
    with pytest.raises(
        AnalyticsWorkspaceCacheSerializationError,
        match="unsupported type",
    ):
        AnalyticsWorkspaceCache(workspace).save(
            _identity("opaque"),
            opaque,
        )
