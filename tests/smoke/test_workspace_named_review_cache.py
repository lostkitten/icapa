"""Focused tests for named workspace review artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from icapa.backtesting import Backtester, Calendar
from icapa.workspace import (
    CacheIntegrityError,
    CacheMissError,
    CachePolicy,
    CacheSource,
    WorkspaceStore,
    clear_memory_cache,
    get_workspace_root,
)


class CountingMethodology:
    calls = 0

    def execute(self, data_context):
        type(self).calls += 1
        data_context.universe_id = "DEMO_UNIVERSE"
        data_context.diagnostics = {"solver": {"iterations": 3}}
        data_context.set_dataframe(
            pd.DataFrame(
                {
                    "instrument_id": [101, 102],
                    "benchmark_weight": [0.4, 0.6],
                    "index_weight": [0.4, 0.6],
                }
            )
        )
        return data_context


REVIEW_ROWS = [
    {"reference_date": "2026-01-05", "effective_date": "2026-01-12"},
    {"reference_date": "2026-02-02", "effective_date": "2026-02-09"},
    {"reference_date": "2026-03-02", "effective_date": "2026-03-09"},
]


def _backtester(rows, **kwargs):
    return Backtester(
        index_id="DEMO_INDEX",
        calendar=Calendar.from_dates(rows),
        methodology=CountingMethodology(),
        workspace_name="baseline_run",
        data_revision="revision-1",
        **kwargs,
    )


def test_named_workspace_reuses_overlap_across_shorter_and_longer_ranges(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    clear_memory_cache()
    CountingMethodology.calls = 0

    initial = _backtester(REVIEW_ROWS[:2]).run()
    assert CountingMethodology.calls == 2
    assert {
        item.cache_source for item in initial.metadata.reviews.values()
    } == {CacheSource.COMPUTED}

    shorter = _backtester(REVIEW_ROWS[1:2]).run()
    assert CountingMethodology.calls == 2
    assert next(iter(shorter.metadata.reviews.values())).cache_source is CacheSource.MEMORY
    assert shorter.metadata.fingerprint == initial.metadata.fingerprint

    clear_memory_cache()
    longer = _backtester(REVIEW_ROWS).run()
    sources = [
        longer.metadata.reviews[pd.Timestamp(row["effective_date"])].cache_source
        for row in REVIEW_ROWS
    ]
    assert sources == [CacheSource.DISK, CacheSource.DISK, CacheSource.COMPUTED]
    assert CountingMethodology.calls == 3
    assert longer.metadata.fingerprint == initial.metadata.fingerprint
    assert all(
        context.diagnostics == {"solver": {"iterations": 3}}
        for context in longer.reviews.values()
    )

    manifest = longer.metadata
    assert Path(manifest.workspace_path).joinpath("manifest.json").is_file()


def test_cache_policies_and_fingerprint_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    clear_memory_cache()
    CountingMethodology.calls = 0

    original = _backtester(REVIEW_ROWS[:1]).run()
    refreshed = _backtester(
        REVIEW_ROWS[:1],
        cache_policy=CachePolicy.REFRESH,
    ).run()
    assert CountingMethodology.calls == 2
    assert next(iter(refreshed.metadata.reviews.values())).cache_source is CacheSource.COMPUTED
    assert refreshed.metadata.fingerprint == original.metadata.fingerprint

    changed_revision = Backtester(
        index_id="DEMO_INDEX",
        calendar=Calendar.from_dates(REVIEW_ROWS[:1]),
        methodology=CountingMethodology(),
        workspace_name="baseline_run",
        data_revision="revision-2",
    ).run()
    assert changed_revision.metadata.fingerprint != original.metadata.fingerprint
    assert CountingMethodology.calls == 3

    changed_configuration = _backtester(
        REVIEW_ROWS[:1],
        cache_configuration={"calculation_mode": "strict"},
    ).run()
    assert changed_configuration.metadata.fingerprint != original.metadata.fingerprint
    assert CountingMethodology.calls == 4

    read_only = _backtester(
        REVIEW_ROWS[:1],
        cache_policy=CachePolicy.READ_ONLY,
    ).run()
    assert next(iter(read_only.metadata.reviews.values())).cache_source in {
        CacheSource.MEMORY,
        CacheSource.DISK,
    }
    with pytest.raises(CacheMissError):
        _backtester(
            REVIEW_ROWS,
            cache_policy=CachePolicy.READ_ONLY,
        ).run()


def test_workspace_integrity_generic_artifacts_and_report_paths(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    clear_memory_cache()
    CountingMethodology.calls = 0

    backtester = _backtester(REVIEW_ROWS[:1])
    result = backtester.run()
    store = backtester.workspace_store
    assert isinstance(store, WorkspaceStore)
    assert get_workspace_root() == tmp_path.resolve()

    frame = pd.DataFrame(
        {"value": [1.5, 2.5]},
        index=pd.Index(["a", "b"], name="instrument_id"),
    )
    store.save_frame("simulation", "segment-1", "daily_returns", frame)
    store.save_json("simulation", "segment-1", "metadata", {"rows": 2})
    clear_memory_cache()
    pd.testing.assert_frame_equal(
        store.load_frame("simulation", "segment-1", "daily_returns"),
        frame,
    )
    assert store.load_json("simulation", "segment-1", "metadata") == {"rows": 2}
    assert backtester.report_path("summary.xlsx").parent == backtester.reports_path
    assert backtester.reports_path.is_dir()

    review_metadata = next(iter(result.metadata.reviews.values()))
    Path(review_metadata.artifact_path).write_text("damaged", encoding="utf-8")
    clear_memory_cache()
    with pytest.raises(CacheIntegrityError):
        _backtester(REVIEW_ROWS[:1]).run()


def test_no_workspace_preserves_in_memory_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    CountingMethodology.calls = 0
    result = Backtester(
        index_id="DEMO_INDEX",
        calendar=Calendar.from_dates(REVIEW_ROWS[:1]),
        methodology=CountingMethodology(),
    ).run()

    assert CountingMethodology.calls == 1
    assert result.metadata.workspace_name is None
    assert result.metadata.fingerprint is None
    assert next(iter(result.metadata.reviews.values())).cache_source is CacheSource.COMPUTED
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    "workspace_name",
    ("", "../outside", "two words", "/absolute", "."),
)
def test_workspace_names_are_safe(tmp_path, monkeypatch, workspace_name):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    with pytest.raises(ValueError):
        Backtester(
            index_id="DEMO_INDEX",
            calendar=Calendar.from_dates(REVIEW_ROWS[:1]),
            methodology=CountingMethodology(),
            workspace_name=workspace_name,
        )
