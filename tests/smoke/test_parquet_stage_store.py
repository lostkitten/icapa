from __future__ import annotations

from decimal import Decimal
from enum import Enum
import json

import numpy as np
import pandas as pd
import pytest

import icapa.workspace.caches.diagnostic_types as diagnostic_types_module
from icapa.backtesting import Backtester, Calendar
from icapa.workspace import (
    CachePolicy,
    CacheMissError,
    CacheSource,
    CacheStage,
    ArtifactIntegrityError,
    ParquetStageStore,
    ReviewArtifact,
    WorkspaceRepository,
    automatic_digest,
    register_review_diagnostic_enum,
)
from icapa.workspace.identity import canonical_json_bytes


@register_review_diagnostic_enum
class _DiagnosticState(str, Enum):
    COMPLETE = "complete"


class _CountingMethodology:
    def __init__(self, calls: dict[str, int]) -> None:
        self.calls = calls

    def execute(self, context):
        self.calls["reviews"] += 1
        context.set_dataframe(
            pd.DataFrame(
                {
                    "instrument_id": ["A", "B"],
                    "benchmark_weight": [0.6, 0.4],
                    "index_weight": [0.55, 0.45],
                }
            ).set_index("instrument_id")
        )
        context.diagnostics["review"] = {"status": "complete"}
        return context


def _calendar() -> Calendar:
    return Calendar(
        pd.DataFrame(
            {
                "reference_date": ["2026-03-20"],
                "effective_date": ["2026-03-23"],
            }
        )
    )


def test_parquet_stage_store_reuses_reviews_without_legacy_json_writes(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("parquet_review_cache")
    namespace = automatic_digest({"methodology": "test", "snapshot": "v1"})
    store = ParquetStageStore(
        workspace,
        index_id="DEMO",
        namespace_digest=namespace,
    )
    calls = {"reviews": 0}

    first = Backtester(
        index_id="DEMO",
        calendar=_calendar(),
        methodology=_CountingMethodology(calls),
        workspace_name=workspace.workspace_name,
        cache_policy=CachePolicy.REUSE,
        data_revision="snapshot-v1",
        cache_store=store,
    ).run()
    second = Backtester(
        index_id="DEMO",
        calendar=_calendar(),
        methodology=_CountingMethodology(calls),
        workspace_name=workspace.workspace_name,
        cache_policy=CachePolicy.REUSE,
        data_revision="snapshot-v1",
        cache_store=store,
    ).run()

    assert calls["reviews"] == 1
    pd.testing.assert_frame_equal(first.weights, second.weights)
    metadata = next(iter(second.metadata.reviews.values()))
    assert metadata.cache_source is CacheSource.DISK
    assert tuple(workspace.workspace_path.rglob("*.frame.json")) == ()
    assert tuple(workspace.workspace_path.rglob("reviews/*.json")) == ()
    assert tuple(workspace.workspace_path.rglob("*.parquet"))


def test_parquet_stage_store_supports_simulator_frame_and_metadata_protocol(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("parquet_simulation_cache")
    store = ParquetStageStore(
        workspace,
        index_id="DEMO",
        namespace_digest=automatic_digest({"simulation": "v1"}),
    )
    cache_key = automatic_digest({"segment": "2026-07"})
    daily = pd.DataFrame(
        {"index_return": [0.01, -0.02]},
        index=pd.DatetimeIndex(
            ["2026-07-01", "2026-07-02"],
            name="business_date",
        ),
    )

    store.save_frame("simulation", cache_key, "daily", daily)
    store.save_json(
        "simulation",
        cache_key,
        "metadata",
        {"cache_source": "computed", "base_value": 1000.0},
    )

    pd.testing.assert_frame_equal(
        store.load_frame("simulation", cache_key, "daily"),
        daily,
    )
    assert store.load_json("simulation", cache_key, "metadata") == {
        "base_value": 1000.0,
        "cache_source": "computed",
    }


def test_missing_review_artifact_is_rebuilt_but_corruption_is_rejected(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("parquet_review_repair")
    store = ParquetStageStore(
        workspace,
        index_id="DEMO",
        namespace_digest=automatic_digest({"methodology": "repair"}),
    )
    calls = {"reviews": 0}

    def run(policy=CachePolicy.REUSE):
        return Backtester(
            index_id="DEMO",
            calendar=_calendar(),
            methodology=_CountingMethodology(calls),
            workspace_name=workspace.workspace_name,
            cache_policy=policy,
            data_revision="snapshot-v1",
            cache_store=store,
        ).run()

    run()
    cache_key = store._review_key(  # internal identity is automatic
        pd.Timestamp("2026-03-20"),
        pd.Timestamp("2026-03-23"),
    )
    reference = workspace.resolve_artifact(
        stage=CacheStage.REVIEWS,
        cache_key=cache_key,
        name="review-commit",
    )
    assert reference is not None
    path = workspace.workspace_path.joinpath(reference.relative_path)
    path.unlink()

    run()

    assert calls["reviews"] == 2
    repaired = workspace.resolve_artifact(
        stage=CacheStage.REVIEWS,
        cache_key=cache_key,
        name="review-commit",
    )
    assert repaired is not None
    repaired_path = workspace.workspace_path.joinpath(
        repaired.relative_path
    )
    repaired_path.write_bytes(repaired_path.read_bytes() + b"corrupt")
    with pytest.raises(ArtifactIntegrityError):
        run()


def _review_artifact(diagnostics, *, provenance_records=()):
    return ReviewArtifact(
        reference_date=pd.Timestamp("2026-03-20"),
        effective_date=pd.Timestamp("2026-03-23"),
        index_id="DEMO",
        universe_id="GENERIC_UNIVERSE",
        constituents=pd.DataFrame(
            {
                "instrument_id": ["A", "B"],
                "benchmark_weight": [0.6, 0.4],
                "index_weight": [0.55, 0.45],
            }
        ).set_index("instrument_id"),
        diagnostics=diagnostics,
        provenance_records=tuple(provenance_records),
    )


def test_review_provider_provenance_roundtrips_with_cached_context(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("review_provenance_roundtrip")
    store = ParquetStageStore(
        workspace,
        index_id="DEMO",
        namespace_digest=automatic_digest({"provenance": "provider"}),
    )
    records = (
        {
            "provider": {
                "provider_name": "synthetic",
                "capability": "load_universe",
            },
            "request_digest": "1" * 64,
            "content_digest": "2" * 64,
        },
    )

    store.save_review(
        _review_artifact({}, provenance_records=records)
    )
    restored = store.load_review(
        "2026-03-20",
        "2026-03-23",
    ).artifact

    assert restored.provenance_records == records


def test_nested_review_diagnostic_tables_and_typed_metadata_roundtrip(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("review_diagnostic_roundtrip")
    store = ParquetStageStore(
        workspace,
        index_id="DEMO",
        namespace_digest=automatic_digest({"diagnostics": "nested"}),
    )
    constraint_index = pd.MultiIndex.from_arrays(
        [
            pd.DatetimeIndex(
                ["2026-03-23", "2026-03-23"]
            ).as_unit("s"),
            ["A", "B"],
        ],
        names=["effective_date", "instrument_id"],
    )
    constraint_columns = pd.MultiIndex.from_tuples(
        [
            ("constraint", "requested"),
            ("constraint", "achieved"),
        ],
        names=["category", "measure"],
    )
    constraint_table = pd.DataFrame(
        [[0.40, 0.39], [0.60, 0.61]],
        index=constraint_index,
        columns=constraint_columns,
    )
    slack = pd.Series(
        [0.01, 0.02],
        index=pd.date_range(
            "2026-03-23",
            periods=2,
            freq="D",
            name="business_date",
        ).as_unit("ms"),
        name=("constraint", "slack"),
    )
    lagged = pd.Series(
        [0.03, 0.04],
        index=pd.timedelta_range(
            "0h",
            periods=2,
            freq="2h",
            name="lag",
        ).as_unit("us"),
        name="lagged_slack",
    )
    rolling = pd.DataFrame(
        [[0.10, 0.20]],
        columns=pd.date_range(
            "2026-03-23",
            periods=2,
            freq="D",
            name="business_date",
        ).as_unit("s"),
    )
    diagnostics = {
        "solver": {
            "state": _DiagnosticState.COMPLETE,
            "as_of": pd.Timestamp("2026-03-20 17:00:00").as_unit("us"),
            "iterations": np.int64(4),
            "objective": float("nan"),
            "tolerance": Decimal("0.00000001"),
            "phases": ("phase_one", "main"),
            "constraint_diagnostics": [
                {
                    "group": "country",
                    "detail": constraint_table,
                },
                {
                    "group": "instrument",
                    "slack": slack,
                },
            ],
            "lagged": lagged,
            "rolling": rolling,
        }
    }

    store.save_review(_review_artifact(diagnostics))
    enum_id = (
        f"{_DiagnosticState.__module__}:{_DiagnosticState.__qualname__}"
    )
    diagnostic_types_module.DIAGNOSTIC_ENUM_REGISTRY.pop(enum_id)
    try:
        with pytest.raises(ValueError, match="diagnostic Enum is not registered"):
            store.load_review("2026-03-20", "2026-03-23")
    finally:
        register_review_diagnostic_enum(_DiagnosticState)
    restored = store.load_review("2026-03-20", "2026-03-23").artifact.diagnostics

    solver = restored["solver"]
    assert solver["state"] is _DiagnosticState.COMPLETE
    assert solver["as_of"] == diagnostics["solver"]["as_of"]
    assert solver["as_of"].unit == "us"
    assert solver["iterations"] == 4
    assert np.isnan(solver["objective"])
    assert solver["tolerance"] == Decimal("0.00000001")
    assert solver["phases"] == ("phase_one", "main")
    pd.testing.assert_frame_equal(
        solver["constraint_diagnostics"][0]["detail"],
        constraint_table,
    )
    pd.testing.assert_series_equal(
        solver["constraint_diagnostics"][1]["slack"],
        slack,
    )
    pd.testing.assert_series_equal(solver["lagged"], lagged)
    pd.testing.assert_frame_equal(solver["rolling"], rolling)


def test_unregistered_diagnostic_enum_is_rejected_before_write(
    tmp_path,
    monkeypatch,
):
    class LocalState(str, Enum):
        COMPLETE = "complete"

    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("unregistered_diagnostic_enum")
    store = ParquetStageStore(
        workspace,
        index_id="DEMO",
        namespace_digest=automatic_digest({"diagnostics": "enum"}),
    )

    with pytest.raises(ValueError, match="unregistered diagnostic Enum"):
        store.save_review(
            _review_artifact({"state": LocalState.COMPLETE})
        )

    assert workspace.resolve_artifact(
        stage=CacheStage.REVIEWS,
        cache_key=store._review_key(
            pd.Timestamp("2026-03-20"),
            pd.Timestamp("2026-03-23"),
        ),
        name="review-commit",
    ) is None


@pytest.mark.parametrize("damage", ("missing", "corrupt"))
def test_nested_review_diagnostic_artifact_damage_is_detected(
    tmp_path,
    monkeypatch,
    damage,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open(f"review_diagnostic_{damage}")
    store = ParquetStageStore(
        workspace,
        index_id="DEMO",
        namespace_digest=automatic_digest(
            {"diagnostics": damage}
        ),
    )
    store.save_review(
        _review_artifact(
            {
                "constraints": {
                    "detail": pd.DataFrame(
                        {
                            "name": ["maximum_weight"],
                            "requested": [0.10],
                            "achieved": [0.09],
                        }
                    )
                }
            }
        )
    )
    cache_key = store._review_key(
        pd.Timestamp("2026-03-20"),
        pd.Timestamp("2026-03-23"),
    )
    commit_reference = workspace.resolve_artifact(
        stage=CacheStage.REVIEWS,
        cache_key=cache_key,
        name="review-commit",
    )
    assert commit_reference is not None
    commit_frame = workspace.load_frame(commit_reference)
    commit = json.loads(commit_frame.iloc[0]["payload_json"])
    table_node = _first_diagnostic_table_node(commit["diagnostics"])
    table_reference = workspace.resolve_artifact(
        stage=CacheStage.REVIEWS,
        cache_key=cache_key,
        name=table_node["binding"],
    )
    assert table_reference is not None
    table_path = workspace.workspace_path.joinpath(
        table_reference.relative_path
    )
    if damage == "missing":
        table_path.unlink()
        expected_error = CacheMissError
    else:
        raw = table_path.read_bytes()
        table_path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        expected_error = ArtifactIntegrityError

    with pytest.raises(expected_error):
        store.load_review("2026-03-20", "2026-03-23")


def test_schema_one_review_commit_remains_readable(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("legacy_review_commit")
    store = ParquetStageStore(
        workspace,
        index_id="DEMO",
        namespace_digest=automatic_digest({"legacy": "schema-one"}),
    )
    artifact = _review_artifact(
        {"solver": {"status": "complete", "iterations": 3}}
    )
    constituents = workspace.save_frame(
        "review_constituents",
        artifact.constituents,
    )
    commit = {
        "schema_version": 1,
        "kind": "review",
        "index_id": "DEMO",
        "reference_date": pd.Timestamp("2026-03-20"),
        "effective_date": pd.Timestamp("2026-03-23"),
        "universe_id": artifact.universe_id,
        "constituents": {
            field: getattr(constituents, field)
            for field in constituents.__dataclass_fields__
        },
        "daily": None,
        "diagnostics": artifact.diagnostics,
    }
    commit_reference = workspace.save_frame(
        "review_commit",
        pd.DataFrame(
            {
                "payload_json": [
                    canonical_json_bytes(commit).decode("utf-8")
                ]
            }
        ),
    )
    legacy_key = store._review_key(
        pd.Timestamp("2026-03-20"),
        pd.Timestamp("2026-03-23"),
        schema_version=1,
    )
    workspace.bind_artifact(
        stage=CacheStage.REVIEWS,
        cache_key=legacy_key,
        name="review-commit",
        artifact=commit_reference,
    )

    loaded = store.load_review(
        "2026-03-20",
        "2026-03-23",
    ).artifact

    pd.testing.assert_frame_equal(
        loaded.constituents,
        artifact.constituents,
    )
    assert loaded.diagnostics == artifact.diagnostics


def test_unsupported_review_diagnostic_value_fails_before_writing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("unsupported_review_diagnostic")
    store = ParquetStageStore(
        workspace,
        index_id="DEMO",
        namespace_digest=automatic_digest({"diagnostics": "unsupported"}),
    )

    with pytest.raises(
        ValueError,
        match="unsupported diagnostic value type object",
    ):
        store.save_review(_review_artifact({"opaque": object()}))

    assert tuple(workspace.workspace_path.rglob("*.parquet")) == ()


def _first_diagnostic_table_node(value):
    if isinstance(value, dict):
        if value.get("__icapa_review_diagnostic_type__") in {
            "dataframe",
            "series",
        }:
            return value
        for item in value.values():
            try:
                return _first_diagnostic_table_node(item)
            except LookupError:
                pass
    elif isinstance(value, list):
        for item in value:
            try:
                return _first_diagnostic_table_node(item)
            except LookupError:
                pass
    raise LookupError("diagnostic table reference not found")
