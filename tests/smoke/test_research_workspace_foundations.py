"""Focused tests for automatic workspace identity and v2 artifact foundations."""

from __future__ import annotations

import builtins
from dataclasses import dataclass
import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from icapa.workspace import (
    ArtifactIntegrityError,
    ArtifactRef,
    CacheMode,
    CacheOptions,
    CacheStage,
    ParquetDependencyError,
    WorkspaceRepository,
    WorkspaceStore,
    automatic_callable_identity,
    automatic_component_identity,
    automatic_data_identity,
    automatic_digest,
    dataframe_content_digest,
    safe_parameter_identity,
)
from icapa.workspace.identity import canonical_json_bytes


@dataclass
class ExampleComponent:
    strength: float = 1.5
    password: str = "must-not-be-persisted"


class ExampleProvider:
    dataset = "research"


def test_cache_options_are_explicit_and_stage_overridable():
    options = CacheOptions(
        mode=CacheMode.REUSE,
        stage_modes={
            CacheStage.SIMULATION: CacheMode.REFRESH,
            "analytics": "off",
        },
    )

    assert options.mode_for(CacheStage.REVIEWS) is CacheMode.REUSE
    assert options.mode_for("simulation") is CacheMode.REFRESH
    assert options.mode_for(CacheStage.ANALYTICS) is CacheMode.OFF
    assert CacheOptions().mode is CacheMode.OFF


def test_automatic_identities_change_with_configuration_and_data():
    first = automatic_component_identity(ExampleComponent(strength=1.5))
    second = automatic_component_identity(ExampleComponent(strength=2.0))

    assert first["source_digest"] == second["source_digest"]
    assert first["configuration_digest"] != second["configuration_digest"]
    assert "must-not-be-persisted" not in str(first)

    frame = pd.DataFrame(
        {"instrument_id": [2, 1], "value": [0.2, 0.1]}
    )
    reordered = frame.iloc[::-1].reset_index(drop=True)
    changed = frame.assign(value=[0.2, 0.3])
    assert dataframe_content_digest(
        frame, sort_by=["instrument_id"]
    ) == dataframe_content_digest(reordered, sort_by=["instrument_id"])
    assert dataframe_content_digest(frame) != dataframe_content_digest(changed)

    identity = automatic_data_identity(
        provider_name="example",
        provider=ExampleProvider(),
        capability="load_reference_data",
        request={
            "dataset": "research",
            "password": "must-not-be-persisted",
        },
        frame=frame,
        sort_by=["instrument_id"],
    )
    assert "must-not-be-persisted" not in str(identity)
    assert identity["rows"] == 2


def test_sensitive_values_are_redacted_without_collapsing_identity():
    first_password = "research-password-one"
    second_password = "research-password-two"
    first_uri = "postgresql://researcher:first@internal/research"
    second_uri = "postgresql://researcher:second@internal/research"

    assert automatic_digest(
        {"password": first_password}
    ) != automatic_digest({"password": second_password})
    assert automatic_digest(first_uri) != automatic_digest(second_uri)

    encoded = canonical_json_bytes(
        {
            "password": first_password,
            "status": first_uri,
        }
    ).decode("utf-8")
    assert first_password not in encoded
    assert first_uri not in encoded
    assert '"redacted":true' in encoded

    first_component = automatic_component_identity(
        ExampleComponent(password=first_password)
    )
    second_component = automatic_component_identity(
        ExampleComponent(password=second_password)
    )
    assert (
        first_component["configuration_digest"]
        != second_component["configuration_digest"]
    )
    assert first_password not in str(first_component)
    assert second_password not in str(second_component)


def test_dataframe_digest_retains_redacted_text_identity():
    first = pd.DataFrame(
        {
            "request": [
                "select instrument_id from source_alpha",
                "password=research-password-one",
            ]
        }
    )
    second = pd.DataFrame(
        {
            "request": [
                "select instrument_id from source_beta",
                "password=research-password-two",
            ]
        }
    )

    assert dataframe_content_digest(first) != dataframe_content_digest(
        second
    )


def test_dataframe_digest_distinguishes_adjacent_float64_values():
    first = np.float64(0.12345678901234566)
    second = np.nextafter(first, np.float64(np.inf))
    assert first != second

    assert dataframe_content_digest(
        pd.DataFrame({"value": [first]})
    ) != dataframe_content_digest(
        pd.DataFrame({"value": [second]})
    )


def test_callable_identity_distinguishes_closure_values_and_rejects_opaque_state():
    def factory(scale):
        def calculate(value=1.0):
            return value * scale

        return calculate

    first = automatic_callable_identity(factory(1.0))
    second = automatic_callable_identity(factory(2.0))
    assert automatic_digest(first) != automatic_digest(second)

    captured_password = "must-never-appear-in-a-manifest"

    def secret_bearing_callable():
        return captured_password

    safe_identity = automatic_callable_identity(
        secret_bearing_callable
    )
    assert captured_password not in str(safe_identity)

    opaque = object()

    def unstable():
        return opaque

    with pytest.raises(ValueError, match="stable automatic identity"):
        automatic_callable_identity(unstable)


def test_component_identity_tracks_reachable_local_dependency_changes(tmp_path):
    package = tmp_path.joinpath("identity_example")
    package.mkdir()
    package.joinpath("__init__.py").write_text("", encoding="utf-8")
    dependency = package.joinpath("dependency.py")
    dependency.write_text("VALUE = 1\n", encoding="utf-8")
    package.joinpath("component.py").write_text(
        "from .dependency import VALUE\n\n"
        "class Component:\n"
        "    value = VALUE\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module("identity_example.component")
        first = automatic_component_identity(module.Component)
        dependency.write_text("VALUE = 2\n", encoding="utf-8")
        second = automatic_component_identity(module.Component)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("identity_example.component", None)
        sys.modules.pop("identity_example.dependency", None)
        sys.modules.pop("identity_example", None)

    assert first["source_digest"] == second["source_digest"]
    assert first["source_closure_digest"] != second["source_closure_digest"]
    assert first["source_file_count"] >= 2


def test_installed_component_identity_includes_distribution_python_digest():
    identity = automatic_component_identity(pd.DataFrame)

    assert identity["distribution"] == "pandas"
    assert identity["distribution_python_file_count"] > 0
    assert len(identity["distribution_python_digest"]) == 64
    assert identity["distribution_python_files_truncated"] is False


def test_parameter_identity_never_serializes_secret_values():
    identity = safe_parameter_identity(
        {
            "database": "research",
            "access_token": "sensitive-token",
            "password": "sensitive-password",
        }
    )

    assert identity["keys"] == ["access_token", "database", "password"]
    assert identity["redacted_keys"] == ["access_token", "password"]
    assert "sensitive" not in str(identity)


def test_manifest_is_automatic_fixed_and_catalogued(tmp_path, monkeypatch):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("automatic_metadata")
    manifest = workspace.start_run(
        index_id="RESEARCH_INDEX",
        definition={
            "methodology": automatic_component_identity(ExampleComponent()),
            "parameters": {"target": 0.4},
        },
        request={
            "start_date": "2025-01-01",
            "end_date": "2026-01-01",
            "password": "must-not-be-persisted",
            "status": (
                "postgresql://private-user:private-password@"
                "internal-host/research"
            ),
        },
        cache=CacheOptions.reuse(),
    )

    reference = workspace.manifest_ref(manifest)
    expected_parent = tmp_path.joinpath(
        "automatic_metadata",
        "runs",
        manifest.definition_fingerprint,
        "executions",
        manifest.execution_id,
    )
    assert Path(reference.path).parent == expected_parent.resolve()
    definition_path = expected_parent.parents[1]
    assert {
        path.name
        for path in definition_path.iterdir()
        if path.is_dir()
    } == {
        "analytics",
        "executions",
        "reports",
        "reviews",
        "simulations",
    }
    assert workspace.catalog_path.is_file()
    assert "must-not-be-persisted" not in Path(reference.path).read_text(
        encoding="utf-8"
    )
    manifest_text = Path(reference.path).read_text(encoding="utf-8")
    for sensitive_text in (
        "postgresql://",
        "private-user",
        "private-password",
        "internal-host",
    ):
        assert sensitive_text not in manifest_text
    assert workspace.open_manifest(reference) == manifest
    assert workspace.latest_manifest() == manifest
    assert workspace.list_manifests() == (reference,)
    assert {
        item["stage"]: item["mode"] for item in manifest.cache_decisions
    } == {
        "source_data": "reuse",
        "reviews": "reuse",
        "simulation": "reuse",
        "analytics": "reuse",
    }

    completed = workspace.complete_run(
        manifest,
        input_digests=[
            {
                "input_type": "canonical_universe",
                "content_digest": "c" * 64,
            }
        ],
    )
    assert completed.status == "complete"
    assert completed.result_fingerprint is not None
    assert completed.input_digests[0]["content_digest"] == "c" * 64
    assert workspace.latest_manifest() == completed

    second = workspace.start_run(
        index_id="RESEARCH_INDEX",
        definition={
            "methodology": automatic_component_identity(ExampleComponent()),
            "parameters": {"target": 0.4},
        },
        request={
            "start_date": "2025-01-01",
            "end_date": "2026-01-01",
            "password": "must-not-be-persisted",
        },
    )
    changed_input = workspace.complete_run(
        second,
        input_digests=[
            {
                "input_type": "canonical_universe",
                "content_digest": "d" * 64,
            }
        ],
    )
    assert changed_input.result_fingerprint != completed.result_fingerprint


def test_snapshot_evidence_is_part_of_result_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("snapshot_result_identity")
    common = {
        "index_id": "RESEARCH_INDEX",
        "definition": {"methodology": "automatic"},
        "request": {"effective_date": "2026-06-22"},
    }

    first = workspace.complete_run(
        workspace.start_run(**common),
        input_digests=[
            {
                "input_type": "reviews_snapshot",
                "content_digest": "a" * 64,
                "cache_source": "provider",
            }
        ],
    )
    second = workspace.complete_run(
        workspace.start_run(**common),
        input_digests=[
            {
                "input_type": "reviews_snapshot",
                "content_digest": "b" * 64,
                "cache_source": "workspace",
            }
        ],
    )

    assert first.definition_fingerprint == second.definition_fingerprint
    assert first.request_fingerprint == second.request_fingerprint
    assert first.result_fingerprint != second.result_fingerprint


def test_request_identity_can_exclude_non_calculation_governance_metadata(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("request_identity")
    calculation = {
        "start_date": "2026-01-01",
        "end_date": "2026-12-31",
    }
    baseline = workspace.start_run(
        index_id="RESEARCH_INDEX",
        definition={"methodology": "demo"},
        request={**calculation, "label": "baseline", "status": "approved"},
        request_identity=calculation,
    )
    candidate = workspace.start_run(
        index_id="RESEARCH_INDEX",
        definition={"methodology": "demo"},
        request={**calculation, "label": "candidate", "status": "draft"},
        request_identity=calculation,
    )

    assert baseline.request_fingerprint == candidate.request_fingerprint
    assert baseline.request["label"] == "baseline"
    assert candidate.request["label"] == "candidate"


def test_failed_manifest_sanitizes_original_exception(tmp_path, monkeypatch):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("failed_metadata")
    manifest = workspace.start_run(
        index_id="RESEARCH_INDEX",
        definition={"methodology": "demo"},
        request={"review": "2026-01-01"},
    )

    failed = workspace.fail_run(
        manifest,
        RuntimeError("password=do-not-persist-this"),
    )

    assert failed.status == "failed"
    assert failed.failure["error_type"] == "RuntimeError"
    assert "do-not-persist-this" not in str(failed.failure)
    assert workspace.open_manifest(workspace.manifest_ref(failed)) == failed


def test_existing_json_frame_remains_readable_from_v2_workspace(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    legacy = WorkspaceStore(
        workspace_name="legacy_read",
        fingerprint="a" * 64,
        index_id="INDEX",
        methodology_name="demo",
        configuration_digest="b" * 64,
        data_revision="automatic",
    )
    frame = pd.DataFrame(
        {"weight": [0.4, 0.6]},
        index=pd.Index(["A", "B"], name="instrument_id"),
    )
    metadata = legacy.save_frame("simulation", "segment", "weights", frame)
    manifest_entry = legacy.manifest()["artifacts"][
        "simulation/segment/weights.frame.json"
    ]
    research = WorkspaceRepository.open("legacy_read")
    path = Path(metadata.path)
    reference = ArtifactRef(
        artifact_type="simulation_weights",
        content_digest=manifest_entry["payload_checksum"],
        file_checksum=metadata.checksum,
        relative_path=str(path.relative_to(research.workspace_path)),
        format="legacy_json",
        schema_version=1,
        size_bytes=path.stat().st_size,
    )

    pd.testing.assert_frame_equal(research.load_frame(reference), frame)
    assert path.is_file()


@pytest.mark.parametrize("count", (3_000, 10_000))
def test_parquet_roundtrip_preserves_precision_and_deduplicates(
    tmp_path,
    monkeypatch,
    count,
):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("parquet_precision")
    frame = pd.DataFrame(
        {
            "instrument_id": np.arange(count),
            "index_weight": np.full(count, 1.0 / count),
            "labels": [["research", str(item % 3)] for item in range(count)],
        }
    ).set_index("instrument_id")

    first = workspace.save_frame("review_weights", frame)
    second = workspace.save_frame("review_weights", frame)
    restored = workspace.load_frame(first)
    cache_key = automatic_digest({"review": "2026-01-01"})
    workspace.bind_artifact(
        stage=CacheStage.REVIEWS,
        cache_key=cache_key,
        name="weights",
        artifact=first,
    )

    assert first == second
    assert workspace.resolve_artifact(
        stage="reviews",
        cache_key=cache_key,
        name="weights",
    ) == first
    assert first.format == "parquet"
    assert float(restored["index_weight"].sum()) == pytest.approx(1.0, abs=1e-12)
    pd.testing.assert_frame_equal(restored, frame)


def test_parquet_explicit_sort_ignores_incidental_range_index_order(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("canonical_range_index")
    frame = pd.DataFrame(
        {
            "effective_date": pd.to_datetime(
                ["2026-06-22", "2026-03-23", "2026-03-23"]
            ),
            "instrument_id": ["B", "B", "A"],
            "index_weight": [0.6, 0.4, 0.6],
        }
    )
    reordered = frame.iloc[[2, 0, 1]].reset_index(drop=True)
    sort_by = ["effective_date", "instrument_id"]

    first = workspace.save_frame(
        "canonical_weights",
        frame,
        sort_by=sort_by,
    )
    second = workspace.save_frame(
        "canonical_weights",
        reordered,
        sort_by=sort_by,
    )

    assert first.content_digest == second.content_digest
    expected = frame.sort_values(sort_by, kind="mergesort").reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(workspace.load_frame(first), expected)


def test_parquet_roundtrip_preserves_temporal_units_and_index_frequency(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("parquet_temporal_metadata")
    rows = 3
    frame = pd.DataFrame(
        {
            "observed_seconds": pd.date_range(
                "2026-01-01 09:00:00",
                periods=rows,
                freq="h",
            ).as_unit("s").array,
            "observed_milliseconds": pd.date_range(
                "2026-01-01 09:00:00",
                periods=rows,
                freq="h",
            ).as_unit("ms").array,
            "observed_microseconds": pd.date_range(
                "2026-01-01 09:00:00",
                periods=rows,
                freq="h",
            ).as_unit("us").array,
            "elapsed_microseconds": pd.timedelta_range(
                "0h",
                periods=rows,
                freq="2h",
            ).as_unit("us").array,
        },
        index=pd.date_range(
            "2026-01-01",
            periods=rows,
            freq="D",
            name="business_date",
        ).as_unit("s"),
    )
    timedelta_indexed = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0]},
        index=pd.timedelta_range(
            "0h",
            periods=rows,
            freq="2h",
            name="elapsed",
        ).as_unit("ms"),
    )

    frame_reference = workspace.save_frame("temporal_frame", frame)
    timedelta_reference = workspace.save_frame(
        "timedelta_indexed_frame",
        timedelta_indexed,
    )

    pd.testing.assert_frame_equal(
        workspace.load_frame(frame_reference),
        frame,
    )
    pd.testing.assert_frame_equal(
        workspace.load_frame(timedelta_reference),
        timedelta_indexed,
    )
    assert workspace.verify()["status"] == "ok"


def test_parquet_corruption_is_detected(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("parquet_integrity")
    frame = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.Index(["A", "B"], name="instrument_id"),
    )
    reference = workspace.save_frame("small_frame", frame)
    path = workspace.workspace_path.joinpath(reference.relative_path)
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))

    with pytest.raises(ArtifactIntegrityError):
        workspace.load_frame(reference)

    verification = workspace.verify()
    assert verification["status"] == "corrupt"
    assert verification["artifact_failures"][0]["error_type"] == (
        "ArtifactIntegrityError"
    )


def test_verify_detects_catalog_artifact_without_metadata_sidecar(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("missing_artifact_metadata")
    reference = workspace.save_frame(
        "review_weights",
        pd.DataFrame(
            {"index_weight": [0.4, 0.6]},
            index=pd.Index(["A", "B"], name="instrument_id"),
        ),
    )
    metadata_paths = tuple(
        workspace.workspace_path.glob("objects/metadata/*/*/*.json")
    )
    assert len(metadata_paths) == 1
    metadata_paths[0].unlink()

    verification = workspace.verify()

    assert verification["status"] == "corrupt"
    assert verification["artifact_failures"] == []
    assert {
        (
            failure["record_type"],
            failure["error_type"],
            failure.get("content_digest"),
            failure.get("file_checksum"),
        )
        for failure in verification["catalog_failures"]
    } == {
        (
            "artifact_metadata",
            "MissingArtifactMetadataSidecar",
            reference.content_digest,
            reference.file_checksum,
        )
    }


def test_catalog_lifecycle_preserves_immutable_objects(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("catalog_lifecycle")
    frame = pd.DataFrame(
        {"index_weight": [0.4, 0.6]},
        index=pd.Index(["A", "B"], name="instrument_id"),
    )
    reference = workspace.save_frame("review_weights", frame)
    cache_key = automatic_digest({"effective_date": "2026-06-22"})
    workspace.bind_artifact(
        stage=CacheStage.REVIEWS,
        cache_key=cache_key,
        name="weights",
        artifact=reference,
    )
    manifest = workspace.start_run(
        index_id="RESEARCH_INDEX",
        definition={"producer": "automatic"},
        request={"effective_date": "2026-06-22"},
    )
    workspace.complete_run(manifest, artifacts=[reference])

    assert workspace.verify()["status"] == "ok"
    assert workspace.invalidate(
        stage=CacheStage.REVIEWS,
        cache_key=cache_key,
        name="weights",
    ) == 1
    assert workspace.resolve_artifact(
        stage=CacheStage.REVIEWS,
        cache_key=cache_key,
        name="weights",
    ) is None
    assert workspace.workspace_path.joinpath(reference.relative_path).is_file()
    assert workspace.prune(dry_run=True)["candidate_count"] == 0

    workspace.bind_artifact(
        stage=CacheStage.REVIEWS,
        cache_key=cache_key,
        name="weights",
        artifact=reference,
    )
    workspace.catalog_path.unlink()
    rebuilt_workspace = WorkspaceRepository.open("catalog_lifecycle")
    rebuilt = rebuilt_workspace.rebuild_catalog()
    assert rebuilt == {"manifests": 1, "artifacts": 1, "bindings": 1}
    assert rebuilt_workspace.resolve_artifact(
        stage=CacheStage.REVIEWS,
        cache_key=cache_key,
        name="weights",
    ) == reference
    assert rebuilt_workspace.verify()["status"] == "ok"


def test_prune_removes_orphan_object_sidecar_and_catalog_row(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("consistent_prune")
    reference = workspace.save_frame(
        "orphan_frame",
        pd.DataFrame({"value": [1.0]}),
    )
    object_path = workspace.workspace_path.joinpath(reference.relative_path)
    metadata_paths = tuple(
        workspace.workspace_path.glob("objects/metadata/*/*/*.json")
    )
    assert object_path.is_file()
    assert len(metadata_paths) == 1

    result = workspace.prune(dry_run=False)

    assert result["candidate_count"] == 1
    assert not object_path.exists()
    assert not metadata_paths[0].exists()
    assert workspace.verify()["status"] == "ok"
    rebuilt = workspace.rebuild_catalog()
    assert rebuilt == {"manifests": 0, "artifacts": 0, "bindings": 0}


def test_parallel_process_writers_do_not_register_partial_artifacts(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    worker = """
import os
import sys
import pandas as pd
from icapa.workspace import CacheStage, WorkspaceRepository, automatic_digest

os.environ["ICAPA_WORKSPACE_ROOT"] = sys.argv[1]
item = int(sys.argv[2])
workspace = WorkspaceRepository.open("concurrent_writers")
frame = pd.DataFrame(
    {"value": [float(item)]},
    index=pd.Index([f"I{item}"], name="instrument_id"),
)
reference = workspace.save_frame("concurrent_frame", frame)
workspace.bind_artifact(
    stage=CacheStage.REVIEWS,
    cache_key=automatic_digest({"item": item}),
    name="weights",
    artifact=reference,
)
print(reference.content_digest)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", worker, str(tmp_path), str(item)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for item in range(8)
    ]
    results = [process.communicate(timeout=30) for process in processes]
    assert all(process.returncode == 0 for process in processes), results
    digests = [stdout.strip() for stdout, _ in results]

    assert len(set(digests)) == 8
    workspace = WorkspaceRepository.open("concurrent_writers")
    verification = workspace.verify()
    assert verification["status"] == "ok"
    assert verification["artifact_count"] == 8
    for item in range(8):
        reference = workspace.resolve_artifact(
            stage=CacheStage.REVIEWS,
            cache_key=automatic_digest({"item": item}),
            name="weights",
        )
        assert reference is not None
        assert workspace.load_frame(reference).iloc[0, 0] == float(item)


def test_parallel_same_key_binding_matches_rebuild_metadata(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    cache_key = automatic_digest({"shared": "binding"})
    worker = """
import os
import sys
import pandas as pd
from icapa.workspace import CacheStage, WorkspaceRepository

os.environ["ICAPA_WORKSPACE_ROOT"] = sys.argv[1]
item = int(sys.argv[2])
cache_key = sys.argv[3]
workspace = WorkspaceRepository.open("same_key_writers")
frame = pd.DataFrame(
    {"value": [float(item)]},
    index=pd.Index([f"I{item}"], name="instrument_id"),
)
reference = workspace.save_frame("concurrent_frame", frame)
for _ in range(5):
    workspace.bind_artifact(
        stage=CacheStage.REVIEWS,
        cache_key=cache_key,
        name="weights",
        artifact=reference,
    )
"""
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                worker,
                str(tmp_path),
                str(item),
                cache_key,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for item in range(8)
    ]
    results = [process.communicate(timeout=30) for process in processes]
    assert all(process.returncode == 0 for process in processes), results

    workspace = WorkspaceRepository.open("same_key_writers")
    before_rebuild = workspace.resolve_artifact(
        stage=CacheStage.REVIEWS,
        cache_key=cache_key,
        name="weights",
    )
    assert before_rebuild is not None
    expected = workspace.load_frame(before_rebuild)

    workspace.catalog_path.unlink()
    rebuilt_workspace = WorkspaceRepository.open("same_key_writers")
    rebuilt = rebuilt_workspace.rebuild_catalog()
    after_rebuild = rebuilt_workspace.resolve_artifact(
        stage=CacheStage.REVIEWS,
        cache_key=cache_key,
        name="weights",
    )

    assert rebuilt["bindings"] == 1
    assert after_rebuild == before_rebuild
    pd.testing.assert_frame_equal(
        rebuilt_workspace.load_frame(after_rebuild),
        expected,
    )


def test_missing_pyarrow_has_a_clear_error(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def import_without_pyarrow(name, *args, **kwargs):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("PyArrow deliberately unavailable in this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_pyarrow)
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("missing_pyarrow")
    with pytest.raises(ParquetDependencyError):
        workspace.save_frame("frame", pd.DataFrame({"value": [1]}))
