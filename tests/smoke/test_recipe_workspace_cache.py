"""Focused tests for the persistent recipe-stage cache adapter."""

from __future__ import annotations

import multiprocessing
import os

import pandas as pd
import pytest

from icapa.portfolio_construction import (
    Artifact,
    ArtifactKey,
    ArtifactOutput,
    CORE_TARGET_WEIGHTS,
    IndexRecipe,
    RecipeRunner,
    ReviewIdentity,
    StageCacheScope,
    StageCacheSource,
    StageDescriptor,
    StageDiagnostic,
    StageNode,
    StageRequirements,
    StageResult,
    StageRuntime,
)
from icapa.workspace import CacheMode, CacheStage, WorkspaceRepository
from icapa.workspace.caches.recipe import (
    WorkspaceStageCache,
    WorkspaceStageCacheCollisionError,
    WorkspaceStageCacheIntegrityError,
    WorkspaceStageCacheMissError,
    WorkspaceStageCacheSerializationError,
)


DETAILS = ArtifactKey("test", "details")
SUMMARY = ArtifactKey("test", "summary")


def _concurrent_stage_result(variant: int) -> StageResult:
    index = pd.Index(["A", "B"], name="instrument_id")
    return StageResult(
        artifacts={
            CORE_TARGET_WEIGHTS: Artifact.from_value(
                CORE_TARGET_WEIGHTS,
                pd.Series(
                    [0.4 + 0.1 * variant, 0.6 - 0.1 * variant],
                    index=index,
                    name="index_weight",
                ),
                metadata={"variant": variant},
            ),
            DETAILS: Artifact.from_value(
                DETAILS,
                pd.DataFrame(
                    {
                        "variant": [variant, variant],
                        "signal": [
                            float(variant),
                            float(variant + 1),
                        ],
                    },
                    index=index,
                ),
                metadata={"variant": variant},
            ),
            SUMMARY: Artifact.from_value(
                SUMMARY,
                {"variant": variant, "selected": 2},
                metadata={"variant": variant},
            ),
        },
        diagnostics=(
            StageDiagnostic(
                code=f"variant_{variant}",
                message=f"Variant {variant} completed.",
                metrics={"variant": variant},
            ),
        ),
    )


def _concurrent_stage_cache_writer(
    workspace_root,
    workspace_name,
    cache_key,
    variant,
    initial_check_barrier,
    outcomes,
):
    os.environ["ICAPA_WORKSPACE_ROOT"] = str(workspace_root)
    cache = WorkspaceStageCache(
        WorkspaceRepository.open(workspace_name),
        mode=CacheMode.REUSE,
    )
    original_load = cache._load_committed
    first_check = True

    def synchronized_initial_check(key):
        nonlocal first_check
        existing = original_load(key)
        if first_check:
            first_check = False
            initial_check_barrier.wait(timeout=30)
        return existing

    cache._load_committed = synchronized_initial_check
    try:
        cache.save(cache_key, _concurrent_stage_result(variant))
    except WorkspaceStageCacheCollisionError as exc:
        outcomes.put(("collision", variant, str(exc)))
    except BaseException as exc:
        outcomes.put(("error", variant, type(exc).__name__, str(exc)))
    else:
        outcomes.put(("saved", variant))


class PersistentOutputStage:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def descriptor(self):
        return StageDescriptor(
            "test.persistent_output",
            cache_scope=StageCacheScope.CONTENT,
        )

    @property
    def requirements(self):
        return StageRequirements()

    @property
    def outputs(self):
        return (
            ArtifactOutput(CORE_TARGET_WEIGHTS),
            ArtifactOutput(DETAILS),
            ArtifactOutput(SUMMARY),
        )

    def canonical_configuration(self):
        return {"method": "deterministic"}

    def run(self, inputs, runtime):
        self.calls += 1
        index = pd.Index(["A", "B"], name="instrument_id")
        return StageResult(
            artifacts={
                CORE_TARGET_WEIGHTS: Artifact.from_value(
                    CORE_TARGET_WEIGHTS,
                    pd.Series(
                        [0.4, 0.6],
                        index=index,
                        name="index_weight",
                    ),
                    metadata={"unit": "weight"},
                ),
                DETAILS: Artifact.from_value(
                    DETAILS,
                    pd.DataFrame(
                        {"signal": [1.0, 2.0]},
                        index=index,
                    ),
                    metadata={"source": "pseudodata"},
                ),
                SUMMARY: Artifact.from_value(
                    SUMMARY,
                    {"selected": 2, "quality": ["complete"]},
                ),
            },
            diagnostics=(
                StageDiagnostic(
                    code="stage_complete",
                    message="Stage completed.",
                    metrics={"selected": 2},
                ),
            ),
        )


def _recipe(stage):
    return IndexRecipe(
        recipe_id="persistent_stage_recipe",
        recipe_version="1",
        nodes=(StageNode("weights", stage),),
    )


def _review():
    return ReviewIdentity(
        index_id="RESEARCH_INDEX",
        reference_date="2026-03-10",
        effective_date="2026-03-20",
    )


def _runtime():
    return StageRuntime(
        data_revision="pseudodata-snapshot-1",
        code_revision="automatic-code-1",
    )


def test_stage_result_is_reused_by_new_workspace_and_runner_instances(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    first_stage = PersistentOutputStage()
    first_workspace = WorkspaceRepository.open("recipe_stage_reuse")
    first = RecipeRunner(
        cache=WorkspaceStageCache(first_workspace),
        runtime=_runtime(),
    ).run_review(_recipe(first_stage), _review())

    second_stage = PersistentOutputStage()
    second_workspace = WorkspaceRepository.open("recipe_stage_reuse")
    second = RecipeRunner(
        cache=WorkspaceStageCache(second_workspace),
        runtime=_runtime(),
    ).run_review(_recipe(second_stage), _review())

    assert first_stage.calls == 1
    assert second_stage.calls == 0
    assert second.stages[0].cache_source is StageCacheSource.CACHE
    pd.testing.assert_series_equal(second.target_weights, first.target_weights)
    pd.testing.assert_frame_equal(
        second.artifacts[DETAILS].value,
        first.artifacts[DETAILS].value,
    )
    assert second.artifacts[SUMMARY].value == {
        "quality": ["complete"],
        "selected": 2,
    }
    assert dict(second.artifacts[DETAILS].metadata) == {
        "source": "pseudodata"
    }
    assert second.stages[0].diagnostics == first.stages[0].diagnostics


@pytest.mark.parametrize("damage", ("missing", "corrupt"))
def test_stage_cache_rejects_missing_or_corrupt_immutable_artifacts(
    tmp_path,
    monkeypatch,
    damage,
):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open(f"recipe_stage_{damage}")
    result = RecipeRunner(
        cache=WorkspaceStageCache(workspace),
        runtime=_runtime(),
    ).run_review(_recipe(PersistentOutputStage()), _review())
    cache_key = result.stages[0].cache_key
    assert cache_key is not None

    reference = workspace.resolve_artifact(
        stage=CacheStage.REVIEWS,
        cache_key=cache_key,
        name=(
            "stage-value-0000"
            if damage == "missing"
            else "stage-result"
        ),
    )
    assert reference is not None
    path = workspace.workspace_path.joinpath(reference.relative_path)
    if damage == "missing":
        path.unlink()
    else:
        raw = path.read_bytes()
        path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))

    reloaded = WorkspaceStageCache(
        WorkspaceRepository.open(f"recipe_stage_{damage}")
    )
    with pytest.raises(WorkspaceStageCacheIntegrityError):
        reloaded.load(cache_key)


def test_stage_cache_rejects_opaque_values_without_coercion(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    cache = WorkspaceStageCache(WorkspaceRepository.open("recipe_stage_opaque"))
    opaque_key = ArtifactKey("test", "opaque")
    result = StageResult(
        {opaque_key: Artifact.from_value(opaque_key, object())}
    )

    with pytest.raises(
        WorkspaceStageCacheSerializationError,
        match="unsupported type",
    ):
        cache.save("a" * 64, result)


def test_stage_cache_modes_enforce_refresh_and_read_only(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    workspace = WorkspaceRepository.open("recipe_stage_modes")
    recipe = _recipe(PersistentOutputStage())
    with pytest.raises(WorkspaceStageCacheMissError, match="READ_ONLY"):
        RecipeRunner(
            cache=WorkspaceStageCache(
                workspace,
                mode=CacheMode.READ_ONLY,
            ),
            runtime=_runtime(),
        ).run_review(recipe, _review())

    initial_stage = PersistentOutputStage()
    RecipeRunner(
        cache=WorkspaceStageCache(workspace),
        runtime=_runtime(),
    ).run_review(_recipe(initial_stage), _review())
    refreshed_stage = PersistentOutputStage()
    refreshed = RecipeRunner(
        cache=WorkspaceStageCache(
            workspace,
            mode=CacheMode.REFRESH,
        ),
        runtime=_runtime(),
    ).run_review(_recipe(refreshed_stage), _review())

    assert initial_stage.calls == 1
    assert refreshed_stage.calls == 1
    assert refreshed.stages[0].cache_source is StageCacheSource.EXECUTED


def test_concurrent_distinct_multi_artifact_results_commit_atomically(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("ICAPA_WORKSPACE_ROOT", str(tmp_path))
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    outcomes = context.Queue()
    workspace_name = "recipe_stage_concurrent_commit"
    cache_key = "c" * 64
    processes = [
        context.Process(
            target=_concurrent_stage_cache_writer,
            args=(
                str(tmp_path),
                workspace_name,
                cache_key,
                variant,
                barrier,
                outcomes,
            ),
        )
        for variant in (0, 1)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=45)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)

    observed = [outcomes.get(timeout=10) for _ in processes]
    outcomes.close()
    outcomes.join_thread()
    assert all(process.exitcode == 0 for process in processes), observed
    assert sorted(item[0] for item in observed) == [
        "collision",
        "saved",
    ]
    assert not [item for item in observed if item[0] == "error"]

    committed = WorkspaceStageCache(
        WorkspaceRepository.open(workspace_name),
        mode=CacheMode.READ_ONLY,
    ).load(cache_key)
    assert committed is not None
    variant = int(committed.artifacts[SUMMARY].value["variant"])
    assert variant in {0, 1}
    assert committed.artifacts[SUMMARY].value == {
        "selected": 2,
        "variant": variant,
    }
    assert set(
        committed.artifacts[DETAILS].value["variant"].tolist()
    ) == {variant}
    pd.testing.assert_series_equal(
        committed.artifacts[CORE_TARGET_WEIGHTS].value,
        pd.Series(
            [0.4 + 0.1 * variant, 0.6 - 0.1 * variant],
            index=pd.Index(["A", "B"], name="instrument_id"),
            name="index_weight",
        ),
    )
    assert dict(committed.artifacts[DETAILS].metadata) == {
        "variant": variant
    }
    assert committed.diagnostics[0].code == f"variant_{variant}"
