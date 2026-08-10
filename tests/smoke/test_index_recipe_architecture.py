"""Contract tests for extensible, cache-aware index recipes."""

from dataclasses import dataclass
import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from icapa.portfolio_construction import (
    Artifact,
    ArtifactKey,
    ArtifactOutput,
    ArtifactRequirement,
    CORE_TARGET_WEIGHTS,
    CallableStage,
    IndexRecipe,
    MemoryStageCache,
    PriorArtifactRequirement,
    PriorReviewStateError,
    PriorStatePolicy,
    RecipeCompilationError,
    RecipeCompiler,
    RecipeRunner,
    RecipeWeightProducer,
    ReviewIdentity,
    StageCacheScope,
    StageCacheSource,
    StageDescriptor,
    StageInputs,
    StageNode,
    StageRequirements,
    StageResult,
    StageRuntime,
    StageSideEffect,
)
from icapa.portfolio_construction.context import DataContext
from icapa.portfolio_construction.recipes.artifacts import artifact_digest


RAW_SIGNAL = ArtifactKey("test", "raw_signal")
FEATURES = ArtifactKey("test", "features")
MEMBERSHIP_STATE = ArtifactKey("test", "membership_state")


def test_artifact_digest_distinguishes_adjacent_float64_values():
    first = np.float64(0.12345678901234566)
    second = np.nextafter(first, np.float64(np.inf))

    assert artifact_digest(
        pd.DataFrame({"value": [first]})
    ) != artifact_digest(
        pd.DataFrame({"value": [second]})
    )
    assert artifact_digest(
        pd.Series([first], name="value")
    ) != artifact_digest(
        pd.Series([second], name="value")
    )


class CountingFeatureFunction:
    def __init__(self, counter):
        self._counter = counter

    def __call__(self, inputs, runtime):
        self._counter["feature"] += 1
        values = inputs.value(RAW_SIGNAL).copy()
        values["feature"] = values["raw_signal"] * 2.0
        return {FEATURES: values[["feature"]]}


class CountingWeightFunction:
    def __init__(self, counter, strength):
        self._counter = counter
        self.strength = strength

    def __call__(self, inputs, runtime):
        self._counter["weight"] += 1
        feature = inputs.value(FEATURES)["feature"]
        desired = (1.0 + self.strength * feature).clip(lower=0.0)
        weights = desired / desired.sum()
        return {CORE_TARGET_WEIGHTS: weights.to_frame("index_weight")}


def _recipe(counter, strength):
    feature_stage = CallableStage(
        function=CountingFeatureFunction(counter),
        output_specs=(ArtifactOutput(FEATURES),),
        input_requirements=StageRequirements(
            artifacts=(ArtifactRequirement(RAW_SIGNAL),)
        ),
        configuration={"operation": "double"},
        kind="test.feature",
        cache_scope=StageCacheScope.CONTENT,
    )
    weight_stage = CallableStage(
        function=CountingWeightFunction(counter, strength),
        output_specs=(ArtifactOutput(CORE_TARGET_WEIGHTS),),
        input_requirements=StageRequirements(
            artifacts=(ArtifactRequirement(FEATURES),)
        ),
        configuration={"strength": strength},
        kind="test.weight",
        cache_scope=StageCacheScope.CONTENT,
    )
    return IndexRecipe(
        recipe_id="cacheable_recipe",
        recipe_version="1",
        # Reverse declaration order proves artifact dependencies drive compilation.
        nodes=(
            StageNode("weight", weight_stage),
            StageNode("feature", feature_stage),
        ),
        required_artifacts=(RAW_SIGNAL,),
    )


def _review(month=1):
    return ReviewIdentity(
        index_id="SYNTHETIC_INDEX",
        reference_date=f"2026-{month:02d}-05",
        effective_date=f"2026-{month:02d}-20",
    )


def _raw_signal():
    return pd.DataFrame(
        {"raw_signal": [0.0, 0.5, 1.0]},
        index=pd.Index([1001, 1002, 1003], name="instrument_id"),
    )


def test_recipe_compiles_by_artifact_dependency_and_reuses_independent_stages():
    counter = {"feature": 0, "weight": 0}
    cache = MemoryStageCache()
    runner = RecipeRunner(
        cache=cache,
        runtime=StageRuntime(data_revision="synthetic-v1", code_revision="test-v1"),
    )
    recipe = _recipe(counter, strength=0.5)

    first = runner.run_review(
        recipe,
        _review(),
        initial_artifacts={RAW_SIGNAL: _raw_signal()},
    )
    second = runner.run_review(
        recipe,
        _review(),
        initial_artifacts={RAW_SIGNAL: _raw_signal()},
    )

    assert counter == {"feature": 1, "weight": 1}
    assert [item.node_id for item in first.stages] == ["feature", "weight"]
    assert all(item.cache_source is StageCacheSource.EXECUTED for item in first.stages)
    assert all(item.implementation_digest for item in first.stages)
    derived_seed = first.stages[0].random_seed
    assert derived_seed is not None
    assert {item.random_seed for item in first.stages} == {derived_seed}
    assert {item.random_seed for item in second.stages} == {derived_seed}
    assert all(item.cache_source is StageCacheSource.CACHE for item in second.stages)
    assert float(first.target_weights.sum()) == pytest.approx(1.0)

    changed = runner.run_review(
        _recipe(counter, strength=1.0),
        _review(),
        initial_artifacts={RAW_SIGNAL: _raw_signal()},
    )
    assert counter == {"feature": 1, "weight": 2}
    assert changed.stages[0].cache_source is StageCacheSource.CACHE
    assert changed.stages[1].cache_source is StageCacheSource.EXECUTED

    explicit = RecipeRunner(
        runtime=StageRuntime(random_seed=7),
    ).run_review(
        recipe,
        _review(),
        initial_artifacts={RAW_SIGNAL: _raw_signal()},
    )
    assert {item.random_seed for item in explicit.stages} == {7}

    other_definition = RecipeRunner(
        runtime=StageRuntime(
            data_revision="synthetic-v1",
            code_revision="test-v2",
        ),
    ).run_review(
        recipe,
        _review(),
        initial_artifacts={RAW_SIGNAL: _raw_signal()},
    )
    assert {
        item.random_seed for item in other_definition.stages
    } != {derived_seed}

    standalone = RecipeRunner().run_review(
        recipe,
        _review(),
        initial_artifacts={RAW_SIGNAL: _raw_signal()},
    )
    expected_standalone_seed = (
        int(standalone.recipe_digest[:16], 16) % (2**32)
    )
    assert {
        item.random_seed for item in standalone.stages
    } == {expected_standalone_seed}


def test_recipe_identity_and_source_version_are_derived_automatically():
    stage = CallableStage(
        function=lambda inputs, runtime: {
            CORE_TARGET_WEIGHTS: pd.Series(
                [1.0],
                index=pd.Index([1001], name="instrument_id"),
            )
        },
        output_specs=(ArtifactOutput(CORE_TARGET_WEIGHTS),),
    )
    recipe = IndexRecipe(nodes=(StageNode("weights", stage),))
    assert recipe.recipe_id.startswith("index_recipe_")
    assert recipe.recipe_version.startswith("auto_")

    unfingerprintable = CallableStage(
        function=len,
        output_specs=(ArtifactOutput(CORE_TARGET_WEIGHTS),),
        cache_scope=StageCacheScope.CONTENT,
    )
    with pytest.raises(RecipeCompilationError, match="source digest"):
        RecipeCompiler().compile(
            IndexRecipe(nodes=(StageNode("opaque_builtin", unfingerprintable),))
        )


def test_callable_stage_identity_includes_defaults_and_closure_state():
    def make_stage(scale):
        def calculate(inputs, runtime, *, offset=0.0):
            del inputs, runtime
            weights = pd.Series(
                [scale + offset, 1.0 - scale - offset],
                index=pd.Index(["A", "B"], name="instrument_id"),
                name="index_weight",
            )
            return {CORE_TARGET_WEIGHTS: weights}

        return CallableStage(
            function=calculate,
            output_specs=(ArtifactOutput(CORE_TARGET_WEIGHTS),),
            cache_scope=StageCacheScope.CONTENT,
            side_effect=StageSideEffect.PURE,
        )

    first = RecipeCompiler().compile(
        IndexRecipe(nodes=(StageNode("weights", make_stage(0.4)),))
    )
    second = RecipeCompiler().compile(
        IndexRecipe(nodes=(StageNode("weights", make_stage(0.6)),))
    )

    assert (
        first.nodes[0].implementation_digest
        != second.nodes[0].implementation_digest
    )
    assert first.recipe_digest != second.recipe_digest


def test_unstable_callable_state_is_allowed_only_with_stage_cache_disabled():
    unstable_value = object()

    def calculate(inputs, runtime):
        del inputs, runtime
        if unstable_value is None:
            raise AssertionError("captured value must remain reachable")
        return {
            CORE_TARGET_WEIGHTS: pd.Series(
                [1.0],
                index=pd.Index(["A"], name="instrument_id"),
                name="index_weight",
            )
        }

    cacheable_recipe = IndexRecipe(
        nodes=(
            StageNode(
                "weights",
                CallableStage(
                    function=calculate,
                    output_specs=(ArtifactOutput(CORE_TARGET_WEIGHTS),),
                    cache_scope=StageCacheScope.CONTENT,
                ),
            ),
        )
    )
    with pytest.raises(RecipeCompilationError, match="source digest"):
        RecipeCompiler().compile(
            cacheable_recipe
        )
    off_result = RecipeRunner(cache=None).run_review(
        cacheable_recipe,
        _review(),
    )
    assert off_result.target_weights.to_dict() == {"A": 1.0}

    recipe = IndexRecipe(
        nodes=(
            StageNode(
                "weights",
                CallableStage(
                    function=calculate,
                    output_specs=(ArtifactOutput(CORE_TARGET_WEIGHTS),),
                    cache_scope=StageCacheScope.DISABLED,
                ),
            ),
        )
    )
    result = RecipeRunner().run_review(recipe, _review())
    assert result.target_weights.to_dict() == {"A": 1.0}


def test_recipe_stage_identity_tracks_local_helper_source_changes(tmp_path):
    package = tmp_path.joinpath("recipe_identity_example")
    package.mkdir()
    package.joinpath("__init__.py").write_text("", encoding="utf-8")
    helper = package.joinpath("helper.py")
    helper.write_text("SCALE = 1.0\n", encoding="utf-8")
    package.joinpath("stage.py").write_text(
        "from .helper import SCALE\n"
        "from icapa.portfolio_construction import (\n"
        "    Artifact, ArtifactOutput, CORE_TARGET_WEIGHTS,\n"
        "    StageCacheScope, StageDescriptor, StageRequirements, StageResult,\n"
        ")\n"
        "import pandas as pd\n\n"
        "class HelperStage:\n"
        "    @property\n"
        "    def descriptor(self):\n"
        "        return StageDescriptor(\n"
        "            'test.helper_stage', cache_scope=StageCacheScope.CONTENT\n"
        "        )\n"
        "    @property\n"
        "    def requirements(self):\n"
        "        return StageRequirements()\n"
        "    @property\n"
        "    def outputs(self):\n"
        "        return (ArtifactOutput(CORE_TARGET_WEIGHTS),)\n"
        "    def canonical_configuration(self):\n"
        "        return {}\n"
        "    def run(self, inputs, runtime):\n"
        "        weights = pd.Series(\n"
        "            [SCALE], index=pd.Index(['A'], name='instrument_id')\n"
        "        )\n"
        "        return StageResult({\n"
        "            CORE_TARGET_WEIGHTS: Artifact.from_value(\n"
        "                CORE_TARGET_WEIGHTS, weights\n"
        "            )\n"
        "        })\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module(
            "recipe_identity_example.stage"
        )
        first = RecipeCompiler().compile(
            IndexRecipe(
                nodes=(StageNode("weights", module.HelperStage()),)
            )
        )
        helper.write_text("SCALE = 2.0\n", encoding="utf-8")
        second = RecipeCompiler().compile(
            IndexRecipe(
                nodes=(StageNode("weights", module.HelperStage()),)
            )
        )
    finally:
        sys.path.remove(str(tmp_path))
        for name in (
            "recipe_identity_example.stage",
            "recipe_identity_example.helper",
            "recipe_identity_example",
        ):
            sys.modules.pop(name, None)

    assert (
        first.nodes[0].implementation_digest
        != second.nodes[0].implementation_digest
    )
    assert first.recipe_digest != second.recipe_digest


def test_recipe_compiler_rejects_missing_producers_and_cycles():
    passthrough = CallableStage(
        function=lambda inputs, runtime: {
            CORE_TARGET_WEIGHTS: pd.Series(
                [1.0], index=pd.Index([1], name="instrument_id")
            )
        },
        output_specs=(ArtifactOutput(CORE_TARGET_WEIGHTS),),
        input_requirements=StageRequirements(
            artifacts=(ArtifactRequirement(FEATURES),)
        ),
    )
    with pytest.raises(RecipeCompilationError, match="unproduced artifact"):
        RecipeCompiler().compile(
            IndexRecipe("missing", "1", (StageNode("weights", passthrough),))
        )

    first_key = ArtifactKey("test", "first")
    second_key = ArtifactKey("test", "second")
    first = CallableStage(
        function=lambda inputs, runtime: {first_key: 1},
        output_specs=(ArtifactOutput(first_key),),
        input_requirements=StageRequirements(
            artifacts=(ArtifactRequirement(second_key),)
        ),
    )
    second = CallableStage(
        function=lambda inputs, runtime: {second_key: 1},
        output_specs=(ArtifactOutput(second_key),),
        input_requirements=StageRequirements(
            artifacts=(ArtifactRequirement(first_key),)
        ),
    )
    with pytest.raises(RecipeCompilationError, match="cycle"):
        RecipeCompiler().compile(
            IndexRecipe(
                "cycle",
                "1",
                (StageNode("first", first), StageNode("second", second)),
                final_weights=first_key,
                validate_final_weights=False,
            )
        )


@dataclass(frozen=True)
class StatefulSelectionStage:
    @property
    def descriptor(self):
        return StageDescriptor("test.stateful", cache_scope=StageCacheScope.CONTENT)

    @property
    def requirements(self):
        return StageRequirements(
            prior_artifacts=(
                PriorArtifactRequirement(
                    MEMBERSHIP_STATE,
                    PriorStatePolicy.EMPTY_INITIAL,
                ),
            )
        )

    @property
    def outputs(self):
        return (
            ArtifactOutput(MEMBERSHIP_STATE),
            ArtifactOutput(CORE_TARGET_WEIGHTS),
        )

    def canonical_configuration(self):
        return {}

    def run(self, inputs: StageInputs, runtime):
        previous = inputs.prior_artifacts.get(MEMBERSHIP_STATE)
        selected = [1001] if previous is None else [*previous.value, 1002]
        weights = pd.Series(
            1.0 / len(selected),
            index=pd.Index(selected, name="instrument_id"),
            name="index_weight",
        )
        return StageResult(
            {
                MEMBERSHIP_STATE: Artifact.from_value(
                    MEMBERSHIP_STATE,
                    selected,
                ),
                CORE_TARGET_WEIGHTS: Artifact.from_value(
                    CORE_TARGET_WEIGHTS,
                    weights,
                ),
            }
        )


@dataclass(frozen=True)
class ConsumeAllPreviousStage:
    @property
    def descriptor(self):
        return StageDescriptor(
            "test.consume_all_previous",
            cache_scope=StageCacheScope.CONTENT,
        )

    @property
    def requirements(self):
        return StageRequirements(consume_all_previous=True)

    @property
    def outputs(self):
        return (ArtifactOutput(CORE_TARGET_WEIGHTS),)

    def canonical_configuration(self):
        return {}

    def run(self, inputs: StageInputs, runtime):
        del runtime
        assert not inputs.prior_artifacts
        return StageResult(
            {
                CORE_TARGET_WEIGHTS: Artifact.from_value(
                    CORE_TARGET_WEIGHTS,
                    pd.Series(
                        [1.0],
                        index=pd.Index(
                            [1001],
                            name="instrument_id",
                        ),
                        name="index_weight",
                    ),
                )
            }
        )


def test_prior_review_state_is_explicit_and_never_silently_initialized():
    recipe = IndexRecipe(
        "stateful",
        "1",
        (StageNode("selection", StatefulSelectionStage()),),
    )
    runner = RecipeRunner(cache=MemoryStageCache())

    with pytest.raises(PriorReviewStateError):
        runner.run_review(recipe, _review(1))

    results = runner.run_sequence(
        recipe,
        [_review(2), _review(1)],
        allow_empty_initial_state=True,
    )
    assert results[_review(1).effective_date].target_weights.index.tolist() == [1001]
    assert results[_review(2).effective_date].target_weights.index.tolist() == [
        1001,
        1002,
    ]


def test_consume_all_previous_still_requires_an_explicit_initial_state_policy():
    recipe = IndexRecipe(
        "consume_all_previous",
        "1",
        (StageNode("weights", ConsumeAllPreviousStage()),),
    )
    runner = RecipeRunner()

    with pytest.raises(PriorReviewStateError):
        runner.run_review(recipe, _review(1))

    result = runner.run_review(
        recipe,
        _review(1),
        allow_empty_initial_state=True,
    )
    assert result.target_weights.to_dict() == {1001: 1.0}


class SimpleMethodology:
    def execute(self, data_context):
        frame = pd.DataFrame(
            {"index_weight": [0.4, 0.6]},
            index=pd.Index([1001, 1002], name="instrument_id"),
        )
        data_context.set_dataframe(frame)
        data_context.diagnostics["methodology"] = {"status": "ok"}
        return data_context


def test_methodology_and_recipe_weight_producer_adapters_preserve_contract():
    recipe = IndexRecipe.from_methodology(
        SimpleMethodology(),
        recipe_id="methodology_demo",
    )
    direct = RecipeRunner().run_review(recipe, _review())
    assert direct.target_weights.to_dict() == {1001: 0.4, 1002: 0.6}

    context = DataContext(
        reference_date=_review().reference_date,
        effective_date=_review().effective_date,
        index_id=_review().index_id,
    )
    result = RecipeWeightProducer(recipe).execute(context)
    assert result.cons["index_weight"].to_dict() == {1001: 0.4, 1002: 0.6}
    assert result.diagnostics["methodology"]["status"] == "ok"
    assert result.diagnostics["index_recipe"]["recipe_id"] == "methodology_demo"
