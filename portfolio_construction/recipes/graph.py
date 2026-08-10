"""Immutable recipe graphs and deterministic compilation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .artifacts import (
    ArtifactKey,
    CORE_TARGET_WEIGHTS,
    canonical_digest,
    canonicalize,
    qualified_name,
)
from .contracts import (
    IndexStage,
    PriorStatePolicy,
    RecipeCompilationError,
    StageCacheScope,
)
from .fingerprints import (
    component_tree_identity,
    component_tree_state_digest,
)


@dataclass(frozen=True)
class StageNode:
    """One named stage instance and its order-only dependencies."""

    node_id: str
    stage: IndexStage
    after: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("node_id must be a non-empty string")
        if not isinstance(self.stage, IndexStage):
            raise TypeError("stage must implement the IndexStage protocol")
        object.__setattr__(self, "after", tuple(self.after))


@dataclass(frozen=True)
class IndexRecipe:
    """Immutable construction graph ending in one canonical weight artifact."""

    recipe_id: str | None = None
    recipe_version: str | None = None
    nodes: tuple[StageNode, ...] = ()
    required_artifacts: tuple[ArtifactKey, ...] = ()
    final_weights: ArtifactKey = CORE_TARGET_WEIGHTS
    validate_final_weights: bool = True
    _implementation_state_digests: tuple[str | None, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        nodes = tuple(self.nodes)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "required_artifacts", tuple(self.required_artifacts))
        state_digests: list[str | None] = []
        for node in nodes:
            try:
                state_digests.append(
                    component_tree_state_digest(node.stage)
                )
            except (OSError, TypeError, ValueError):
                state_digests.append(None)
        object.__setattr__(
            self,
            "_implementation_state_digests",
            tuple(state_digests),
        )
        identity_payload = [
            {
                "node_id": node.node_id,
                "kind": node.stage.descriptor.kind,
                "after": sorted(node.after),
            }
            for node in nodes
        ]
        implementation_payload = [
            {
                "kind": node.stage.descriptor.kind,
                "declared_version": node.stage.descriptor.version,
                "implementation_digest": (
                    stage_implementation_digest(
                        node.stage,
                        state_digest=state_digests[position],
                    )
                    or f"unfingerprinted:{qualified_name(node.stage)}"
                ),
            }
            for position, node in enumerate(nodes)
        ]
        if self.recipe_id is None:
            identity = canonical_digest(identity_payload)[:16]
            object.__setattr__(self, "recipe_id", f"index_recipe_{identity}")
        elif not isinstance(self.recipe_id, str) or not self.recipe_id.strip():
            raise ValueError("recipe_id must be None or a non-empty string")
        if self.recipe_version is None:
            version = canonical_digest(implementation_payload)[:16]
            object.__setattr__(self, "recipe_version", f"auto_{version}")
        elif (
            not isinstance(self.recipe_version, str)
            or not self.recipe_version.strip()
        ):
            raise ValueError("recipe_version must be None or a non-empty string")

    @classmethod
    def from_methodology(
        cls,
        methodology: object,
        *,
        recipe_id: str | None = None,
        recipe_version: str | None = None,
    ) -> "IndexRecipe":
        """Wrap an ``execute(DataContext)`` methodology as one canonical stage."""

        from .adapters import MethodologyExecutionStage

        stage = MethodologyExecutionStage(methodology)
        return cls(
            recipe_id=recipe_id or qualified_name(methodology),
            recipe_version=recipe_version,
            nodes=(StageNode("methodology_execute", stage),),
        )


@dataclass(frozen=True)
class CompiledStageNode:
    """A node with all explicit and artifact-derived dependencies resolved."""

    node: StageNode
    dependencies: tuple[str, ...]
    implementation_digest: str | None


@dataclass(frozen=True)
class ExecutionPlan:
    """Validated, topologically ordered recipe plan."""

    recipe: IndexRecipe
    nodes: tuple[CompiledStageNode, ...]
    recipe_digest: str


class RecipeCompiler:
    """Validate a recipe and derive a stable execution plan."""

    def compile(
        self,
        recipe: IndexRecipe,
        *,
        allow_unfingerprintable: bool = False,
    ) -> ExecutionPlan:
        """Compile a recipe, optionally permitting OFF-only opaque stages."""

        node_ids = [node.node_id for node in recipe.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise RecipeCompilationError("recipe node IDs must be unique")
        nodes_by_id = {node.node_id: node for node in recipe.nodes}
        state_digests = {
            node.node_id: recipe._implementation_state_digests[position]
            for position, node in enumerate(recipe.nodes)
        }
        for node in recipe.nodes:
            unknown = set(node.after) - set(nodes_by_id)
            if unknown:
                raise RecipeCompilationError(
                    f"node {node.node_id!r} has unknown dependencies: "
                    f"{sorted(unknown)}"
                )

        producers: dict[ArtifactKey, str | None] = {}
        for key in recipe.required_artifacts:
            if key in producers:
                raise RecipeCompilationError(f"duplicate required artifact: {key}")
            producers[key] = None

        for node in recipe.nodes:
            descriptor = node.stage.descriptor
            implementation_digest = stage_implementation_digest(
                node.stage,
                state_digest=state_digests[node.node_id],
            )
            if (
                implementation_digest is None
                and descriptor.cache_scope is not StageCacheScope.DISABLED
                and not allow_unfingerprintable
            ):
                raise RecipeCompilationError(
                    f"cacheable node {node.node_id!r} has no stable implementation "
                    "source digest"
                )
            if (
                not descriptor.deterministic
                and descriptor.cache_scope is not StageCacheScope.DISABLED
            ):
                raise RecipeCompilationError(
                    f"node {node.node_id!r} is non-deterministic but cacheable"
                )
            output_keys = [output.key for output in node.stage.outputs]
            if len(set(output_keys)) != len(output_keys):
                raise RecipeCompilationError(
                    f"node {node.node_id!r} declares duplicate outputs"
                )
            for key in output_keys:
                if key in producers:
                    owner = producers[key] or "recipe input"
                    raise RecipeCompilationError(
                        f"artifact {key} is produced more than once; "
                        f"existing owner={owner}"
                    )
                producers[key] = node.node_id

        dependencies: dict[str, set[str]] = {
            node.node_id: set(node.after) for node in recipe.nodes
        }
        all_known_keys = set(producers)
        for node in recipe.nodes:
            requirements = node.stage.requirements
            current_keys = [item.key for item in requirements.artifacts]
            if len(set(current_keys)) != len(current_keys):
                raise RecipeCompilationError(
                    f"node {node.node_id!r} declares duplicate artifact requirements"
                )
            for requirement in requirements.artifacts:
                if requirement.key not in producers:
                    if requirement.optional:
                        continue
                    raise RecipeCompilationError(
                        f"node {node.node_id!r} requires unproduced artifact "
                        f"{requirement.key}"
                    )
                producer = producers[requirement.key]
                if producer is not None and producer != node.node_id:
                    dependencies[node.node_id].add(producer)
            prior_keys = [item.key for item in requirements.prior_artifacts]
            if len(set(prior_keys)) != len(prior_keys):
                raise RecipeCompilationError(
                    f"node {node.node_id!r} declares duplicate prior artifacts"
                )
            for requirement in requirements.prior_artifacts:
                if (
                    requirement.key not in all_known_keys
                    and requirement.policy is not PriorStatePolicy.OPTIONAL
                ):
                    raise RecipeCompilationError(
                        f"node {node.node_id!r} requires unknown prior artifact "
                        f"{requirement.key}"
                    )

        if recipe.final_weights not in producers:
            raise RecipeCompilationError(
                f"recipe does not provide final weight artifact {recipe.final_weights}"
            )

        ordered_ids = self._topological_order(node_ids, dependencies)
        compiled = tuple(
            CompiledStageNode(
                node=nodes_by_id[node_id],
                dependencies=tuple(sorted(dependencies[node_id])),
                implementation_digest=stage_implementation_digest(
                    nodes_by_id[node_id].stage,
                    state_digest=state_digests[node_id],
                ),
            )
            for node_id in ordered_ids
        )
        digest = canonical_digest(
            {
                "recipe_id": recipe.recipe_id,
                "recipe_version": recipe.recipe_version,
                "required_artifacts": [
                    key.canonical_name for key in recipe.required_artifacts
                ],
                "final_weights": recipe.final_weights.canonical_name,
                "validate_final_weights": recipe.validate_final_weights,
                "nodes": [
                    {
                        "node_id": item.node.node_id,
                        "dependencies": list(item.dependencies),
                        "descriptor": canonicalize(item.node.stage.descriptor),
                        "implementation_digest": item.implementation_digest,
                        "requirements": canonicalize(
                            item.node.stage.requirements
                        ),
                        "outputs": canonicalize(item.node.stage.outputs),
                        "configuration": (
                            item.node.stage.canonical_configuration()
                        ),
                    }
                    for item in compiled
                ],
            }
        )
        return ExecutionPlan(recipe=recipe, nodes=compiled, recipe_digest=digest)

    @staticmethod
    def _topological_order(
        preferred_order: Sequence[str],
        dependencies: Mapping[str, set[str]],
    ) -> list[str]:
        remaining = {name: set(values) for name, values in dependencies.items()}
        ordered: list[str] = []
        while remaining:
            ready = [
                name
                for name in preferred_order
                if name in remaining and not remaining[name]
            ]
            if not ready:
                raise RecipeCompilationError("recipe dependencies contain a cycle")
            for name in ready:
                ordered.append(name)
                remaining.pop(name)
                for values in remaining.values():
                    values.discard(name)
        return ordered


def stage_implementation_digest(
    stage: IndexStage,
    *,
    state_digest: str | None = None,
) -> str | None:
    """Hash stage implementation source without researcher-maintained versions."""

    try:
        automatic_component_identity = component_tree_identity(
            stage,
            state_digest=state_digest,
        )
    except (OSError, TypeError, ValueError):
        return None
    wrapped_identity = None
    wrapped_identity_collector = getattr(
        stage,
        "wrapped_implementation_identity",
        None,
    )
    if callable(wrapped_identity_collector):
        try:
            wrapped_identity = wrapped_identity_collector()
        except (OSError, TypeError, ValueError):
            return None

    try:
        return canonical_digest(
            {
                "automatic_component_tree": automatic_component_identity,
                "wrapped_implementation": wrapped_identity,
            }
        )
    except (OSError, TypeError, ValueError):
        return None


__all__ = [
    "CompiledStageNode",
    "ExecutionPlan",
    "IndexRecipe",
    "RecipeCompiler",
    "StageNode",
]
