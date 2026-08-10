"""Effective-date target-weight generation for research runs."""

from __future__ import annotations

from typing import Any

from ...backtesting import Backtester
from ...data_sources.providers.registry import registry
from ...portfolio_construction import (
    RecipeCompiler,
    RecipeWeightProducer,
    StageCacheScope,
    StageRuntime,
)
from ...workspace import (
    CacheMode,
    CachePolicy,
    ParquetStageStore,
    automatic_component_identity,
    automatic_digest,
    automatic_provider_identity,
)
from ...workspace.caches.recipe import WorkspaceStageCache
from ...workspace.caches.source import private_parameter_scope_digest
from . import identity as _identity
from .contracts import _CacheDecision, _PerReviewParquetStageStore
from .providers import _recipe_is_stateful
from ..models import ResearchSpec, UnsafeCacheReuseError


class _ReviewRunner:
    """Build the review-weight producer for one research request."""

    def _backtester(
        self,
        spec: ResearchSpec,
        decision: _CacheDecision,
        construction_identity: str,
        definition_fingerprint: str,
    ) -> Backtester:
        kwargs: dict[str, Any] = {}
        stateful_recipe = _recipe_is_stateful(spec)
        review_cache_identity = (
            automatic_digest(
                {
                    "construction_identity": construction_identity,
                    "explicit_random_seed": spec.random_seed,
                }
            )
            if (spec.definition.recipe is not None and spec.random_seed is not None)
            else construction_identity
        )
        if (
            decision.actual is not CacheMode.OFF
            and not stateful_recipe
            and (spec.definition.recipe is None or decision.snapshot_digest is not None)
        ):
            stage_store = (
                _PerReviewParquetStageStore(
                    self._workspace,
                    index_id=spec.definition.index_id,
                    construction_identity=review_cache_identity,
                    evidence=decision.review_snapshot,
                )
                if decision.review_snapshot is not None
                else ParquetStageStore(
                    self._workspace,
                    index_id=spec.definition.index_id,
                    namespace_digest=automatic_digest(
                        {
                            "kind": "review_cache",
                            "construction_identity": review_cache_identity,
                            "snapshot_digest": decision.snapshot_digest,
                        }
                    ),
                )
            )
            kwargs = {
                "workspace_name": self.workspace_name,
                "cache_policy": CachePolicy(decision.actual.value),
                "data_revision": {
                    "automatic_snapshot_digest": decision.snapshot_digest,
                },
                "cache_configuration": {
                    "construction_identity": review_cache_identity,
                },
                "cache_store": stage_store,
            }
        methodology = spec.definition.methodology
        if spec.definition.recipe is not None:
            recipe_kernel_identity = automatic_digest(
                {
                    "backtester": automatic_component_identity(Backtester),
                    "recipe_weight_producer": (
                        automatic_component_identity(RecipeWeightProducer)
                    ),
                    "recipe_compiler": automatic_component_identity(RecipeCompiler),
                    "runtime": _identity.automatic_runtime_identity(),
                }
            )
            if decision.actual is CacheMode.READ_ONLY and any(
                node.stage.descriptor.cache_scope is StageCacheScope.DISABLED
                for node in spec.definition.recipe.nodes
            ):
                raise UnsafeCacheReuseError(
                    "READ_ONLY recipe execution requires every stage to be cacheable"
                )
            stage_cache = (
                None
                if decision.actual is CacheMode.OFF
                else WorkspaceStageCache(
                    self._workspace,
                    mode=decision.actual,
                )
            )
            providers = {
                capability: registry.resolve(
                    capability,
                    binding.provider_name,
                )
                for capability, binding in spec.recipe_providers.items()
            }
            provider_parameters = {
                capability: dict(binding.parameters)
                for capability, binding in spec.recipe_providers.items()
            }
            provider_components = {
                capability: automatic_digest(
                    automatic_provider_identity(
                        binding.provider_name,
                        providers[capability],
                        capability=capability,
                        parameters=binding.parameters,
                    )
                )
                for capability, binding in spec.recipe_providers.items()
            }

            def runtime_for_review(review):
                data_revision = (
                    decision.review_snapshot.digest_for(
                        review.reference_date,
                        review.effective_date,
                    )
                    if decision.review_snapshot is not None
                    else (decision.snapshot_digest or "unversioned")
                )
                return StageRuntime(
                    providers=providers,
                    provider_parameters=provider_parameters,
                    provider_revisions={
                        capability: {
                            "component_digest": component_digest,
                            "snapshot_digest": data_revision,
                            "parameter_scope_digest": (
                                private_parameter_scope_digest(
                                    spec.recipe_providers[capability].parameters
                                )
                            ),
                        }
                        for capability, component_digest in provider_components.items()
                    },
                    data_revision=data_revision,
                    code_revision=recipe_kernel_identity,
                    random_seed_revision=definition_fingerprint,
                    random_seed=spec.random_seed,
                )

            methodology = RecipeWeightProducer(
                spec.definition.recipe,
                cache=stage_cache,
                runtime=StageRuntime(
                    providers=providers,
                    provider_parameters=provider_parameters,
                    provider_revisions={
                        capability: {
                            "component_digest": component_digest,
                            "parameter_scope_digest": (
                                private_parameter_scope_digest(
                                    spec.recipe_providers[capability].parameters
                                )
                            ),
                        }
                        for capability, component_digest in provider_components.items()
                    },
                    data_revision=(decision.snapshot_digest or "unversioned"),
                    code_revision=recipe_kernel_identity,
                    random_seed_revision=definition_fingerprint,
                    random_seed=spec.random_seed,
                ),
                runtime_provider=runtime_for_review,
                allow_empty_initial_state=(spec.allow_empty_recipe_initial_state),
                memory_cache_when_unspecified=(decision.actual is not CacheMode.OFF),
            )
        return Backtester(
            index_id=spec.definition.index_id,
            calendar=spec.calendar,
            methodology=methodology,
            **kwargs,
        )


__all__ = ["_ReviewRunner"]
