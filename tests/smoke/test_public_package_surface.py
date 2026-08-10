"""Smoke tests for the intentionally small public package boundary."""

import ast
from pathlib import Path
import subprocess
import sys
import tomllib

import icapa
from icapa import portfolio_construction
from icapa.portfolio_construction import IndexRecipe, OptimizationProblem


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_PACKAGE_PREFIXES = {
    "icapa.portfolio_construction.engines",
    "icapa.portfolio_construction.methodologies",
    "icapa.portfolio_construction.rules.data_processing",
}
PUBLIC_DOMAIN_DIRECTORIES = (
    "analytics",
    "backtesting",
    "data_sources",
    "portfolio_construction",
    "reporting",
    "research",
    "workspace",
)


def _assigned_literal(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == name
            for target in targets
        ):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "frozenset"
        ):
            value = value.args[0]
        return ast.literal_eval(value)
    raise AssertionError(f"{name} is not assigned in {path}")


def test_public_portfolio_construction_surface_is_explicit_and_client_neutral():
    assert set(portfolio_construction.__all__) == {
        "Artifact",
        "ArtifactKey",
        "ArtifactOutput",
        "ArtifactRequirement",
        "CORE_CONSTITUENTS",
        "CORE_DAILY_DATA",
        "CORE_DIAGNOSTICS",
        "CORE_FINAL_CONSTITUENTS",
        "CORE_TARGET_WEIGHTS",
        "CallableObjectiveSpec",
        "CallableStage",
        "ConstraintEvaluation",
        "ConstraintReport",
        "ConstraintStatus",
        "CovarianceEstimate",
        "CovarianceEstimator",
        "CovarianceMissingDataPolicy",
        "CovarianceShrinkageTarget",
        "DataContainer",
        "DataContext",
        "ExecutionPlan",
        "FeasibilityReport",
        "FeasibilityStatus",
        "FieldExposureConstraintSpec",
        "FactorCovarianceEstimator",
        "FactorStandardizationVariant",
        "GroupWeightConstraintSpec",
        "IndexRecipe",
        "IndexStage",
        "LinearConstraintSpec",
        "LinearObjectiveSpec",
        "LiquidityConstraintSpec",
        "MemoryStageCache",
        "MethodologyExecutionStage",
        "MinimumVarianceObjectiveSpec",
        "ModelSolver",
        "ModelSolveResult",
        "NonlinearConstraintSpec",
        "OSQPBackend",
        "ObjectiveSpec",
        "OptionalSolverDependencyError",
        "OptimizationError",
        "OptimizationModelSpec",
        "OptimizationProblem",
        "OptimizationResult",
        "PortfolioSolver",
        "PortfolioSolverModelAdapter",
        "PortfolioModelSpec",
        "PreviousReviewState",
        "PriorArtifactRequirement",
        "PriorReviewStateError",
        "PriorStatePolicy",
        "ProviderRequestSpec",
        "RecipeCompilationError",
        "RecipeCompiler",
        "RecipeError",
        "RecipeRunner",
        "RecipeWeightProducer",
        "ResolvedReturnWindow",
        "ReturnWindowKind",
        "ReturnWindowSpec",
        "ReviewConstructionResult",
        "ReviewIdentity",
        "RuleExecutionStage",
        "SampleCovarianceEstimator",
        "ShrinkageCovarianceEstimator",
        "ScipySLSQPSolver",
        "SolverCapability",
        "SolverRouter",
        "SquaredDistanceObjectiveSpec",
        "StageCache",
        "StageCacheScope",
        "StageCacheSource",
        "StageDescriptor",
        "StageDiagnostic",
        "StageExecutionError",
        "StageExecutionRecord",
        "StageInputs",
        "StageNode",
        "StageRegistry",
        "StageRequirements",
        "StageResult",
        "StageRuntime",
        "StageSideEffect",
        "StandardizeFactors",
        "WeightVariableSpec",
        "check_linear_feasibility",
        "estimate_covariance_for_window",
        "evaluate_problem_constraints",
        "factor_output_name",
        "get_constituents",
        "standardize_factors",
    }
    assert OptimizationProblem is portfolio_construction.OptimizationProblem
    assert IndexRecipe is portfolio_construction.IndexRecipe
    assert not any(
        name.endswith(("Methodology", "Engine"))
        for name in portfolio_construction.__all__
    )


def test_root_package_exposes_only_the_research_facade():
    assert set(icapa.__all__) == {
        "IndexRecipe",
        "ResearchRun",
        "ResearchSpec",
        "ResearchWorkspace",
    }
    assert not any(name.endswith(("Methodology", "Engine")) for name in icapa.__all__)
    assert not hasattr(icapa.ResearchWorkspace, "open_workspace")


def test_setuptools_explicitly_packages_every_public_domain_subpackage():
    configuration = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    configured = set(configuration["tool"]["setuptools"]["packages"])
    expected = {"icapa"}
    for directory in PUBLIC_DOMAIN_DIRECTORIES:
        for initializer in (PROJECT_ROOT / directory).rglob("__init__.py"):
            if "__pycache__" not in initializer.parts:
                expected.add(
                    "icapa." + ".".join(initializer.parent.relative_to(PROJECT_ROOT).parts)
                )
    expected.difference_update(PRIVATE_PACKAGE_PREFIXES)

    assert configured == expected
    assert not configured.intersection(PRIVATE_PACKAGE_PREFIXES)


def test_build_and_source_manifest_exclude_private_implementations():
    setup_prefixes = set(
        _assigned_literal(PROJECT_ROOT / "setup.py", "_LOCAL_ONLY_PACKAGE_PREFIXES")
    )
    assert setup_prefixes == PRIVATE_PACKAGE_PREFIXES

    manifest = (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    for package_name in PRIVATE_PACKAGE_PREFIXES:
        relative_path = package_name.removeprefix("icapa.").replace(".", "/")
        assert f"prune {relative_path}" in manifest


def test_repository_guard_uses_the_refactored_data_source_paths():
    public_files = set(
        _assigned_literal(
            PROJECT_ROOT / "scripts" / "repository_guard.py",
            "PUBLIC_DATA_SOURCE_FILES",
        )
    )
    assert {
        "data_sources/providers/registry.py",
        "data_sources/providers/factset/adapter.py",
        "data_sources/providers/snowflake/adapter.py",
        "data_sources/provenance/identity.py",
        "data_sources/services/history.py",
        "data_sources/universes/mapping.py",
    }.issubset(public_files)
    assert not any(
        path.startswith(
            (
                "data_sources/factset/",
                "data_sources/snowflake/",
            )
        )
        for path in public_files
    )


def test_gitignore_keeps_private_code_and_unlisted_adapters_local():
    ignored = (
        "portfolio_construction/engines/example_engine.py",
        "portfolio_construction/methodologies/example_methodology.py",
        "portfolio_construction/rules/data_processing/example_rule.py",
        "data_sources/providers/example_local_adapter.py",
    )
    public = (
        "portfolio_construction/engines/.gitkeep",
        "data_sources/providers/factset/adapter.py",
        "data_sources/providers/snowflake/adapter.py",
    )
    for path in ignored:
        result = subprocess.run(
            ("git", "check-ignore", "--quiet", "--no-index", path),
            cwd=PROJECT_ROOT,
            check=False,
        )
        assert result.returncode == 0, path
    for path in public:
        result = subprocess.run(
            ("git", "check-ignore", "--quiet", "--no-index", path),
            cwd=PROJECT_ROOT,
            check=False,
        )
        assert result.returncode == 1, path


def test_build_hook_removes_stale_package_output(tmp_path):
    build_lib = tmp_path / "build"
    stale = (
        build_lib
        / "icapa"
        / "analytics"
        / "comparisons"
        / "obsolete.py"
    )
    stale.parent.mkdir(parents=True)
    stale.write_text("raise RuntimeError('stale build output')\n", encoding="utf-8")

    subprocess.run(
        (
            sys.executable,
            "setup.py",
            "egg_info",
            "--egg-base",
            str(tmp_path),
            "build_py",
            "--build-lib",
            str(build_lib),
        ),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert not stale.exists()
    assert build_lib.joinpath("icapa", "analytics", "comparison", "engine.py").is_file()
    for package_name in PRIVATE_PACKAGE_PREFIXES:
        relative = package_name.removeprefix("icapa.").split(".")
        assert not build_lib.joinpath("icapa", *relative).exists()
