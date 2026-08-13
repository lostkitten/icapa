"""Smoke tests for the intentionally small public package boundary."""

import ast
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import icapa
from icapa import portfolio_construction
from icapa.portfolio_construction import IndexRecipe, OptimizationProblem
from icapa.portfolio_construction import engines, methodologies


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ONLY_PACKAGE_PREFIXES = {
    "icapa.portfolio_construction.rules.data_processing",
}
PROTECTED_IMPLEMENTATION_PACKAGE_PREFIXES = {
    "icapa.portfolio_construction.engines",
    "icapa.portfolio_construction.methodologies",
}
PUBLIC_IMPLEMENTATION_MODULES = {
    "icapa.portfolio_construction.engines.__init__",
    "icapa.portfolio_construction.engines.entropy_exposure_engine",
    "icapa.portfolio_construction.methodologies.__init__",
    (
        "icapa.portfolio_construction.methodologies."
        "entropy_exposure_methodology"
    ),
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
        "EGMUConstrainedElasticSolver",
        "EGMUElasticSolver",
        "EGMUNewtonSolver",
        "EGMUProjectionResult",
        "EGMUProjectionSolver",
        "EGMUResult",
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
        "egmu_elastic",
        "egmu_newton",
        "egmu_project_linear",
        "egmu_project_elastic",
        "estimate_covariance_for_window",
        "evaluate_problem_constraints",
        "relative_entropy",
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


def test_entropy_exposure_subpackages_have_an_explicit_public_surface():
    assert set(engines.__all__) == {
        "EntropyExposureEngine",
        "EntropyExposureMode",
        "ExposureTarget",
        "TargetDirection",
    }
    assert set(methodologies.__all__) == {"EntropyExposureMethodology"}


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
                    "icapa."
                    + ".".join(
                        initializer.parent.relative_to(PROJECT_ROOT).parts
                    )
                )
    expected.difference_update(LOCAL_ONLY_PACKAGE_PREFIXES)

    assert configured == expected
    assert not configured.intersection(LOCAL_ONLY_PACKAGE_PREFIXES)


def test_build_and_source_manifest_whitelist_public_implementations():
    setup_prefixes = set(
        _assigned_literal(PROJECT_ROOT / "setup.py", "_LOCAL_ONLY_PACKAGE_PREFIXES")
    )
    assert setup_prefixes == LOCAL_ONLY_PACKAGE_PREFIXES
    protected_prefixes = set(
        _assigned_literal(
            PROJECT_ROOT / "setup.py",
            "_PROTECTED_IMPLEMENTATION_PACKAGE_PREFIXES",
        )
    )
    assert protected_prefixes == PROTECTED_IMPLEMENTATION_PACKAGE_PREFIXES
    public_modules = set(
        _assigned_literal(
            PROJECT_ROOT / "setup.py",
            "_PUBLIC_IMPLEMENTATION_MODULES",
        )
    )
    assert public_modules == PUBLIC_IMPLEMENTATION_MODULES

    manifest = (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    for package_name in PROTECTED_IMPLEMENTATION_PACKAGE_PREFIXES:
        relative_path = package_name.removeprefix("icapa.").replace(".", "/")
        assert f"recursive-exclude {relative_path} *" in manifest
    for module_name in PUBLIC_IMPLEMENTATION_MODULES:
        relative_path = module_name.removeprefix("icapa.").replace(".", "/")
        if relative_path.endswith("/__init__"):
            relative_path += ".py"
        else:
            relative_path += ".py"
        assert f"include {relative_path}" in manifest


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
    public_implementations = set(
        _assigned_literal(
            PROJECT_ROOT / "scripts" / "repository_guard.py",
            "PUBLIC_PORTFOLIO_IMPLEMENTATION_FILES",
        )
    )
    assert public_implementations == {
        "portfolio_construction/engines/__init__.py",
        "portfolio_construction/engines/entropy_exposure_engine.py",
        "portfolio_construction/methodologies/__init__.py",
        (
            "portfolio_construction/methodologies/"
            "entropy_exposure_methodology.py"
        ),
    }


def test_gitignore_keeps_private_code_and_unlisted_adapters_local():
    ignored = (
        "portfolio_construction/engines/example_engine.py",
        "portfolio_construction/methodologies/example_methodology.py",
        "portfolio_construction/rules/data_processing/example_rule.py",
        "data_sources/providers/example_local_adapter.py",
    )
    public = (
        "portfolio_construction/engines/.gitkeep",
        "portfolio_construction/engines/__init__.py",
        "portfolio_construction/engines/entropy_exposure_engine.py",
        "portfolio_construction/methodologies/__init__.py",
        (
            "portfolio_construction/methodologies/"
            "entropy_exposure_methodology.py"
        ),
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
    for package_name in LOCAL_ONLY_PACKAGE_PREFIXES:
        relative = package_name.removeprefix("icapa.").split(".")
        assert not build_lib.joinpath("icapa", *relative).exists()
    public_package_files = {
        "engines": {"__init__.py", "entropy_exposure_engine.py"},
        "methodologies": {
            "__init__.py",
            "entropy_exposure_methodology.py",
        },
    }
    for package_name, expected_files in public_package_files.items():
        package_path = build_lib / "icapa" / "portfolio_construction" / package_name
        assert {
            path.name for path in package_path.glob("*.py")
        } == expected_files
    imported = subprocess.run(
        (
            sys.executable,
            "-c",
            (
                "from icapa.portfolio_construction.engines import "
                "EntropyExposureEngine; "
                "from icapa.portfolio_construction.methodologies import "
                "EntropyExposureMethodology; "
                "assert EntropyExposureEngine and EntropyExposureMethodology; "
                "assert EntropyExposureMethodology("
                "targets=[], universe_id='u', universe_provider_name='p'"
                ").to_recipe()"
            ),
        ),
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(build_lib)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0
