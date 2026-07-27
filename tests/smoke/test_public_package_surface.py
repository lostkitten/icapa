"""Smoke tests for the intentionally small public package boundary."""

import icapa
from icapa import portfolio_construction
from icapa.portfolio_construction import OptimizationProblem


def test_public_portfolio_construction_surface_is_solver_only():
    assert set(portfolio_construction.__all__) == {
        "LinearConstraintSpec",
        "NonlinearConstraintSpec",
        "OptimizationError",
        "OptimizationProblem",
        "OptimizationResult",
        "PortfolioSolver",
        "ScipySLSQPSolver",
    }
    assert OptimizationProblem is portfolio_construction.OptimizationProblem


def test_root_package_exposes_public_pipeline_components():
    expected = {
        "AnalyticsEngine",
        "Backtester",
        "FileProvider",
        "IndexSimulator",
        "OptimizationProblem",
        "ReportPayload",
        "WorkspaceStore",
    }
    assert expected.issubset(set(icapa.__all__))
