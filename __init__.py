"""ICAPA public package entry point.

Importing the package is intentionally side-effect free: it does not connect
to a database, write usage logs, open release notes, or select external data.
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .analytics import AnalyticsEngine, AnalyticsResult, BrinsonInput, analyze_backtest
from .data_sources import (
    FactSet,
    FileProvider,
    SnowflakePlaceholder,
    get_provider,
    register_provider,
    registry,
)
from .backtesting import (
    Backtester,
    Calendar,
    DividendTreatment,
    IndexSimulationResult,
    IndexSimulator,
    RebalanceTiming,
    SimulationParams,
    WeightDrift,
)
from .helpers import (
    UnderlyingMappingRegistry,
    UnderlyingProfile,
)
from .portfolio_construction import (
    LinearConstraintSpec,
    NonlinearConstraintSpec,
    OptimizationProblem,
    PortfolioSolver,
    ScipySLSQPSolver,
)
from .portfolio_construction.rules.data_loading import AddThirdPartyData
from .reporting import ReportPayload, write_index_research_report
from .tools.enums import DataSource, ThirdPartyDataType
from .workspace import CachePolicy, WorkspaceStore

project_root = Path(__file__).parent

try:
    __version__ = version("icapa")
except PackageNotFoundError:
    __version__ = "0+local"

__all__ = [
    "FactSet",
    "FileProvider",
    "SnowflakePlaceholder",
    "get_provider",
    "project_root",
    "register_provider",
    "registry",
    "AddThirdPartyData",
    "AnalyticsEngine",
    "AnalyticsResult",
    "Backtester",
    "BrinsonInput",
    "Calendar",
    "DataSource",
    "DividendTreatment",
    "IndexSimulationResult",
    "IndexSimulator",
    "LinearConstraintSpec",
    "NonlinearConstraintSpec",
    "OptimizationProblem",
    "PortfolioSolver",
    "RebalanceTiming",
    "ReportPayload",
    "ScipySLSQPSolver",
    "SimulationParams",
    "ThirdPartyDataType",
    "UnderlyingMappingRegistry",
    "UnderlyingProfile",
    "WeightDrift",
    "WorkspaceStore",
    "CachePolicy",
    "analyze_backtest",
    "write_index_research_report",
]
