"""Structural guards for the backtesting business domain."""

from pathlib import Path

from icapa.backtesting import Backtester, IndexSimulator
from icapa.backtesting.reviews import BacktestResult
from icapa.backtesting.simulation import (
    IndexSimulationResult,
    SimulationParams,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_backtesting_uses_review_and_simulation_domains() -> None:
    assert Backtester.__module__ == "icapa.backtesting.reviews.runner"
    assert BacktestResult.__module__ == "icapa.backtesting.reviews.models"
    assert IndexSimulator.__module__ == "icapa.backtesting.simulation.engine"
    assert (
        IndexSimulationResult.__module__
        == "icapa.backtesting.simulation.models"
    )
    assert SimulationParams.__module__ == "icapa.backtesting.simulation.config"


def test_nested_legacy_backtesting_packages_do_not_exist() -> None:
    backtesting = PACKAGE_ROOT.joinpath("backtesting")
    assert backtesting.joinpath("backtester.py").is_file()
    assert not backtesting.joinpath("backtester").exists()
    assert not backtesting.joinpath("simulation_params").exists()
    assert not backtesting.joinpath("simulator").exists()
