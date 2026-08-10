from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from icapa.research.catalog import CatalogEntry, ResearchCatalog
from icapa.research.notebook import plot_index_levels
from icapa.research.runners import BatchItem, run_batch
from icapa.research.scenarios import (
    Scenario,
    ScenarioShock,
    ShockOperation,
    apply_scenario,
)
from icapa.research.sensitivity import NoiseSpec, add_noise, bootstrap_rows


def test_scenario_applies_ordered_provider_neutral_shocks() -> None:
    frame = pd.DataFrame(
        {
            "instrument_id": ["A", "B", "C"],
            "country": ["US", "US", "CA"],
            "signal": [1.0, 2.0, 3.0],
        }
    )
    scenario = Scenario(
        "field shock",
        (
            ScenarioShock(
                "signal",
                ShockOperation.MULTIPLY,
                2.0,
                where={"country": ("US",)},
            ),
            ScenarioShock(
                "signal",
                ShockOperation.ADD,
                1.0,
                instrument_ids=("B",),
            ),
        ),
    )

    result = apply_scenario(frame, scenario)

    assert result.frame["signal"].tolist() == [2.0, 5.0, 3.0]
    assert result.diagnostics["selected_count"].tolist() == [2, 1]
    assert frame["signal"].tolist() == [1.0, 2.0, 3.0]


def test_noise_and_bootstrap_are_reproducible() -> None:
    frame = pd.DataFrame({"signal": [1.0, 2.0, 3.0]})
    spec = NoiseSpec(columns=("signal",), scale=0.1, seed=17)

    pdt.assert_frame_equal(add_noise(frame, spec), add_noise(frame, spec))
    pdt.assert_frame_equal(
        bootstrap_rows(frame, seed=9),
        bootstrap_rows(frame, seed=9),
    )


def test_catalog_and_batch_runner_have_no_global_state() -> None:
    catalog = ResearchCatalog()
    catalog.register(CatalogEntry("demo", lambda value: {"value": value}))
    assert catalog.create("DEMO", value=3) == {"value": 3}

    class Workspace:
        def run(self, spec):
            return spec["value"] * 2

    outcomes = run_batch(
        Workspace(),
        (
            BatchItem("first", catalog.create("demo", value=2)),
            BatchItem("second", catalog.create("demo", value=4)),
        ),
    )
    assert [outcome.result for outcome in outcomes] == [4, 8]


def test_notebook_plot_uses_stored_daily_levels() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    daily = pd.DataFrame(
        {
            "index_net_total_level": [100.0, 101.0],
            "benchmark_net_total_level": [100.0, 100.5],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
    )

    axis = plot_index_levels(daily)

    assert len(axis.lines) == 2
    assert axis.get_xlabel() == "Business date"
