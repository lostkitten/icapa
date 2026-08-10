"""Opt-in controls and metric recording for the independent scale suite."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import pytest


def pytest_addoption(parser) -> None:
    group = parser.getgroup("icapa-scale")
    group.addoption(
        "--run-scale",
        action="store_true",
        default=False,
        help="Run the opt-in ICAPA scale and performance suite.",
    )
    group.addoption(
        "--scale-profile",
        choices=("quick", "full"),
        default="quick",
        help=(
            "Select quick harness validation or the approved full "
            "5,000/10,000-instrument matrix."
        ),
    )
    group.addoption(
        "--scale-output",
        default=None,
        metavar="PATH",
        help="Append one canonical JSON metric record per benchmark to PATH.",
    )


def pytest_collection_modifyitems(config, items) -> None:
    if config.getoption("--run-scale"):
        return
    skip = pytest.mark.skip(
        reason="scale suite is opt-in; pass --run-scale to execute it"
    )
    for item in items:
        if item.get_closest_marker("scale") is not None:
            item.add_marker(skip)


@pytest.fixture
def scale_profile(pytestconfig) -> str:
    """Return the explicitly selected benchmark profile."""

    return str(pytestconfig.getoption("--scale-profile"))


@pytest.fixture
def record_scale_metric(pytestconfig, record_property):
    """Return a recorder that prints and optionally persists JSON Lines."""

    output = pytestconfig.getoption("--scale-output")
    output_path = None if output is None else Path(output).expanduser().resolve()

    def record(benchmark: str, **values: Any) -> dict[str, Any]:
        payload = {
            "benchmark": benchmark,
            "profile": str(pytestconfig.getoption("--scale-profile")),
            "process_id": os.getpid(),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **values,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        print(f"ICAPA_SCALE_METRIC {encoded}")
        record_property(f"icapa_scale_{benchmark}", encoded)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("a", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.write("\n")
        return payload

    return record
