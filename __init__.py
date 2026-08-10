"""Minimal, side-effect-free public entry point for ICAPA."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any


try:
    __version__ = version("icapa")
except PackageNotFoundError:
    __version__ = "0+local"


_LAZY_EXPORTS = {
    "IndexRecipe": ("icapa.portfolio_construction", "IndexRecipe"),
    "ResearchRun": ("icapa.research", "ResearchRun"),
    "ResearchSpec": ("icapa.research", "ResearchSpec"),
    "ResearchWorkspace": ("icapa.research", "ResearchWorkspace"),
}

__all__ = [
    "IndexRecipe",
    "ResearchRun",
    "ResearchSpec",
    "ResearchWorkspace",
]


def __getattr__(name: str) -> Any:
    """Load the deliberately small public facade only when requested."""

    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module 'icapa' has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
