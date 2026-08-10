"""Explicit, customer-neutral index-research catalog."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """A named factory for producing a fresh research specification."""

    name: str
    factory: Callable[..., object]
    description: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("catalog entry name must be a non-empty string")
        if not callable(self.factory):
            raise TypeError("catalog entry factory must be callable")


class ResearchCatalog:
    """Register explicit spec factories without import-time global state."""

    def __init__(self) -> None:
        self._entries: dict[str, CatalogEntry] = {}

    def register(self, entry: CatalogEntry, *, replace: bool = False) -> None:
        if not isinstance(entry, CatalogEntry):
            raise TypeError("entry must be a CatalogEntry")
        key = _key(entry.name)
        if key in self._entries and not replace:
            raise KeyError(f"catalog entry is already registered: {entry.name}")
        self._entries[key] = entry

    def create(self, name: str, **parameters: Any) -> object:
        """Build a new specification from one registered factory."""

        try:
            entry = self._entries[_key(name)]
        except KeyError as error:
            raise KeyError(f"catalog entry is not registered: {name}") from error
        return entry.factory(**parameters)

    def get(self, name: str) -> CatalogEntry:
        try:
            return self._entries[_key(name)]
        except KeyError as error:
            raise KeyError(f"catalog entry is not registered: {name}") from error

    def unregister(self, name: str) -> None:
        self._entries.pop(_key(name), None)

    def __iter__(self) -> Iterator[CatalogEntry]:
        for key in sorted(self._entries):
            yield self._entries[key]


def _key(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("catalog names must be non-empty strings")
    return value.strip().casefold()


__all__ = ["CatalogEntry", "ResearchCatalog"]
