"""Automatic, secret-safe provider lineage collected during data loading."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass
class ProvenanceRecorder:
    """Append-only provider and artifact lineage for one review calculation."""

    _records: list[Mapping[str, Any]] = field(default_factory=list, repr=False)

    def record_provider_call(self, identity: Mapping[str, Any]) -> None:
        """Record one already-sanitized automatic data identity."""

        if not isinstance(identity, Mapping):
            raise TypeError("provider identity must be a mapping")
        self._records.append(MappingProxyType(deepcopy(dict(identity))))

    @property
    def records(self) -> tuple[Mapping[str, Any], ...]:
        """Return immutable snapshots in execution order."""

        return tuple(
            MappingProxyType(deepcopy(dict(record)))
            for record in self._records
        )

    def copy(self) -> "ProvenanceRecorder":
        duplicate = ProvenanceRecorder()
        for record in self._records:
            duplicate.record_provider_call(record)
        return duplicate


__all__ = ["ProvenanceRecorder"]
