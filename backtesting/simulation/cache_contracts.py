"""Storage and identity contracts for optional simulation reuse."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum, StrEnum
from hashlib import sha256
from importlib import metadata
import inspect
import json
import math
from pathlib import Path
import platform
from typing import Any, Protocol

import numpy as np
import pandas as pd


class SimulationCachePolicy(StrEnum):
    """Control reading and writing of reusable simulation artifacts."""

    REUSE = "reuse"
    REFRESH = "refresh"
    READ_ONLY = "read_only"


class SimulationCacheMissError(RuntimeError):
    """Raised when a required simulation artifact is unavailable."""


class SimulationCacheStore(Protocol):
    """Physical persistence operations required by simulation cache controllers."""

    def load_frame(self, stage: str, key: str, name: str) -> pd.DataFrame: ...

    def save_frame(
        self,
        stage: str,
        key: str,
        name: str,
        frame: pd.DataFrame,
    ) -> object: ...

    def load_json(self, stage: str, key: str, name: str) -> Any: ...

    def save_json(self, stage: str, key: str, name: str, value: Any) -> object: ...

    def simulation_catalog_lock(
        self,
        namespace: str,
    ) -> AbstractContextManager[None]: ...


class SimulationIdentityService(Protocol):
    """Automatic identity operations needed by simulation cache keys."""

    def digest(self, value: Any) -> str: ...

    def safe_parameters(
        self,
        parameters: Mapping[str, Any] | None,
    ) -> dict[str, Any]: ...

    def source_identity(self, component: object) -> dict[str, Any]: ...

    def runtime_identity(self) -> tuple[dict[str, Any], ...]: ...

    def dataframe_digest(
        self,
        frame: pd.DataFrame,
        *,
        sort_by: Sequence[str] | None = None,
    ) -> str: ...


class DefaultSimulationIdentityService:
    """Deterministic, credential-safe identity implementation for local runs."""

    def digest(self, value: Any) -> str:
        encoded = json.dumps(
            _canonicalize(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def safe_parameters(
        self,
        parameters: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        values = dict(parameters or {})
        visible = {
            str(key): _canonicalize(value)
            for key, value in sorted(values.items(), key=lambda item: str(item[0]))
            if not _is_secret_key(str(key))
        }
        redacted = sorted(
            str(key) for key in values if _is_secret_key(str(key))
        )
        return {
            "keys": sorted(map(str, values)),
            "redacted_keys": redacted,
            "semantic_digest": self.digest(visible),
        }

    def source_identity(self, component: object) -> dict[str, Any]:
        target = (
            component
            if inspect.isclass(component)
            or inspect.isfunction(component)
            or inspect.ismethod(component)
            else type(component)
        )
        source = inspect.getsourcefile(target)
        if source is None:
            raise ValueError(
                f"cannot identify source for {target!r}; disable cache reuse"
            )
        source_path = Path(source).resolve()
        paths = [source_path]
        if "backtesting/simulation" in source_path.as_posix():
            paths = sorted(source_path.parent.glob("*.py"))
        closure = sha256()
        for path in paths:
            closure.update(path.name.encode("utf-8"))
            closure.update(path.read_bytes())
        return {
            "type": (
                f"{getattr(target, '__module__', '')}."
                f"{getattr(target, '__qualname__', getattr(target, '__name__', ''))}"
            ).strip("."),
            "source_digest": sha256(source_path.read_bytes()).hexdigest(),
            "source_closure_digest": closure.hexdigest(),
            "source_file_count": len(paths),
        }

    def runtime_identity(self) -> tuple[dict[str, Any], ...]:
        records: list[dict[str, Any]] = [
            {
                "name": "python",
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
            }
        ]
        for package_name in (
            "icapa",
            "numpy",
            "pandas",
            "scipy",
            "pyarrow",
            "osqp",
        ):
            try:
                version = metadata.version(package_name)
            except metadata.PackageNotFoundError:
                continue
            records.append({"name": package_name, "version": version})
        return tuple(records)

    def dataframe_digest(
        self,
        frame: pd.DataFrame,
        *,
        sort_by: Sequence[str] | None = None,
    ) -> str:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        working = frame.reset_index()
        if sort_by:
            missing = set(sort_by).difference(working.columns)
            if missing:
                raise KeyError(
                    f"dataframe digest sort columns are missing: {sorted(missing)}"
                )
            working = working.sort_values(
                list(sort_by),
                kind="mergesort",
            ).reset_index(drop=True)
        return self.digest(
            {
                "index_names": list(frame.index.names),
                "columns": [
                    {
                        "name": str(column),
                        "dtype": str(working[column].dtype),
                        "values": [
                            _canonicalize(value)
                            for value in working[column].tolist()
                        ],
                    }
                    for column in working.columns
                ],
            }
        )


def _canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("identity values must be finite")
        return value
    if isinstance(value, np.generic):
        return _canonicalize(value.item())
    if isinstance(value, (pd.Timestamp, np.datetime64, datetime, date)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonicalize(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_canonicalize(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    if isinstance(value, Path):
        return value.as_posix()
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "configuration": {
                key: _canonicalize(item)
                for key, item in sorted(attributes.items())
                if not key.startswith("_") and not callable(item)
            },
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(
        part in normalized
        for part in (
            "connection",
            "credential",
            "password",
            "private_key",
            "query",
            "secret",
            "sql",
            "token",
        )
    )


__all__ = [
    "DefaultSimulationIdentityService",
    "SimulationCacheMissError",
    "SimulationCachePolicy",
    "SimulationCacheStore",
    "SimulationIdentityService",
]
