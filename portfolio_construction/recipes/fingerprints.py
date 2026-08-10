"""Source and callable fingerprints used by recipe cache identities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from hashlib import sha256
import ast
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any

from icapa.data_sources.provenance import private_parameter_digest


_SOURCE_FILE_CACHE: dict[
    tuple[Path, str, Path],
    tuple[tuple[int, int, int], str, tuple[tuple[Path, str], ...]],
] = {}
_SOURCE_CLOSURE_CACHE: dict[
    tuple[Path, str],
    tuple[
        tuple[tuple[Path, tuple[int, int, int]], ...],
        dict[Path, str],
    ],
] = {}
_BEHAVIOR_METHOD_NAMES = (
    "apply",
    "build_request",
    "calculate",
    "compute",
    "estimate",
    "execute",
    "optimise",
    "optimize",
    "run",
    "select",
    "solve",
    "transform",
)


def source_closure_identity(component: object) -> dict[str, Any]:
    """Hash a component and its reachable local Python import closure."""

    target = _callable_target(component)
    source_path = _source_path(target)
    module_name = getattr(target, "__module__", "")
    files = _local_source_closure(source_path, module_name)
    project_root = _project_root(source_path, module_name)
    source_key = source_path.relative_to(project_root)
    source_digest = files.get(source_key)
    if source_digest is None:
        source_digest = sha256(source_path.read_bytes()).hexdigest()
    payload = [
        {
            "path": str(path),
            "digest": digest,
        }
        for path, digest in sorted(files.items(), key=lambda item: str(item[0]))
    ]
    return {
        "type": (
            f"{module_name}."
            f"{getattr(target, '__qualname__', getattr(target, '__name__', ''))}"
        ).strip("."),
        "source_digest": source_digest,
        "source_closure_digest": _json_digest(payload),
        "source_file_count": len(files),
    }


def callable_identity(value: object) -> dict[str, Any]:
    """Identify callable source together with behavior-bearing bound state."""

    if not callable(value):
        raise TypeError("value must be callable")
    if inspect.ismethod(value):
        function = value.__func__
        bound_self = value.__self__
    elif inspect.isfunction(value):
        function = value
        bound_self = None
    else:
        public_state = {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
        return {
            "callable": f"{type(value).__module__}.{type(value).__qualname__}",
            "source": source_closure_identity(type(value)),
            "configuration_digest": _value_digest(public_state),
        }

    closure_values: dict[str, Any] = {}
    for name, cell in zip(
        function.__code__.co_freevars,
        function.__closure__ or (),
        strict=True,
    ):
        try:
            captured = cell.cell_contents
        except ValueError:
            closure_values[name] = {"empty_cell": True}
        else:
            closure_values[name] = {"value_digest": _value_digest(captured)}
    result = {
        "callable": f"{function.__module__}.{function.__qualname__}",
        "source": source_closure_identity(function),
        "defaults_digest": _value_digest(function.__defaults__),
        "keyword_defaults_digest": _value_digest(function.__kwdefaults__),
        "closure": closure_values,
    }
    if bound_self is not None:
        result["bound_self_digest"] = _value_digest(bound_self)
    return result


def component_tree_identity(
    component: object,
    *,
    state_digest: str | None = None,
) -> dict[str, Any]:
    """Identify implementation source for a component and injected behavior.

    Dataclass fields, mappings, and sequences are traversed recursively so a
    custom solver, covariance estimator, or callable cannot hide behind the
    source identity of its containing methodology. Unsupported opaque objects
    fail closed and therefore make cacheable execution unavailable.
    """

    records: list[dict[str, Any]] = []
    visited: set[int] = set()

    def visit(value: Any, path: str, *, root: bool = False) -> None:
        if value is None or isinstance(
            value,
            (bool, int, float, str, bytes, bytearray, Enum, date, datetime, Path),
        ):
            return
        marker = id(value)
        if marker in visited:
            return
        visited.add(marker)
        if isinstance(value, Mapping):
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                visit(item, f"{path}[{key!s}]")
            return
        if isinstance(value, (set, frozenset, list, tuple)):
            for position, item in enumerate(value):
                visit(item, f"{path}[{position}]")
            return
        if inspect.isclass(value):
            records.append(
                {
                    "path": path,
                    "component": source_closure_identity(value),
                }
            )
            return
        if callable(value) and not root:
            records.append(
                {
                    "path": path,
                    "callable": callable_identity(value),
                }
            )
            if inspect.isfunction(value) or inspect.ismethod(value):
                return
        if root or _has_behavior_methods(value):
            records.append(
                {
                    "path": path,
                    "component": source_closure_identity(value),
                }
            )
        if is_dataclass(value) and not isinstance(value, type):
            values = (
                (item.name, getattr(value, item.name))
                for item in fields(value)
            )
        else:
            attributes = getattr(value, "__dict__", None)
            if not isinstance(attributes, dict):
                return
            values = (
                (str(name), item)
                for name, item in sorted(attributes.items())
            )
        for name, item in values:
            visit(item, f"{path}.{name}")

    visit(component, "root", root=True)
    return {
        "root_type": (
            f"{type(component).__module__}.{type(component).__qualname__}"
        ),
        "components": records,
        "state_digest": (
            state_digest
            if state_digest is not None
            else component_tree_state_digest(component)
        ),
    }


def _has_behavior_methods(value: object) -> bool:
    if any(
        callable(getattr(value, name, None))
        for name in _BEHAVIOR_METHOD_NAMES
    ):
        return True
    attributes = getattr(type(value), "__dict__", {})
    return any(
        not str(name).startswith("_") and callable(item)
        for name, item in attributes.items()
    )


_VOLATILE_COMPONENT_STATE_FIELDS = frozenset(
    {
        "cache",
        "call_count",
        "calls",
        "counter",
        "counters",
        "lock",
        "memo",
        "mutex",
        "snapshot_requests",
    }
)


def component_tree_state_digest(component: object) -> str:
    """Hash recursively typed component state without exposing raw values."""

    records: list[dict[str, str]] = []
    visited: set[int] = set()

    def record(value: Any, path: str) -> None:
        records.append(
            {
                "path": path,
                "value_digest": private_parameter_digest({"value": value}),
            }
        )

    def visit(value: Any, path: str) -> None:
        if value is None or isinstance(
            value,
            (
                bool,
                int,
                float,
                str,
                bytes,
                bytearray,
                Enum,
                date,
                datetime,
                Path,
            ),
        ):
            record(value, path)
            return
        marker = id(value)
        if marker in visited:
            return
        visited.add(marker)
        if isinstance(value, Mapping):
            key_records = [
                (
                    private_parameter_digest({"key": key}),
                    item,
                )
                for key, item in value.items()
            ]
            record(
                {
                    "mapping_type": (
                        f"{type(value).__module__}."
                        f"{type(value).__qualname__}"
                    ),
                    "key_digests": tuple(
                        key_digest
                        for key_digest, _ in sorted(
                            key_records,
                            key=lambda item: item[0],
                        )
                    ),
                },
                path,
            )
            for position, (_, item) in enumerate(
                sorted(key_records, key=lambda record: record[0])
            ):
                visit(item, f"{path}[{position}]")
            return
        if isinstance(value, (set, frozenset)):
            # Supported scalar sets are identified atomically. A set of
            # behavior objects is unordered and therefore fails closed.
            record(value, path)
            return
        if isinstance(value, (list, tuple)):
            record(
                {
                    "sequence_type": (
                        f"{type(value).__module__}."
                        f"{type(value).__qualname__}"
                    ),
                    "length": len(value),
                },
                path,
            )
            for position, item in enumerate(value):
                visit(item, f"{path}[{position}]")
            return
        if inspect.isclass(value):
            return
        if inspect.isfunction(value) or inspect.ismethod(value):
            records.append(
                {
                    "path": path,
                    "value_digest": _json_digest(callable_identity(value)),
                }
            )
            return
        if is_dataclass(value) and not isinstance(value, type):
            values = (
                (item.name, getattr(value, item.name))
                for item in fields(value)
            )
        else:
            attributes = getattr(value, "__dict__", None)
            if not isinstance(attributes, dict):
                scalar = getattr(value, "item", None)
                if callable(scalar):
                    try:
                        record(scalar(), path)
                        return
                    except (TypeError, ValueError):
                        pass
                raise ValueError(
                    "component state contains unsupported value type "
                    f"{type(value).__module__}.{type(value).__qualname__}"
                )
            values = (
                (str(name), item)
                for name, item in sorted(attributes.items())
            )
        for name, item in values:
            normalized = name.strip("_").casefold()
            if normalized in _VOLATILE_COMPONENT_STATE_FIELDS:
                continue
            visit(item, f"{path}.{name}")

    visit(component, "root")
    return _json_digest(records)


def _callable_target(component: object) -> object:
    if inspect.isclass(component) or inspect.isfunction(component) or inspect.ismethod(
        component
    ):
        return component
    return type(component)


def _source_path(target: object) -> Path:
    try:
        raw = inspect.getsourcefile(target) or inspect.getfile(target)
    except (OSError, TypeError) as exc:
        raise ValueError(f"cannot locate source for {target!r}") from exc
    path = _lexical_absolute(Path(raw))
    if not path.is_file():
        raise ValueError(f"component source is not a readable file: {path}")
    return path


def _local_source_closure(
    source_path: Path,
    module_name: str,
) -> dict[Path, str]:
    project_root = _project_root(source_path, module_name)
    cache_key = (source_path, module_name)
    cached = _SOURCE_CLOSURE_CACHE.get(cache_key)
    if cached is not None and all(
        _file_signature(path) == signature
        for path, signature in cached[0]
    ):
        return dict(cached[1])
    pending: list[tuple[Path, str]] = [(source_path, module_name)]
    scheduled = {source_path}
    visited: dict[Path, str] = {}
    while pending:
        path, current_module = pending.pop()
        if not path.is_relative_to(project_root):
            continue
        source = _cached_source_file(
            path,
            current_module=current_module,
            project_root=project_root,
        )
        if source is None:
            continue
        digest, imports = source
        relative = path.relative_to(project_root) if project_root else path
        visited[relative] = digest
        if len(visited) >= 512:
            break
        for candidate, name in imports:
            if candidate in scheduled:
                continue
            scheduled.add(candidate)
            pending.append((candidate, name))
    absolute_signatures = tuple(
        sorted(
            (
                (absolute, _file_signature(absolute))
                for relative in visited
                for absolute in (project_root.joinpath(relative),)
            ),
            key=lambda item: str(item[0]),
        )
    )
    if len(_SOURCE_CLOSURE_CACHE) >= 256:
        _SOURCE_CLOSURE_CACHE.clear()
    _SOURCE_CLOSURE_CACHE[cache_key] = (
        absolute_signatures,
        dict(visited),
    )
    return visited


def _cached_source_file(
    path: Path,
    *,
    current_module: str,
    project_root: Path,
) -> tuple[str, tuple[tuple[Path, str], ...]] | None:
    signature = _file_signature(path)
    key = (path, current_module, project_root)
    cached = _SOURCE_FILE_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1], cached[2]
    try:
        raw = path.read_bytes()
        tree = ast.parse(raw, filename=path.name)
    except (OSError, SyntaxError):
        return None
    result = (
        sha256(raw).hexdigest(),
        tuple(
            _resolved_imports(
                tree,
                current_module=current_module,
                current_path=path,
                project_root=project_root,
            )
        ),
    )
    if len(_SOURCE_FILE_CACHE) >= 4096:
        _SOURCE_FILE_CACHE.clear()
    _SOURCE_FILE_CACHE[key] = (signature, result[0], result[1])
    return result


def _file_signature(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


def _project_root(source_path: Path, module_name: str) -> Path:
    module_parts = module_name.split(".") if module_name else []
    root = source_path.parent
    if source_path.name == "__init__.py":
        levels = len(module_parts)
    else:
        levels = max(0, len(module_parts) - 1)
    for _ in range(levels):
        root = root.parent
    return root


def _lexical_absolute(path: Path) -> Path:
    """Return one normalized absolute path without resolving filesystem links."""

    absolute = path if path.is_absolute() else Path.cwd().joinpath(path)
    return Path(os.path.normpath(os.fspath(absolute)))


def _resolved_imports(
    tree: ast.AST,
    *,
    current_module: str,
    current_path: Path,
    project_root: Path,
) -> list[tuple[Path, str]]:
    resolved: list[tuple[Path, str]] = []
    package = (
        current_module
        if current_path.name == "__init__.py"
        else current_module.rpartition(".")[0]
    )
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                relative = "." * node.level + base
                try:
                    base = importlib.util.resolve_name(relative, package)
                except (ImportError, ValueError):
                    continue
            if base:
                names.append(base)
                names.extend(f"{base}.{alias.name}" for alias in node.names)
        for name in names:
            candidate = _module_path(name, project_root)
            if candidate is not None:
                resolved.append((candidate, name))
    return resolved


def _module_path(module_name: str, project_root: Path) -> Path | None:
    relative = Path(*module_name.split("."))
    module_file = project_root.joinpath(relative).with_suffix(".py")
    package_file = project_root.joinpath(relative, "__init__.py")
    if module_file.is_file():
        return module_file
    if package_file.is_file():
        return package_file
    return None


def _value_digest(value: Any) -> str:
    return _json_digest(_stable_value(value))


def _json_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _stable_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("fingerprint values must be finite")
        return value
    if isinstance(value, Enum):
        return _stable_value(value.value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return {"path": str(value)}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _stable_value(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        items = [_stable_value(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_stable_value(item) for item in value]
    if callable(value):
        return callable_identity(value)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        public = {
            key: item
            for key, item in attributes.items()
            if not key.startswith("_")
        }
        if public:
            return {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "configuration": _stable_value(public),
            }
    raise ValueError(
        f"value of type {type(value).__module__}.{type(value).__qualname__} "
        "cannot be fingerprinted"
    )


__all__ = [
    "callable_identity",
    "component_tree_identity",
    "component_tree_state_digest",
    "source_closure_identity",
]
