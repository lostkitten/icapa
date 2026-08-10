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
from pathlib import Path
from typing import Any


def source_closure_identity(component: object) -> dict[str, Any]:
    """Hash a component and its reachable local Python import closure."""

    target = _callable_target(component)
    source_path = _source_path(target)
    module_name = getattr(target, "__module__", "")
    files = _local_source_closure(source_path, module_name)
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
        "source_digest": sha256(source_path.read_bytes()).hexdigest(),
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
    path = Path(raw)
    if not path.is_file():
        raise ValueError(f"component source is not a readable file: {path}")
    return path.resolve()


def _local_source_closure(
    source_path: Path,
    module_name: str,
) -> dict[Path, str]:
    project_root = _project_root(source_path, module_name)
    pending: list[tuple[Path, str, int]] = [(source_path, module_name, 0)]
    visited: dict[Path, str] = {}
    while pending:
        path, current_module, depth = pending.pop()
        path = path.resolve()
        if path in visited or not path.is_file():
            continue
        if project_root is not None and not path.is_relative_to(project_root):
            continue
        raw = path.read_bytes()
        try:
            tree = ast.parse(raw, filename=path.name)
        except SyntaxError:
            continue
        relative = path.relative_to(project_root) if project_root else path
        visited[relative] = sha256(raw).hexdigest()
        if len(visited) >= 128:
            break
        if depth >= 2:
            continue
        pending.extend(
            (candidate, name, depth + 1)
            for candidate, name in _resolved_imports(
                tree,
                current_module=current_module,
                current_path=path,
                project_root=project_root,
            )
        )
    return visited


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


__all__ = ["callable_identity", "source_closure_identity"]
