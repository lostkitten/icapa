"""Automatic identities for executable components and source closures."""

from __future__ import annotations

from dataclasses import is_dataclass
from functools import lru_cache
from hashlib import sha256
from importlib import metadata
import ast
import inspect
from pathlib import Path
from typing import Any

from .identity_canonical import (
    UnfingerprintableComponentError,
    _is_secret_key,
    _sensitive_identity_token,
    automatic_digest,
    canonicalize,
)


def automatic_component_identity(component: object) -> dict[str, Any]:
    """Collect source, distribution, repository, and configuration identity."""

    target = (
        component
        if inspect.isclass(component)
        or inspect.isfunction(component)
        or inspect.ismethod(component)
        else type(component)
    )
    module = inspect.getmodule(target)
    module_name = getattr(target, "__module__", "")
    qualified_name = getattr(target, "__qualname__", getattr(target, "__name__", ""))
    source_path = _source_path(target)
    source_digest = sha256(source_path.read_bytes()).hexdigest()
    source_closure_digest, source_file_count = _source_closure_identity(
        source_path,
        module_name,
    )
    distribution_name, distribution_version = _distribution_identity(module_name)
    (
        distribution_python_digest,
        distribution_python_file_count,
        distribution_python_files_truncated,
    ) = _distribution_python_identity(distribution_name)
    repository = _repository_identity(source_path)
    return {
        "type": f"{module_name}.{qualified_name}".strip("."),
        "module": None if module is None else module.__name__,
        "source_file": source_path.name,
        "source_digest": source_digest,
        "source_closure_digest": source_closure_digest,
        "source_file_count": source_file_count,
        "distribution": distribution_name,
        "distribution_version": distribution_version,
        "distribution_python_digest": distribution_python_digest,
        "distribution_python_file_count": distribution_python_file_count,
        "distribution_python_files_truncated": (distribution_python_files_truncated),
        "repository": repository,
        "configuration_digest": automatic_digest(_component_configuration(component)),
    }


def automatic_source_closure_identity(component: object) -> dict[str, Any]:
    """Return the content identity of a component's local Python dependency closure.

    This narrower helper is used by cacheable recipe stages. It deliberately
    excludes mutable configuration, repository state, and installed-package
    metadata because those inputs are represented separately in stage cache
    keys.
    """

    target = (
        component
        if inspect.isclass(component)
        or inspect.isfunction(component)
        or inspect.ismethod(component)
        else type(component)
    )
    source_path = _source_path(target)
    module_name = getattr(target, "__module__", "")
    closure_digest, file_count = _source_closure_identity(
        source_path,
        module_name,
    )
    return {
        "type": (
            f"{module_name}."
            f"{getattr(target, '__qualname__', getattr(target, '__name__', ''))}"
        ).strip("."),
        "source_digest": sha256(source_path.read_bytes()).hexdigest(),
        "source_closure_digest": closure_digest,
        "source_file_count": file_count,
    }


def automatic_callable_identity(value: object) -> dict[str, Any]:
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
        return {
            "callable": (f"{type(value).__module__}." f"{type(value).__qualname__}"),
            "source": automatic_source_closure_identity(type(value)),
            "configuration_digest": automatic_digest(canonicalize(value)),
        }
    closure_values: dict[str, Any] = {}
    cells = function.__closure__ or ()
    for name, cell in zip(
        function.__code__.co_freevars,
        cells,
        strict=True,
    ):
        try:
            captured = cell.cell_contents
        except ValueError:
            closure_values[name] = {"empty_cell": True}
        else:
            closure_values[name] = {"value_digest": automatic_digest(captured)}
    result = {
        "callable": (f"{function.__module__}.{function.__qualname__}"),
        "source": automatic_source_closure_identity(function),
        "defaults_digest": automatic_digest(function.__defaults__),
        "keyword_defaults_digest": automatic_digest(function.__kwdefaults__),
        "closure": closure_values,
    }
    if bound_self is not None:
        result["bound_self_digest"] = automatic_digest(bound_self)
    return result


def _component_configuration(component: object) -> Any:
    if inspect.isclass(component):
        return {}
    if is_dataclass(component):
        return canonicalize(component)
    attributes = getattr(component, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            key: (
                _sensitive_identity_token(value)
                if _is_secret_key(key)
                else canonicalize(value)
            )
            for key, value in sorted(attributes.items())
            if not key.startswith("_")
        }
    return {}


def _source_path(target: object) -> Path:
    try:
        source = inspect.getsourcefile(target) or inspect.getfile(target)
    except (OSError, TypeError) as exc:
        raise UnfingerprintableComponentError(
            f"cannot locate source for {target!r}"
        ) from exc
    path = Path(source)
    if not path.is_file():
        raise UnfingerprintableComponentError(
            f"component source is not a readable file: {path}"
        )
    return path.resolve()


@lru_cache(maxsize=256)
def _distribution_identity(module_name: str) -> tuple[str | None, str | None]:
    top_level = module_name.partition(".")[0]
    candidates = metadata.packages_distributions().get(top_level, ())
    if not candidates:
        return None, None
    distribution_name = sorted(candidates)[0]
    try:
        return distribution_name, metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return distribution_name, None


@lru_cache(maxsize=256)
def _distribution_python_identity(
    distribution_name: str | None,
) -> tuple[str | None, int, bool]:
    """Hash all installed Python files for one distribution once per process."""

    if distribution_name is None:
        return None, 0, False
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError:
        return None, 0, False
    python_files = sorted(
        (
            file
            for file in (distribution.files or ())
            if str(file).casefold().endswith(".py")
        ),
        key=lambda file: str(file),
    )
    records: list[dict[str, Any]] = []
    for file in python_files:
        path = Path(distribution.locate_file(file))
        try:
            file_digest = _streaming_file_digest(path)
        except OSError:
            file_digest = None
        records.append(
            {
                "path": str(file).replace("\\", "/"),
                "digest": file_digest,
            }
        )
    payload = {
        "distribution": distribution_name,
        "files": records,
        "python_file_count": len(python_files),
        "truncated": False,
    }
    return automatic_digest(payload), len(python_files), False


def _streaming_file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_identity(source_path: Path) -> dict[str, Any] | None:
    current = source_path.parent
    for candidate in (current, *current.parents):
        git_path = candidate.joinpath(".git")
        if not git_path.exists():
            continue
        if git_path.is_file():
            content = git_path.read_text(encoding="utf-8").strip()
            if not content.startswith("gitdir:"):
                return None
            git_path = (candidate / content.partition(":")[2].strip()).resolve()
        head_path = git_path.joinpath("HEAD")
        if not head_path.is_file():
            return None
        head = head_path.read_text(encoding="utf-8").strip()
        commit = head
        if head.startswith("ref:"):
            ref = head.partition(":")[2].strip()
            ref_path = git_path.joinpath(ref)
            if ref_path.is_file():
                commit = ref_path.read_text(encoding="utf-8").strip()
            else:
                commit = _packed_ref(git_path, ref) or head
        return {
            "commit": commit,
            "source_content_digest": sha256(source_path.read_bytes()).hexdigest(),
        }
    return None


def _source_closure_identity(
    source_path: Path,
    module_name: str,
) -> tuple[str, int]:
    """Hash the reachable local Python import closure for one component."""

    project_root = _find_project_root(source_path)
    pending: list[tuple[Path, str]] = [(source_path.resolve(), module_name)]
    visited: dict[Path, str] = {}
    while pending:
        path, current_module = pending.pop()
        path = path.resolve()
        if path in visited or not path.is_file():
            continue
        if project_root is not None and not path.is_relative_to(project_root):
            continue
        try:
            raw = path.read_bytes()
            tree = ast.parse(raw, filename=path.name)
        except (OSError, SyntaxError):
            continue
        visited[path] = sha256(raw).hexdigest()
        if len(visited) >= 512:
            break
        for imported_module, level, aliases in _python_imports(tree):
            candidates = _resolve_local_imports(
                source_path=path,
                current_module=current_module,
                imported_module=imported_module,
                level=level,
                aliases=aliases,
                project_root=project_root,
            )
            pending.extend(candidates)
    payload = [
        {
            "module_path": (
                str(path.relative_to(project_root))
                if project_root is not None and path.is_relative_to(project_root)
                else path.name
            ),
            "digest": digest,
        }
        for path, digest in sorted(visited.items(), key=lambda item: str(item[0]))
    ]
    return automatic_digest(payload), len(payload)


def _python_imports(
    tree: ast.AST,
) -> list[tuple[str | None, int, tuple[str, ...]]]:
    imports: list[tuple[str | None, int, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, 0, ()))
        elif isinstance(node, ast.ImportFrom):
            imports.append(
                (
                    node.module,
                    int(node.level),
                    tuple(alias.name for alias in node.names if alias.name != "*"),
                )
            )
    return imports


def _resolve_local_imports(
    *,
    source_path: Path,
    current_module: str,
    imported_module: str | None,
    level: int,
    aliases: tuple[str, ...],
    project_root: Path | None,
) -> list[tuple[Path, str]]:
    module_candidates: list[str] = []
    if level:
        current_parts = current_module.split(".")
        is_package_init = source_path.name == "__init__.py"
        package_parts = current_parts if is_package_init else current_parts[:-1]
        keep = max(0, len(package_parts) - (level - 1))
        base = package_parts[:keep]
        if imported_module:
            base.extend(imported_module.split("."))
        if base:
            module_candidates.append(".".join(base))
            module_candidates.extend(".".join((*base, alias)) for alias in aliases)
    elif imported_module:
        module_candidates.append(imported_module)

    resolved: list[tuple[Path, str]] = []
    for module in dict.fromkeys(module_candidates):
        path = _module_source_path(
            module,
            source_path=source_path,
            project_root=project_root,
        )
        if path is not None:
            resolved.append((path, module))
    return resolved


def _module_source_path(
    module_name: str,
    *,
    source_path: Path,
    project_root: Path | None,
) -> Path | None:
    parts = module_name.split(".")
    roots: list[Path] = []
    if project_root is not None:
        roots.append(project_root)
        if project_root.parent not in roots:
            roots.append(project_root.parent)
    roots.extend(ancestor for ancestor in source_path.parents if ancestor not in roots)
    for root in roots:
        relative_parts = parts
        if root.name == parts[0]:
            relative_parts = parts[1:]
        module_path = root.joinpath(*relative_parts)
        file_candidate = module_path.with_suffix(".py")
        package_candidate = module_path.joinpath("__init__.py")
        for candidate in (file_candidate, package_candidate):
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if project_root is None or resolved.is_relative_to(project_root):
                return resolved
    return None


def _find_project_root(source_path: Path) -> Path | None:
    for candidate in (source_path.parent, *source_path.parents):
        if candidate.joinpath("pyproject.toml").is_file():
            return candidate.resolve()
        if candidate.joinpath(".git").exists():
            return candidate.resolve()
    return source_path.parent.resolve()


def _dependency_lock_identity(source_path: Path) -> str | None:
    project_root = _find_project_root(source_path)
    if project_root is None:
        return None
    candidates = [
        project_root.joinpath("pyproject.toml"),
        project_root.joinpath("uv.lock"),
        project_root.joinpath("poetry.lock"),
        *sorted(project_root.glob("requirements*.txt")),
    ]
    records = [
        {
            "name": path.name,
            "digest": sha256(path.read_bytes()).hexdigest(),
        }
        for path in candidates
        if path.is_file()
    ]
    return None if not records else automatic_digest(records)


def _packed_ref(git_path: Path, ref: str) -> str | None:
    packed = git_path.joinpath("packed-refs")
    if not packed.is_file():
        return None
    for line in packed.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "^")):
            continue
        commit, _, candidate = line.partition(" ")
        if candidate == ref:
            return commit
    return None


__all__ = [
    "automatic_callable_identity",
    "automatic_component_identity",
    "automatic_source_closure_identity",
]
