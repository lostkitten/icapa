from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_RULES = {
    "data_sources": {
        "analytics",
        "backtesting",
        "portfolio_construction",
        "reporting",
        "research",
        "workspace",
    },
    "portfolio_construction": {
        "analytics",
        "backtesting",
        "reporting",
        "research",
        "workspace",
    },
    "backtesting": {"analytics", "reporting", "research", "workspace"},
    "analytics": {"reporting", "research", "workspace"},
    "reporting": {"workspace"},
}
PURE_CROSS_DOMAIN_SERVICES = {
    ("reporting", "workspace.identity"),
}
ROOT_MODULES = {
    "analytics": {"__init__.py", "contracts.py", "registry.py"},
    "backtesting": {"__init__.py", "backtester.py"},
    "data_sources": {"__init__.py", "contracts.py"},
    "portfolio_construction": {"__init__.py", "context.py"},
    "reporting": {"__init__.py", "bundle.py", "contracts.py"},
    "research": {"__init__.py", "models.py", "results.py", "workspace.py"},
}


def test_ambiguous_utility_packages_do_not_exist() -> None:
    assert not PACKAGE_ROOT.joinpath("tools").exists()
    assert not PACKAGE_ROOT.joinpath("helpers").exists()


def test_domain_imports_follow_the_declared_direction() -> None:
    violations: list[str] = []
    for domain, forbidden in DOMAIN_RULES.items():
        for path in PACKAGE_ROOT.joinpath(domain).rglob("*.py"):
            for imported in _imports(path):
                root = imported.split(".", 1)[0]
                if (
                    root in forbidden
                    and (domain, imported) not in PURE_CROSS_DOMAIN_SERVICES
                ):
                    violations.append(
                        f"{path.relative_to(PACKAGE_ROOT)} imports {imported}"
                    )
    assert not violations, "\n".join(violations)


def test_research_workspace_name_is_unique() -> None:
    definitions: list[str] = []
    for domain in (
        "analytics",
        "backtesting",
        "data_sources",
        "portfolio_construction",
        "reporting",
        "research",
        "workspace",
    ):
        for path in PACKAGE_ROOT.joinpath(domain).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if any(
                isinstance(node, ast.ClassDef) and node.name == "ResearchWorkspace"
                for node in tree.body
            ):
                definitions.append(str(path.relative_to(PACKAGE_ROOT)))
    assert definitions == ["research/workspace.py"]


def test_ordinary_source_modules_remain_navigable() -> None:
    oversized: list[str] = []
    for domain in (
        "analytics",
        "backtesting",
        "data_sources",
        "portfolio_construction",
        "reporting",
        "research",
        "workspace",
    ):
        for path in PACKAGE_ROOT.joinpath(domain).rglob("*.py"):
            lines = len(path.read_text(encoding="utf-8").splitlines())
            if lines > 800:
                oversized.append(f"{path.relative_to(PACKAGE_ROOT)}: {lines}")
    assert not oversized, "\n".join(oversized)


def test_domain_roots_contain_only_owned_facades() -> None:
    for domain, expected in ROOT_MODULES.items():
        actual = {
            path.name
            for path in PACKAGE_ROOT.joinpath(domain).glob("*.py")
        }
        assert actual == expected, domain


def test_research_facade_and_orchestrators_remain_small() -> None:
    paths = (
        PACKAGE_ROOT / "research" / "workspace.py",
        PACKAGE_ROOT / "research" / "runners" / "single.py",
        PACKAGE_ROOT / "research" / "runners" / "preflight.py",
        PACKAGE_ROOT / "research" / "runners" / "reviews.py",
        PACKAGE_ROOT / "research" / "runners" / "simulation.py",
        PACKAGE_ROOT / "research" / "runners" / "analytics.py",
        PACKAGE_ROOT / "research" / "runners" / "artifacts.py",
    )
    oversized = {
        str(path.relative_to(PACKAGE_ROOT)): len(
            path.read_text(encoding="utf-8").splitlines()
        )
        for path in paths
        if len(path.read_text(encoding="utf-8").splitlines()) > 400
    }
    assert not oversized


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = path.relative_to(PACKAGE_ROOT).with_suffix("").parts
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.startswith("icapa."):
                    imports.add(name.removeprefix("icapa."))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = list(package[:-1])
                if path.name == "__init__.py":
                    anchor = list(package)
                keep = max(0, len(anchor) - node.level + 1)
                parts = anchor[:keep]
                if node.module:
                    parts.extend(node.module.split("."))
                if parts:
                    imports.add(parts[0])
            elif node.module and node.module.startswith("icapa."):
                imports.add(node.module.removeprefix("icapa."))
    return imports
