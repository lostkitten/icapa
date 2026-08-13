"""Setuptools hook that enforces the public wheel module boundary."""

from __future__ import annotations

from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


_LOCAL_ONLY_MODULES = frozenset(
    {
        "icapa.demo",
        "icapa.setup",
    }
)
_LOCAL_ONLY_PACKAGE_PREFIXES = (
    "icapa.portfolio_construction.rules.data_processing",
)
_PROTECTED_IMPLEMENTATION_PACKAGE_PREFIXES = (
    "icapa.portfolio_construction.engines",
    "icapa.portfolio_construction.methodologies",
)
_PUBLIC_IMPLEMENTATION_MODULES = frozenset(
    {
        "icapa.portfolio_construction.engines.__init__",
        "icapa.portfolio_construction.engines.entropy_exposure_engine",
        "icapa.portfolio_construction.methodologies.__init__",
        (
            "icapa.portfolio_construction.methodologies."
            "entropy_exposure_methodology"
        ),
    }
)


class PublicBuildPy(build_py):
    """Exclude local research implementations from distributable wheels."""

    def run(self):
        package_root = Path(self.build_lib).joinpath("icapa")
        if package_root.is_dir():
            shutil.rmtree(package_root)
        super().run()
        build_root = Path(self.build_lib)
        for qualified_name in _LOCAL_ONLY_MODULES:
            candidate = build_root.joinpath(
                *qualified_name.split(".")
            ).with_suffix(".py")
            candidate.unlink(missing_ok=True)
        for package_prefix in _LOCAL_ONLY_PACKAGE_PREFIXES:
            package_path = build_root.joinpath(*package_prefix.split("."))
            if package_path.is_dir():
                shutil.rmtree(package_path)

    def find_all_modules(self):
        modules = super().find_all_modules()
        return [
            item
            for item in modules
            if _public_module_name(item[0], item[1])
        ]

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            item
            for item in modules
            if _public_module_name(item[0], item[1])
        ]


def _public_module_name(package: str, module: str) -> bool:
    qualified_name = f"{package}.{module}"
    return (
        qualified_name not in _LOCAL_ONLY_MODULES
        and not any(
            qualified_name == prefix
            or qualified_name.startswith(f"{prefix}.")
            for prefix in _LOCAL_ONLY_PACKAGE_PREFIXES
        )
        and (
            not any(
                qualified_name == prefix
                or qualified_name.startswith(f"{prefix}.")
                for prefix in _PROTECTED_IMPLEMENTATION_PACKAGE_PREFIXES
            )
            or qualified_name in _PUBLIC_IMPLEMENTATION_MODULES
        )
    )


setup(cmdclass={"build_py": PublicBuildPy})
