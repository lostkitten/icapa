"""Low-level facade for one fixed-location research workspace."""

from __future__ import annotations

from .artifact_repository import ArtifactOperations
from .artifacts import ParquetArtifactStore
from .catalog import CatalogOperations
from .maintenance import MaintenanceOperations
from .manifest_repository import ManifestOperations
from .manifests import (
    ManifestIntegrityError,
    RUN_MANIFEST_SCHEMA_VERSION,
    WorkspaceRepositoryError,
)
from .readers import get_workspace_root, validate_workspace_name


class WorkspaceRepository(
    ManifestOperations,
    ArtifactOperations,
    MaintenanceOperations,
    CatalogOperations,
):
    """Coordinate manifests, immutable artifacts, and the local catalog.

    The facade owns the stable public API. Each inherited operation group is
    implemented by the workspace module named for that storage responsibility.
    """

    def __init__(self, workspace_name: str) -> None:
        self.workspace_name = validate_workspace_name(workspace_name)
        self.root = get_workspace_root()
        self.workspace_path = self.root.joinpath(self.workspace_name)
        self.catalog_path = self.workspace_path.joinpath("catalog.sqlite")
        self.artifact_store = ParquetArtifactStore(self.workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self._initialize_catalog()

    @classmethod
    def open(cls, workspace_name: str) -> "WorkspaceRepository":
        """Open or create a workspace without accepting an arbitrary path."""

        return cls(workspace_name)

    @classmethod
    def list(cls) -> tuple[str, ...]:
        """List valid named workspaces beneath the fixed deployment root."""

        root = get_workspace_root()
        if not root.is_dir():
            return ()
        return tuple(
            sorted(
                path.name
                for path in root.iterdir()
                if path.is_dir() and _is_workspace_name(path.name)
            )
        )


def _is_workspace_name(value: str) -> bool:
    try:
        validate_workspace_name(value)
    except ValueError:
        return False
    return True


__all__ = [
    "ManifestIntegrityError",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "WorkspaceRepository",
    "WorkspaceRepositoryError",
]
