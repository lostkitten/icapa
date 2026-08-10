from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass
class FileProvider:
    """Generic CSV/Excel loader for controlled ad-hoc inputs."""

    name: str = "file"

    def read(self, file_path: str | Path, **kwargs: Any) -> pd.DataFrame:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path, **kwargs)
        if suffix in {".xls", ".xlsx", ".xlsm"}:
            try:
                return pd.read_excel(path, **kwargs)
            except ImportError as exc:
                engine = "xlrd" if suffix == ".xls" else "openpyxl"
                raise ImportError(
                    f"reading {suffix} files requires the optional {engine!r} package"
                ) from exc
        raise ValueError(f"unsupported input file type: {suffix}")

    def describe_snapshot(
        self,
        capability: str,
        request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return an automatic content identity without exposing a local path."""

        raw_path = request.get("file_path")
        if raw_path is None:
            raise ValueError("file snapshot requests require file_path")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"input file does not exist: {path.name}")
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return {
            "capability": str(capability),
            "file_name": path.name,
            "file_type": path.suffix.casefold(),
            "size_bytes": path.stat().st_size,
            "content_digest": digest.hexdigest(),
        }
