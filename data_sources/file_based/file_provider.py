from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
