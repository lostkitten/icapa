#!/usr/bin/env python3
"""Validate that a Git snapshot is safe for the public repository."""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath


ZERO_OID = "0" * 40
PLACEHOLDER_DIRECTORIES = (
    "portfolio_construction/methodologies",
    "portfolio_construction/rules/engines",
    "portfolio_construction/rules/data_processing",
)
ALLOWED_PLACEHOLDERS = {
    f"{directory}/.gitkeep".casefold() for directory in PLACEHOLDER_DIRECTORIES
}
LOCAL_ONLY_FILES = {
    "portfolio_construction/calculation_variant.py",
    "demo.py",
    "tests/smoke/test_methodology_demos.py",
    "tests/smoke/test_end_to_end_research_demo.py",
}
PUBLIC_DATA_SOURCE_FILES = {
    "data_sources/__init__.py",
    "data_sources/contracts.py",
    "data_sources/exceptions.py",
    "data_sources/file_provider.py",
    "data_sources/interfaces.py",
    "data_sources/registry.py",
    "data_sources/sql_server.py",
    "data_sources/factset/__init__.py",
    "data_sources/factset/factset.py",
    "data_sources/snowflake/__init__.py",
    "data_sources/snowflake/snowflake.py",
}
ALLOWED_BINARY_FILES = {
    "assets/icapa.png",
    "reporting/templates/index_research_report.xlsx",
}
BLOCKED_SUFFIXES = {
    ".csv",
    ".db",
    ".feather",
    ".key",
    ".parquet",
    ".pem",
    ".pickle",
    ".pkl",
    ".sqlite",
    ".sqlite3",
    ".tsv",
    ".xls",
    ".xlsx",
}
GENERATED_PARTS = {
    ".ds_store",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
}
PROHIBITED_SUBSTRING_HASHES = {
    4: {
        "fd1fa5baa00345fc6bc833e7ef18d4da3b4bf3a7dcf1863e6500e6300382c61f",
        "9b66ebf8bce2099dfa88d9bf25b4d5b18ec542fdb51756b5937b262e265adc97",
        "4976aaff941a2601b5a5c509e89259ff0feac8f5ccbb3235295e90f010cfe17f",
    },
    5: {
        "a2c2ff8694ae9bf6542bfcdd6759d7b9ddd50fbef6d0f291809e1da0f1cfa4dc",
        "3f8180582ae4d3c1e105ee16cf29ac66458e1653f04a06168c4d611c3607ed6b",
    },
    7: {
        "ec87d0e0735ce8d20ddf792630a066ab99a7e3370281adc2eba9c3192033ff7f",
    },
    15: {
        "9899ffe89013fae6f7c74c6f248e24a4bf383dfd34d13583a9dccaf48f333b6c",
        "f361c7c2f1d1214ba59b3da0b4c8947823da9f11dd607a40231d3bf267d7e1c6",
    },
    20: {
        "7a8144cc4f4e8ae319aaadfc065c29a0106e8e8bc0b232a820d10c2c5246c745",
        "2c0ecc16e95e4432ba6f695b10a4ecd2fb2874b1c462348fca61a76f1bf8b0f5",
        "c32c0549854fcdd3cd7de635d342abdb032a41e460d5d3d466862db7b6a2af6d",
    },
    21: {
        "bfecd7e4b868856bfeac83fc24ae2d304eb4268ed2682e528f6a633b69d4e1fe",
    },
    25: {
        "1fa1387cf6353a2372c1a46063cea4778ed8041540ac63479fce88fc862313c8",
    },
    26: {
        "fd5fb61f546cdc90cebc5921cd459107a9784e8418d61ae3241b9cd0a724c677",
    },
}
HAN_PATTERN = re.compile(
    "[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    "\U00020000-\U0002fa1f]"
)
LOCAL_PATH_PATTERN = re.compile(
    (
        r"(?:/"
        + r"Users/"
        + r"|/"
        + r"home/[A-Za-z0-9._-]+/"
        + r"|[A-Za-z]:\\"
        + r"Users\\)"
    ),
    re.IGNORECASE,
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
)


@dataclass(frozen=True)
class Entry:
    mode: str
    oid: str
    path: str


def _git(*arguments: str, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        ("git", *arguments),
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or f"git command failed: {' '.join(arguments)}")
    return result.stdout


def _index_entries() -> list[Entry]:
    output = _git("ls-files", "--stage", "-z")
    entries: list[Entry] = []
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, oid, stage = metadata.decode("ascii").split()
        if stage != "0":
            raise RuntimeError("the index contains unresolved merge entries")
        entries.append(Entry(mode, oid, raw_path.decode("utf-8")))
    return entries


def _tree_entries(revision: str) -> list[Entry]:
    output = _git("ls-tree", "-rz", "--full-tree", revision)
    entries: list[Entry] = []
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_type, oid = metadata.decode("ascii").split()
        if object_type != "blob":
            entries.append(Entry(mode, oid, raw_path.decode("utf-8")))
            continue
        entries.append(Entry(mode, oid, raw_path.decode("utf-8")))
    return entries


def _blob(oid: str) -> bytes:
    return _git("cat-file", "blob", oid)


def _validate_path(
    entry: Entry,
    errors: list[str],
    *,
    enforce_public_data_sources: bool,
) -> None:
    path = PurePosixPath(entry.path)
    folded = entry.path.casefold()
    parts = {part.casefold() for part in path.parts}

    if enforce_public_data_sources and folded.startswith("data_sources/"):
        allowed_files = {item.casefold() for item in PUBLIC_DATA_SOURCE_FILES}
        if folded not in allowed_files:
            errors.append(f"non-public data-source path is tracked: {entry.path}")

    for directory in PLACEHOLDER_DIRECTORIES:
        prefix = f"{directory.casefold()}/"
        if folded.startswith(prefix) and folded not in ALLOWED_PLACEHOLDERS:
            errors.append(f"protected implementation path is tracked: {entry.path}")

    if folded in {item.casefold() for item in LOCAL_ONLY_FILES}:
        errors.append(f"local-only implementation file is tracked: {entry.path}")
    if (
        folded.startswith("portfolio_construction/")
        and path.name.casefold().endswith(("_methodology.py", "_engine.py"))
    ):
        errors.append(f"implementation-shaped source file is tracked: {entry.path}")
    if parts & GENERATED_PARTS:
        errors.append(f"generated artifact is tracked: {entry.path}")
    if path.suffix.casefold() in BLOCKED_SUFFIXES and entry.path not in ALLOWED_BINARY_FILES:
        errors.append(f"unapproved data or binary file is tracked: {entry.path}")
    if entry.mode in {"120000", "160000"}:
        errors.append(f"symlink or submodule is not allowed: {entry.path}")
    if folded in ALLOWED_PLACEHOLDERS and entry.mode != "100644":
        errors.append(f"placeholder must use mode 100644: {entry.path}")


def _scan_text(label: str, text: str, errors: list[str]) -> None:
    folded = text.casefold()
    for length, prohibited_hashes in PROHIBITED_SUBSTRING_HASHES.items():
        for start in range(len(folded) - length + 1):
            candidate = folded[start : start + length].encode("utf-8")
            if hashlib.sha256(candidate).hexdigest() in prohibited_hashes:
                errors.append(f"prohibited identifier found in {label}")
                break
    if HAN_PATTERN.search(text):
        errors.append(f"Han character found in {label}")
    if LOCAL_PATH_PATTERN.search(text):
        errors.append(f"developer-specific absolute path found in {label}")
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            errors.append(f"credential-like value found in {label}")


def _validate_blob(entry: Entry, payload: bytes, errors: list[str]) -> None:
    folded = entry.path.casefold()
    if folded in ALLOWED_PLACEHOLDERS and payload.strip():
        errors.append(f"placeholder must be empty: {entry.path}")

    if entry.path.endswith(".xlsx"):
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    data = archive.read(member)
                    text = data.decode("utf-8", errors="ignore")
                    _scan_text(f"{entry.path}!{member.filename}", text, errors)
        except zipfile.BadZipFile:
            errors.append(f"invalid XLSX container: {entry.path}")
        return

    if b"\0" in payload:
        return
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return
    _scan_text(entry.path, text, errors)


def _validate(
    entries: list[Entry],
    label: str,
    *,
    require_placeholders: bool = True,
    enforce_public_data_sources: bool = True,
) -> None:
    errors: list[str] = []
    seen_casefolded: dict[str, str] = {}
    for entry in entries:
        folded = entry.path.casefold()
        previous = seen_casefolded.get(folded)
        if previous is not None and previous != entry.path:
            errors.append(
                f"case-colliding paths are not allowed: {previous!r} and {entry.path!r}"
            )
        seen_casefolded[folded] = entry.path
        _validate_path(
            entry,
            errors,
            enforce_public_data_sources=enforce_public_data_sources,
        )
        if entry.mode.startswith("100"):
            _validate_blob(entry, _blob(entry.oid), errors)

    if require_placeholders:
        missing = sorted(ALLOWED_PLACEHOLDERS - set(seen_casefolded))
        for path in missing:
            errors.append(f"required placeholder is missing: {path}")

    if errors:
        print(f"Public repository guard failed for {label}:", file=sys.stderr)
        for error in sorted(set(errors)):
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Public repository guard passed for {label}: {len(entries)} tracked entries")


def _history(revision: str) -> None:
    commits = _git("rev-list", revision).decode("ascii").splitlines()
    if not commits:
        raise RuntimeError(f"no commits are reachable from {revision!r}")
    for position, commit in enumerate(commits):
        _validate(
            _tree_entries(commit),
            commit,
            require_placeholders=position == 0,
            enforce_public_data_sources=position == 0,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", action="store_true")
    group.add_argument("--tree", metavar="REVISION")
    group.add_argument("--history", metavar="REVISION")
    arguments = parser.parse_args()

    try:
        if arguments.index:
            _validate(_index_entries(), "Git index")
        elif arguments.tree:
            _validate(_tree_entries(arguments.tree), arguments.tree)
        else:
            _history(arguments.history)
    except RuntimeError as exc:
        print(f"Public repository guard could not run: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
