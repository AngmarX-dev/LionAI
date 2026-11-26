"""Reconstruct safe, reviewable project manifests from trees or directories.

Filesystem inspection is read-only. File contents are excluded by default;
optional text capture is bounded and skips likely secret files. Applying a
returned ProjectPlan is a separate operation handled by project_engine.
"""
from __future__ import annotations

import os
from pathlib import Path
import re

from training.project_engine import ProjectPlan, ProjectStructureError


DEFAULT_MAX_ENTRIES = 10_000
DEFAULT_MAX_FILE_BYTES = 256 * 1024
_SECRET_NAME = re.compile(
    r"(^\.env(?:\.|$)|\.pem$|\.key$|id_rsa|id_ed25519|credentials?|"
    r"secrets?|token|password|\.p12$|\.pfx$)",
    re.IGNORECASE,
)
_TREE_BRANCH = re.compile(r"^(?P<prefix>(?:(?:│   )|(?:    ))*)(?:├── |└── )(?P<name>.+)$")


def _safe_component(name: str) -> str:
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
    ):
        raise ProjectStructureError(f"Unsafe tree entry name: {name!r}")
    return name


def _make_plan(name: str, kinds: dict[str, str], contents: dict[str, str] | None = None) -> ProjectPlan:
    contents = contents or {}
    entries = [
        {
            "path": path,
            "kind": kind,
            **({"content": contents.get(path, "")} if kind == "file" else {}),
        }
        for path, kind in kinds.items()
    ]
    return ProjectPlan.from_entries(name, entries)


def parse_tree_text(
    tree_text: str,
    *,
    project_name: str | None = None,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> ProjectPlan:
    """Parse a Unicode/ASCII tree listing into a structure-only ProjectPlan.

    Canonical input uses lines such as ├── app/ and │   └── main.py.
    A trailing slash marks a directory. A node with children is also a directory;
    other leaves are files. No file contents are inferred from names.
    """
    if not isinstance(tree_text, str) or not tree_text.strip():
        raise ProjectStructureError("Tree text must be a non-empty string.")
    if not isinstance(max_entries, int) or max_entries < 1:
        raise ProjectStructureError("max_entries must be a positive integer.")

    lines = [line.rstrip() for line in tree_text.splitlines() if line.strip()]
    if not lines:
        raise ProjectStructureError("Tree text contains no entries.")

    first = lines[0].strip()
    root_name = project_name
    start_index = 0
    if not _TREE_BRANCH.match(first):
        header = first[:-1] if first.endswith("/") else first
        _safe_component(header)
        root_name = root_name or header
        start_index = 1
    if root_name is None:
        raise ProjectStructureError("Provide project_name or a root directory header.")
    _safe_component(root_name)

    kinds: dict[str, str] = {}
    stack: list[str] = []
    child_parents: set[str] = set()
    for line_number, line in enumerate(lines[start_index:], start=start_index + 1):
        match = _TREE_BRANCH.match(line)
        if not match:
            raise ProjectStructureError(f"Unsupported tree syntax on line {line_number}.")
        prefix = match.group("prefix")
        if len(prefix) % 4:
            raise ProjectStructureError(f"Invalid indentation on line {line_number}.")
        depth = len(prefix) // 4
        if depth > len(stack):
            raise ProjectStructureError(f"Tree indentation skips a parent on line {line_number}.")
        name = match.group("name").strip()
        is_dir = name.endswith("/")
        if is_dir:
            name = name[:-1]
        _safe_component(name)

        if depth == 0:
            parent = ""
            stack = []
        else:
            parent = stack[depth - 1]
            stack = stack[:depth]

        path = f"{parent}/{name}" if parent else name
        if path in kinds:
            raise ProjectStructureError(f"Duplicate tree path: {path}")
        if len(kinds) >= max_entries:
            raise ProjectStructureError(f"Tree exceeds the {max_entries} entry limit.")
        kinds[path] = "directory" if is_dir else "file"
        if parent:
            child_parents.add(parent)
        # Keep every node on the ancestry stack; if a later node is nested
        # beneath it, child_parents will correctly promote it to a directory.
        stack.append(path)

    for parent in child_parents:
        if parent not in kinds:
            raise ProjectStructureError(f"Missing parent directory: {parent}")
        kinds[parent] = "directory"

    return _make_plan(root_name, kinds)


def scan_directory(
    source: str | Path,
    *,
    project_name: str | None = None,
    include_text_contents: bool = False,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> ProjectPlan:
    """Read a directory tree without following symlinks or modifying anything.

    By default only paths and file/directory kinds are captured. If
    include_text_contents=True, bounded UTF-8 files are included, except files
    with names commonly associated with credentials or secrets.
    """
    root = Path(source).expanduser()
    if root.is_symlink() or not root.exists() or not root.is_dir():
        raise ProjectStructureError("Source must be an existing, non-symlink directory.")
    if not isinstance(max_entries, int) or max_entries < 1:
        raise ProjectStructureError("max_entries must be a positive integer.")
    if not isinstance(max_file_bytes, int) or max_file_bytes < 0:
        raise ProjectStructureError("max_file_bytes must be a non-negative integer.")

    root_name = project_name or root.name
    _safe_component(root_name)
    kinds: dict[str, str] = {}
    contents: dict[str, str] = {}
    count = 0

    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        safe_dirs: list[str] = []
        for dirname in sorted(dirnames):
            full = current_path / dirname
            if full.is_symlink():
                raise ProjectStructureError(f"Symlink directories are not supported: {full}")
            rel = full.relative_to(root).as_posix()
            kinds[rel] = "directory"
            count += 1
            safe_dirs.append(dirname)
        dirnames[:] = safe_dirs

        for filename in sorted(filenames):
            full = current_path / filename
            if full.is_symlink():
                raise ProjectStructureError(f"Symlink files are not supported: {full}")
            rel = full.relative_to(root).as_posix()
            kinds[rel] = "file"
            count += 1
            if include_text_contents and not _SECRET_NAME.search(filename):
                try:
                    if full.stat().st_size <= max_file_bytes:
                        contents[rel] = full.read_bytes().decode("utf-8")
                except (OSError, UnicodeDecodeError):
                    pass
        if count > max_entries:
            raise ProjectStructureError(f"Directory exceeds the {max_entries} entry limit.")

    return _make_plan(root_name, kinds, contents)


def reconstruct_project(
    source: str | Path,
    *,
    project_name: str | None = None,
    include_text_contents: bool = False,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> ProjectPlan:
    """Convenience wrapper for read-only filesystem reconstruction."""
    return scan_directory(
        source,
        project_name=project_name,
        include_text_contents=include_text_contents,
        max_entries=max_entries,
        max_file_bytes=max_file_bytes,
    )
