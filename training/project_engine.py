"""Safe project structure planning and scaffolding for LionAI.

This module deliberately does not interpret natural-language prompts. It accepts a
validated manifest produced by a planner, renders a preview, and applies it only
after explicit confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping


class ProjectStructureError(ValueError):
    """Raised when a project manifest is invalid or unsafe."""


@dataclass(frozen=True)
class Entry:
    path: str
    kind: str
    content: str = ""


@dataclass(frozen=True)
class ProjectPlan:
    name: str
    entries: tuple[Entry, ...]

    @classmethod
    def from_entries(
        cls, name: str, entries: Iterable[Mapping[str, str]]
    ) -> "ProjectPlan":
        if not isinstance(name, str) or not name.strip():
            raise ProjectStructureError("Project name must be a non-empty string.")
        if name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
            raise ProjectStructureError("Project name must be one safe path component.")

        normalized: list[Entry] = []
        seen: dict[str, str] = {}
        for raw in entries:
            if not isinstance(raw, Mapping):
                raise ProjectStructureError("Each entry must be an object.")
            if set(raw) - {"path", "kind", "content"}:
                raise ProjectStructureError("Entry contains unsupported fields.")
            path = raw.get("path")
            kind = raw.get("kind")
            content = raw.get("content", "")
            if not isinstance(path, str) or not path or "\x00" in path:
                raise ProjectStructureError("Entry path must be a non-empty string.")
            if "\\" in path:
                raise ProjectStructureError("Use '/' separators in manifest paths.")
            parsed = PurePosixPath(path)
            if parsed.is_absolute() or any(part in {"", ".", ".."} for part in path.split("/")):
                raise ProjectStructureError(f"Unsafe relative path: {path!r}")
            if kind not in {"directory", "file"}:
                raise ProjectStructureError("Entry kind must be 'directory' or 'file'.")
            if not isinstance(content, str):
                raise ProjectStructureError("File content must be text.")
            if kind == "directory" and content:
                raise ProjectStructureError("Directory entries cannot contain file content.")
            key = parsed.as_posix()
            if key in seen:
                raise ProjectStructureError(f"Duplicate path: {key}")
            seen[key] = kind
            normalized.append(Entry(key, kind, content))

        for entry in normalized:
            parts = PurePosixPath(entry.path).parts
            for index in range(1, len(parts)):
                parent = "/".join(parts[:index])
                if seen.get(parent) != "directory":
                    raise ProjectStructureError(
                        f"Parent directory must be explicitly declared: {parent}"
                    )

        for entry in normalized:
            if entry.kind == "file":
                prefix = entry.path + "/"
                if any(other.path.startswith(prefix) for other in normalized):
                    raise ProjectStructureError(f"File cannot contain children: {entry.path}")

        ordered = tuple(sorted(normalized, key=lambda item: (item.path.count("/"), item.path)))
        return cls(name=name, entries=ordered)

    def preview(self) -> str:
        """Return a stable, human-readable tree without touching the filesystem."""
        children: dict[str, list[Entry]] = {}
        for entry in self.entries:
            parent = str(PurePosixPath(entry.path).parent)
            if parent == ".":
                parent = ""
            children.setdefault(parent, []).append(entry)

        lines = [f"{self.name}/"]

        def visit(parent: str, prefix: str) -> None:
            items = sorted(
                children.get(parent, []),
                key=lambda item: (item.kind != "directory", item.path.rsplit("/", 1)[-1].lower()),
            )
            for index, item in enumerate(items):
                last = index == len(items) - 1
                branch = "└── " if last else "├── "
                leaf = item.path.rsplit("/", 1)[-1]
                suffix = "/" if item.kind == "directory" else ""
                lines.append(prefix + branch + leaf + suffix)
                if item.kind == "directory":
                    visit(item.path, prefix + ("    " if last else "│   "))

        visit("", "")
        return "\n".join(lines)

    def apply(self, destination: str | Path, *, confirmed: bool = False) -> Path:
        """Create the planned project; never overwrite existing paths."""
        if confirmed is not True:
            raise ProjectStructureError("Explicit confirmation is required to apply a plan.")

        base = Path(destination).expanduser()
        if base.exists() and (base.is_symlink() or not base.is_dir()):
            raise ProjectStructureError("Destination must be a real directory.")
        base.mkdir(parents=True, exist_ok=True)
        root = base / self.name
        if root.exists() or root.is_symlink():
            raise ProjectStructureError(f"Project already exists: {root}")
        root.mkdir()

        created_files: list[Path] = []
        created_dirs: list[Path] = [root]
        try:
            for entry in self.entries:
                if entry.kind == "directory":
                    target = root.joinpath(*PurePosixPath(entry.path).parts)
                    target.mkdir(parents=True, exist_ok=False)
                    created_dirs.append(target)

            for entry in self.entries:
                if entry.kind == "file":
                    target = root.joinpath(*PurePosixPath(entry.path).parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    current = root
                    for part in PurePosixPath(entry.path).parts[:-1]:
                        current = current / part
                        if current.is_symlink():
                            raise ProjectStructureError("Symlink path components are not allowed.")
                    with target.open("x", encoding="utf-8", newline="") as handle:
                        handle.write(entry.content)
                    created_files.append(target)
        except Exception:
            for file_path in reversed(created_files):
                try:
                    file_path.unlink()
                except OSError:
                    pass
            for dir_path in sorted(created_dirs, key=lambda item: len(item.parts), reverse=True):
                try:
                    dir_path.rmdir()
                except OSError:
                    pass
            raise
        return root
