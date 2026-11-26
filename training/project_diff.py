"""Read-only comparison of reconstructed and desired project plans.

This module produces review data only. It never writes, deletes, renames, or
overwrites filesystem entries.
"""
from __future__ import annotations

from dataclasses import dataclass

from training.project_engine import ProjectPlan, ProjectStructureError


@dataclass(frozen=True)
class DiffEntry:
    """One path in a project comparison."""

    path: str
    current_kind: str | None
    desired_kind: str | None
    reason: str = ""


@dataclass(frozen=True)
class ProjectDiff:
    """Immutable, read-only result of comparing two project plans."""

    project_name: str
    added: tuple[DiffEntry, ...]
    removed: tuple[DiffEntry, ...]
    modified: tuple[DiffEntry, ...]
    unchanged: tuple[DiffEntry, ...]
    conflicts: tuple[DiffEntry, ...]
    content_compared: bool = False

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.modified or self.conflicts)

    def summary(self) -> dict[str, int | bool]:
        """Return machine-friendly counts without exposing mutable internals."""
        return {
            "added": len(self.added),
            "removed": len(self.removed),
            "modified": len(self.modified),
            "unchanged": len(self.unchanged),
            "conflicts": len(self.conflicts),
            "content_compared": self.content_compared,
            "has_changes": self.has_changes,
        }

    def preview(self) -> str:
        """Render a stable textual report; this method is strictly read-only."""
        lines = [
            f"Project diff: {self.project_name}",
            f"Content comparison: {'enabled' if self.content_compared else 'disabled (structure only)'}",
        ]
        sections = (
            ("ADDED", self.added),
            ("REMOVED (review only; never auto-deleted)", self.removed),
            ("MODIFIED", self.modified),
            ("TYPE CONFLICTS (manual resolution required)", self.conflicts),
            ("UNCHANGED", self.unchanged),
        )
        for title, entries in sections:
            lines.append(f"{title}: {len(entries)}")
            for entry in entries:
                kinds = ""
                if entry.current_kind or entry.desired_kind:
                    kinds = f" [{entry.current_kind or '-'} -> {entry.desired_kind or '-'}]"
                suffix = f" — {entry.reason}" if entry.reason else ""
                lines.append(f"  {entry.path}{kinds}{suffix}")
        return "\n".join(lines)


def compare_project_plans(
    current: ProjectPlan,
    desired: ProjectPlan,
    *,
    compare_contents: bool = False,
) -> ProjectDiff:
    """Compare two plans without touching their source or destination.

    By default only paths and entry kinds are compared. Set compare_contents
    only when both plans contain complete, trustworthy text snapshots; when
    enabled, differing file content is reported as modified. The function
    never applies suggested changes or performs deletions.
    """
    if not isinstance(current, ProjectPlan) or not isinstance(desired, ProjectPlan):
        raise ProjectStructureError("Both inputs must be ProjectPlan instances.")
    if current.name != desired.name:
        raise ProjectStructureError("Project names must match before comparing plans.")
    if not isinstance(compare_contents, bool):
        raise ProjectStructureError("compare_contents must be a boolean.")

    current_entries = {entry.path: entry for entry in current.entries}
    desired_entries = {entry.path: entry for entry in desired.entries}
    shared_paths = current_entries.keys() & desired_entries.keys()

    conflict_paths = {
        path for path in shared_paths
        if current_entries[path].kind != desired_entries[path].kind
    }
    conflicts = tuple(
        DiffEntry(
            path=path,
            current_kind=current_entries[path].kind,
            desired_kind=desired_entries[path].kind,
            reason="A file/directory replacement is not safe to automate.",
        )
        for path in sorted(conflict_paths)
    )

    def blocked_by_conflict(path: str) -> bool:
        return any(path.startswith(conflict + "/") for conflict in conflict_paths)

    added: list[DiffEntry] = []
    removed: list[DiffEntry] = []
    modified: list[DiffEntry] = []
    unchanged: list[DiffEntry] = []

    all_paths = sorted(current_entries.keys() | desired_entries.keys())
    for path in all_paths:
        if path in conflict_paths or blocked_by_conflict(path):
            continue
        before = current_entries.get(path)
        after = desired_entries.get(path)
        if before is None:
            added.append(DiffEntry(path, None, after.kind))
            continue
        if after is None:
            removed.append(DiffEntry(path, before.kind, None,
                                     "Suggested removal only; no deletion is performed."))
            continue

        if before.kind == "file" and compare_contents:
            if before.content != after.content:
                modified.append(DiffEntry(
                    path, before.kind, after.kind,
                    "Text content differs between the supplied snapshots.",
                ))
            else:
                unchanged.append(DiffEntry(path, before.kind, after.kind))
        else:
            unchanged.append(DiffEntry(path, before.kind, after.kind))

    return ProjectDiff(
        project_name=current.name,
        added=tuple(added),
        removed=tuple(removed),
        modified=tuple(modified),
        unchanged=tuple(unchanged),
        conflicts=conflicts,
        content_compared=compare_contents,
    )
