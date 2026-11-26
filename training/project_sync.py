"""Build explicit, review-only synchronization operations from a ProjectDiff.

No filesystem operations are performed here. In particular, deletion proposals
are opt-in and remain inert data until a future separately-confirmed executor.
"""
from __future__ import annotations

from dataclasses import dataclass

from training.project_diff import ProjectDiff
from training.project_engine import Entry, ProjectStructureError, ProjectPlan


@dataclass(frozen=True)
class SyncAction:
    """A proposed operation; this object does not execute the operation."""

    operation: str
    path: str
    kind: str
    content: str = ""
    destructive: bool = False
    expected_content: str | None = None


@dataclass(frozen=True)
class ProjectSyncPlan:
    """Immutable synchronization proposal requiring external review."""

    project_name: str
    actions: tuple[SyncAction, ...]
    pending_removals: tuple[str, ...]
    conflicts: tuple[str, ...]

    @property
    def has_actions(self) -> bool:
        return bool(self.actions)

    @property
    def requires_destructive_approval(self) -> bool:
        return any(action.destructive for action in self.actions)

    def summary(self) -> dict[str, int | bool]:
        return {
            "actions": len(self.actions),
            "create_directories": sum(a.operation == "create_directory" for a in self.actions),
            "create_files": sum(a.operation == "create_file" for a in self.actions),
            "update_files": sum(a.operation == "update_file" for a in self.actions),
            "delete_files": sum(a.operation == "delete_file" for a in self.actions),
            "delete_directories": sum(a.operation == "delete_directory" for a in self.actions),
            "pending_removals": len(self.pending_removals),
            "conflicts": len(self.conflicts),
            "requires_destructive_approval": self.requires_destructive_approval,
        }

    def preview(self) -> str:
        lines = [f"Synchronization plan: {self.project_name}"]
        if not self.actions:
            lines.append("Actions: none")
        else:
            lines.append(f"Actions: {len(self.actions)}")
            for action in self.actions:
                marker = " [DESTRUCTIVE — explicit approval required]" if action.destructive else ""
                lines.append(f"  {action.operation}: {action.path} ({action.kind}){marker}")
        lines.append(f"Pending removals not included: {len(self.pending_removals)}")
        for path in self.pending_removals:
            lines.append(f"  review removal: {path}")
        lines.append(f"Unresolved conflicts: {len(self.conflicts)}")
        for path in self.conflicts:
            lines.append(f"  manual resolution: {path}")
        lines.append("This is a preview only; no filesystem changes have been made.")
        return "\n".join(lines)


def build_sync_plan(
    current: ProjectPlan,
    desired: ProjectPlan,
    diff: ProjectDiff,
    *,
    include_deletions: bool = False,
) -> ProjectSyncPlan:
    """Convert a previously computed diff into an ordered, inert action plan.

    New directories are proposed before their children; files are created after
    directories; updates follow creations; removals are ordered deepest-first.
    Deletion proposals are excluded unless include_deletions=True. Even when
    included, they are only descriptive actions and are never executed.
    """
    if not isinstance(current, ProjectPlan) or not isinstance(desired, ProjectPlan):
        raise ProjectStructureError("Current and desired must be ProjectPlan instances.")
    if not isinstance(diff, ProjectDiff):
        raise ProjectStructureError("diff must be a ProjectDiff instance.")
    if current.name != desired.name or diff.project_name != current.name:
        raise ProjectStructureError("Current, desired, and diff project names must match.")
    if not isinstance(include_deletions, bool):
        raise ProjectStructureError("include_deletions must be a boolean.")

    current_by_path = {entry.path: entry for entry in current.entries}
    desired_by_path = {entry.path: entry for entry in desired.entries}

    # Reject stale or unrelated diff objects instead of silently planning
    # operations against a different pair of manifests.
    expected = {
        "added": {path for path in desired_by_path if path not in current_by_path},
        "removed": {path for path in current_by_path if path not in desired_by_path},
        "conflicts": {
            path for path in current_by_path.keys() & desired_by_path.keys()
            if current_by_path[path].kind != desired_by_path[path].kind
        },
    }
    shared_same_kind = {
        path for path in current_by_path.keys() & desired_by_path.keys()
        if current_by_path[path].kind == desired_by_path[path].kind
    }
    expected["modified"] = {
        path for path in shared_same_kind
        if diff.content_compared
        and current_by_path[path].kind == "file"
        and current_by_path[path].content != desired_by_path[path].content
    }
    expected["unchanged"] = shared_same_kind - expected["modified"]
    actual = {
        "added": {item.path for item in diff.added},
        "removed": {item.path for item in diff.removed},
        "modified": {item.path for item in diff.modified},
        "unchanged": {item.path for item in diff.unchanged},
        "conflicts": {item.path for item in diff.conflicts},
    }
    # Diff intentionally omits descendants below a type conflict.
    for key in ("added", "removed", "modified", "unchanged"):
        blocked = {
            path for conflict in expected["conflicts"]
            for path in expected[key] if path.startswith(conflict + "/")
        }
        expected[key] -= blocked
    if any(actual[key] != expected[key] for key in expected):
        raise ProjectStructureError("Diff does not match the supplied project plans.")

    conflict_paths = tuple(sorted(actual["conflicts"]))
    added_entries = [desired_by_path[path] for path in actual["added"]]
    added_dirs = sorted(
        (entry for entry in added_entries if entry.kind == "directory"),
        key=lambda entry: (entry.path.count("/"), entry.path),
    )
    added_files = sorted(
        (entry for entry in added_entries if entry.kind == "file"),
        key=lambda entry: entry.path,
    )
    actions: list[SyncAction] = []
    actions.extend(
        SyncAction("create_directory", entry.path, "directory")
        for entry in added_dirs
    )
    actions.extend(
        SyncAction("create_file", entry.path, "file", entry.content)
        for entry in added_files
    )

    modified_paths = {item.path for item in diff.modified}
    for path in sorted(modified_paths):
        entry = desired_by_path.get(path)
        if entry is None or entry.kind != "file":
            raise ProjectStructureError("Modified entries must map to desired files.")
        actions.append(SyncAction(
            "update_file", path, "file", entry.content,
            expected_content=current_by_path[path].content,
        ))

    removed_paths = sorted(
        actual["removed"],
        key=lambda path: (-path.count("/"), path),
    )
    pending_removals = () if include_deletions else tuple(removed_paths)
    if include_deletions:
        for path in removed_paths:
            entry = current_by_path[path]
            operation = "delete_directory" if entry.kind == "directory" else "delete_file"
            actions.append(SyncAction(operation, path, entry.kind, destructive=True))

    return ProjectSyncPlan(
        project_name=current.name,
        actions=tuple(actions),
        pending_removals=pending_removals,
        conflicts=conflict_paths,
    )
