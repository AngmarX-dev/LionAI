"""Explicitly confirmed, rollback-aware executor for safe project sync plans.

This executor supports creation and text-file updates. Destructive operations
are deliberately rejected in this milestone; they require a separate,
transactional deletion implementation and confirmation path.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from training.project_engine import ProjectStructureError
from training.project_sync import ProjectSyncPlan, SyncAction


class ProjectSyncExecutionError(ProjectStructureError):
    """Raised when execution is unsafe, stale, or cannot be completed."""


@dataclass(frozen=True)
class ProjectSyncResult:
    project_root: Path
    created_directories: tuple[str, ...]
    created_files: tuple[str, ...]
    updated_files: tuple[str, ...]
    rolled_back: bool = False


def _relative_parts(value: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ProjectSyncExecutionError("Action path must be a safe relative POSIX path.")
    parsed = PurePosixPath(value)
    parts = parsed.parts
    if parsed.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
        raise ProjectSyncExecutionError(f"Unsafe action path: {value!r}")
    return parts


def _read_text_exact(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _check_no_symlink_components(root: Path, parts: tuple[str, ...], *, include_leaf: bool) -> Path:
    current = root
    limit = len(parts) if include_leaf else len(parts) - 1
    for part in parts[:limit]:
        current = current / part
        if current.is_symlink():
            raise ProjectSyncExecutionError(f"Symlink component is not allowed: {current}")
    return root.joinpath(*parts)


def execute_sync_plan(
    plan: ProjectSyncPlan,
    project_root: str | Path,
    *,
    confirmed: bool = False,
) -> ProjectSyncResult:
    """Apply non-destructive actions after explicit confirmation.

    All paths and expected update contents are preflighted before the first
    mutation. Updates use same-directory temporary files and atomic replacement.
    If an operation fails, changes made by this invocation are rolled back on a
    best-effort basis. The executor never follows symlinks intentionally.

    Deletions are not supported: any delete action is rejected even when marked
    destructive. This avoids turning a general project plan into a data-loss
    operation.
    """
    if not isinstance(plan, ProjectSyncPlan):
        raise ProjectSyncExecutionError("plan must be a ProjectSyncPlan.")
    if confirmed is not True:
        raise ProjectSyncExecutionError("Explicit user confirmation is required.")
    if plan.conflicts:
        raise ProjectSyncExecutionError("Resolve all file/directory conflicts before execution.")
    if plan.requires_destructive_approval:
        raise ProjectSyncExecutionError(
            "Destructive actions are not supported by this executor; no changes were made."
        )
    if plan.pending_removals:
        # Pending removals are informational and are not acted upon.
        pass

    root = Path(project_root).expanduser()
    if root.is_symlink() or not root.exists() or not root.is_dir():
        raise ProjectSyncExecutionError("Project root must be an existing, real directory.")
    root = root.resolve(strict=True)

    allowed = {"create_directory", "create_file", "update_file"}
    seen: set[str] = set()
    parsed_actions: list[tuple[SyncAction, tuple[str, ...]]] = []
    for action in plan.actions:
        if not isinstance(action, SyncAction) or action.operation not in allowed:
            raise ProjectSyncExecutionError("Plan contains an unsupported or malformed action.")
        if action.kind != ("directory" if action.operation == "create_directory" else "file"):
            raise ProjectSyncExecutionError("Action kind does not match its operation.")
        parts = _relative_parts(action.path)
        if action.path in seen:
            raise ProjectSyncExecutionError(f"Duplicate action path: {action.path}")
        seen.add(action.path)
        if not isinstance(action.content, str):
            raise ProjectSyncExecutionError("Action content must be text.")
        if action.operation == "update_file" and action.expected_content is None:
            raise ProjectSyncExecutionError(
                f"Update requires a captured original content snapshot: {action.path}"
            )
        parsed_actions.append((action, parts))

    directory_actions = {
        action.path for action, _ in parsed_actions
        if action.operation == "create_directory"
    }

    # Preflight the complete plan before mutating anything.
    for action, parts in parsed_actions:
        target = _check_no_symlink_components(root, parts, include_leaf=True)
        parent = target.parent
        parent_parts = parts[:-1]
        if parent_parts:
            _check_no_symlink_components(root, parent_parts, include_leaf=True)
            parent_is_planned = any(
                "/".join(parent_parts[:index]) in directory_actions
                for index in range(1, len(parent_parts) + 1)
            )
            if not parent.exists() and not parent_is_planned:
                raise ProjectSyncExecutionError(f"Parent directory does not exist: {parent}")
            if parent.exists() and not parent.is_dir():
                raise ProjectSyncExecutionError(f"Parent is not a directory: {parent}")

        if action.operation in {"create_directory", "create_file"}:
            if target.exists() or target.is_symlink():
                raise ProjectSyncExecutionError(f"Creation target already exists: {target}")
        else:
            if target.is_symlink() or not target.exists() or not target.is_file():
                raise ProjectSyncExecutionError(f"Update target is not a regular file: {target}")
            actual = _read_text_exact(target)
            if actual != action.expected_content:
                raise ProjectSyncExecutionError(
                    f"Stale update target; contents changed since planning: {action.path}"
                )

    created_dirs: list[Path] = []
    created_files: list[Path] = []
    updated_originals: list[tuple[Path, str]] = []
    updated_names: list[str] = []
    try:
        for action, parts in parsed_actions:
            target = _check_no_symlink_components(root, parts, include_leaf=True)
            if action.operation == "create_directory":
                target.mkdir()
                created_dirs.append(target)
            elif action.operation == "create_file":
                with target.open("x", encoding="utf-8", newline="") as handle:
                    handle.write(action.content)
                created_files.append(target)
            else:
                original = _read_text_exact(target)
                if original != action.expected_content:
                    raise ProjectSyncExecutionError(
                        f"Stale update target detected during execution: {action.path}"
                    )
                fd, temp_name = tempfile.mkstemp(prefix=".lionai-sync-", dir=str(target.parent))
                os.close(fd)
                temp_path = Path(temp_name)
                try:
                    with temp_path.open("w", encoding="utf-8", newline="") as handle:
                        handle.write(action.content)
                    if target.is_symlink() or not target.is_file():
                        raise ProjectSyncExecutionError(
                            f"Update target changed type during execution: {action.path}"
                        )
                    os.replace(temp_path, target)
                finally:
                    try:
                        temp_path.unlink()
                    except FileNotFoundError:
                        pass
                updated_originals.append((target, original))
                updated_names.append(action.path)
    except Exception as error:
        rollback_errors: list[str] = []
        for target, original in reversed(updated_originals):
            try:
                fd, temp_name = tempfile.mkstemp(prefix=".lionai-rollback-", dir=str(target.parent))
                os.close(fd)
                temp_path = Path(temp_name)
                try:
                    with temp_path.open("w", encoding="utf-8", newline="") as handle:
                        handle.write(original)
                    os.replace(temp_path, target)
                finally:
                    try:
                        temp_path.unlink()
                    except FileNotFoundError:
                        pass
            except OSError as rollback_error:
                rollback_errors.append(f"{target}: {rollback_error}")
        for target in reversed(created_files):
            try:
                target.unlink()
            except OSError as rollback_error:
                rollback_errors.append(f"{target}: {rollback_error}")
        for target in sorted(created_dirs, key=lambda item: len(item.parts), reverse=True):
            try:
                target.rmdir()
            except OSError as rollback_error:
                rollback_errors.append(f"{target}: {rollback_error}")
        detail = f"Execution failed and rollback was attempted: {error}"
        if rollback_errors:
            detail += "; rollback issues: " + "; ".join(rollback_errors)
        raise ProjectSyncExecutionError(detail) from error

    return ProjectSyncResult(
        project_root=root,
        created_directories=tuple(
            str(path.relative_to(root).as_posix()) for path in created_dirs
        ),
        created_files=tuple(
            str(path.relative_to(root).as_posix()) for path in created_files
        ),
        updated_files=tuple(updated_names),
        rolled_back=False,
    )
