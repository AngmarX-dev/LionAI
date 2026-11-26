import tempfile
import unittest
from pathlib import Path

from training.project_diff import compare_project_plans
from training.project_engine import ProjectPlan, ProjectStructureError
from training.project_sync import ProjectSyncPlan, SyncAction, build_sync_plan
from training.project_sync_executor import (
    ProjectSyncExecutionError,
    execute_sync_plan,
)


class ProjectSyncExecutorTests(unittest.TestCase):
    def plan(self, entries):
        return ProjectPlan.from_entries("demo", entries)

    def sync(self, current, desired, *, compare_contents=False, include_deletions=False):
        diff = compare_project_plans(current, desired, compare_contents=compare_contents)
        return build_sync_plan(
            current, desired, diff, include_deletions=include_deletions
        )

    def test_requires_exact_explicit_confirmation(self):
        with tempfile.TemporaryDirectory() as temp:
            current = self.plan([])
            desired = self.plan([{"path": "hello.txt", "kind": "file", "content": "hi"}])
            sync = self.sync(current, desired)
            for value in (False, None, 1, "yes"):
                with self.subTest(value=value):
                    with self.assertRaises(ProjectSyncExecutionError):
                        execute_sync_plan(sync, temp, confirmed=value)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_creates_directories_and_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "demo"
            root.mkdir()
            current = self.plan([])
            desired = self.plan([
                {"path": "src", "kind": "directory"},
                {"path": "src/main.py", "kind": "file", "content": "print('ok')\n"},
            ])
            result = execute_sync_plan(self.sync(current, desired), root, confirmed=True)
            self.assertEqual((root / "src" / "main.py").read_text(encoding="utf-8"), "print('ok')\n")
            self.assertEqual(result.created_directories, ("src",))
            self.assertEqual(result.created_files, ("src/main.py",))

    def test_updates_only_when_current_content_matches_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "main.py"
            target.write_text("old", encoding="utf-8")
            current = self.plan([{"path": "main.py", "kind": "file", "content": "old"}])
            desired = self.plan([{"path": "main.py", "kind": "file", "content": "new"}])
            sync = self.sync(current, desired, compare_contents=True)
            result = execute_sync_plan(sync, root, confirmed=True)
            self.assertEqual(target.read_text(encoding="utf-8"), "new")
            self.assertEqual(result.updated_files, ("main.py",))

    def test_stale_update_aborts_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "main.py"
            target.write_text("changed after planning", encoding="utf-8")
            current = self.plan([
                {"path": "src", "kind": "directory"},
                {"path": "main.py", "kind": "file", "content": "planned original"},
            ])
            desired = self.plan([
                {"path": "src", "kind": "directory"},
                {"path": "src/new.py", "kind": "file", "content": "new"},
                {"path": "main.py", "kind": "file", "content": "updated"},
            ])
            sync = self.sync(current, desired, compare_contents=True)
            with self.assertRaises(ProjectSyncExecutionError):
                execute_sync_plan(sync, root, confirmed=True)
            self.assertFalse((root / "src").exists())
            self.assertEqual(target.read_text(encoding="utf-8"), "changed after planning")

    def test_rejects_delete_actions_even_if_plan_marks_them_destructive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "old.txt"
            target.write_text("keep", encoding="utf-8")
            sync = ProjectSyncPlan(
                project_name="demo",
                actions=(SyncAction("delete_file", "old.txt", "file", destructive=True),),
                pending_removals=(),
                conflicts=(),
            )
            with self.assertRaises(ProjectSyncExecutionError):
                execute_sync_plan(sync, root, confirmed=True)
            self.assertEqual(target.read_text(encoding="utf-8"), "keep")

    def test_rejects_conflicts_and_symlink_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sync = ProjectSyncPlan("demo", (), (), ("node",))
            with self.assertRaises(ProjectSyncExecutionError):
                execute_sync_plan(sync, root, confirmed=True)
            link = root / "link"
            target = root / "target"
            target.mkdir()
            try:
                link.symlink_to(target, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("Symlinks are unavailable on this platform")
            empty = ProjectSyncPlan("demo", (), (), ())
            with self.assertRaises(ProjectSyncExecutionError):
                execute_sync_plan(empty, link, confirmed=True)

    def test_rejects_unsafe_paths_and_duplicate_actions(self):
        with tempfile.TemporaryDirectory() as temp:
            for actions in (
                (SyncAction("create_file", "../escape.txt", "file", "x"),),
                (
                    SyncAction("create_file", "same.txt", "file", "a"),
                    SyncAction("create_file", "same.txt", "file", "b"),
                ),
            ):
                sync = ProjectSyncPlan("demo", actions, (), ())
                with self.assertRaises(ProjectSyncExecutionError):
                    execute_sync_plan(sync, temp, confirmed=True)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_update_requires_captured_original_content(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "file.txt"
            target.write_text("original", encoding="utf-8")
            sync = ProjectSyncPlan(
                "demo",
                (SyncAction("update_file", "file.txt", "file", "new"),),
                (),
                (),
            )
            with self.assertRaises(ProjectSyncExecutionError):
                execute_sync_plan(sync, temp, confirmed=True)
            self.assertEqual(target.read_text(encoding="utf-8"), "original")


if __name__ == "__main__":
    unittest.main()
