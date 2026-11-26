import tempfile
import unittest
from pathlib import Path

from training.project_diff import compare_project_plans
from training.project_engine import ProjectPlan, ProjectStructureError
from training.project_sync import build_sync_plan


class ProjectSyncPlannerTests(unittest.TestCase):
    def plan(self, entries):
        return ProjectPlan.from_entries("demo", entries)

    def compare(self, current, desired, *, contents=False):
        return compare_project_plans(current, desired, compare_contents=contents)

    def test_orders_creations_before_files_and_updates(self):
        current = self.plan([])
        desired = self.plan([
            {"path": "src", "kind": "directory"},
            {"path": "src/new.py", "kind": "file", "content": "new"},
            {"path": "README.md", "kind": "file", "content": "readme"},
        ])
        sync = build_sync_plan(current, desired, self.compare(current, desired))
        self.assertEqual(
            [(a.operation, a.path) for a in sync.actions],
            [
                ("create_directory", "src"),
                ("create_file", "README.md"),
                ("create_file", "src/new.py"),
            ],
        )

    def test_modified_file_becomes_update_action_when_content_diff_enabled(self):
        current = self.plan([{"path": "main.py", "kind": "file", "content": "old"}])
        desired = self.plan([{"path": "main.py", "kind": "file", "content": "new"}])
        diff = self.compare(current, desired, contents=True)
        sync = build_sync_plan(current, desired, diff)
        self.assertEqual(len(sync.actions), 1)
        self.assertEqual(sync.actions[0].operation, "update_file")
        self.assertEqual(sync.actions[0].content, "new")

    def test_removals_are_excluded_by_default(self):
        current = self.plan([
            {"path": "old", "kind": "directory"},
            {"path": "old/file.txt", "kind": "file", "content": "x"},
        ])
        desired = self.plan([])
        sync = build_sync_plan(current, desired, self.compare(current, desired))
        self.assertEqual(sync.actions, ())
        self.assertEqual(sync.pending_removals, ("old/file.txt", "old"))
        self.assertFalse(sync.requires_destructive_approval)

    def test_deletions_are_opt_in_and_remain_inert_proposals(self):
        current = self.plan([
            {"path": "old", "kind": "directory"},
            {"path": "old/file.txt", "kind": "file", "content": "x"},
        ])
        desired = self.plan([])
        sync = build_sync_plan(
            current, desired, self.compare(current, desired), include_deletions=True
        )
        self.assertEqual(
            [(a.operation, a.path) for a in sync.actions],
            [("delete_file", "old/file.txt"), ("delete_directory", "old")],
        )
        self.assertTrue(all(a.destructive for a in sync.actions))
        self.assertTrue(sync.requires_destructive_approval)
        self.assertIn("no filesystem changes have been made", sync.preview())

    def test_conflicts_are_reported_but_never_actioned(self):
        current = self.plan([{"path": "node", "kind": "file", "content": "x"}])
        desired = self.plan([{"path": "node", "kind": "directory"}])
        sync = build_sync_plan(current, desired, self.compare(current, desired))
        self.assertEqual(sync.actions, ())
        self.assertEqual(sync.conflicts, ("node",))

    def test_stale_diff_is_rejected(self):
        current = self.plan([])
        desired = self.plan([{"path": "one.txt", "kind": "file"}])
        unrelated = self.compare(current, self.plan([]))
        with self.assertRaises(ProjectStructureError):
            build_sync_plan(current, desired, unrelated)

    def test_project_names_must_match(self):
        current = ProjectPlan.from_entries("one", [])
        desired = ProjectPlan.from_entries("two", [])
        diff = compare_project_plans(current, current)
        with self.assertRaises(ProjectStructureError):
            build_sync_plan(current, desired, diff)

    def test_planning_does_not_modify_filesystem(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            current = self.plan([{"path": "keep.txt", "kind": "file"}])
            desired = self.plan([{"path": "new.txt", "kind": "file"}])
            sync = build_sync_plan(current, desired, self.compare(current, desired))
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertEqual(sorted(p.name for p in Path(temp).iterdir()), ["keep.txt"])
            self.assertEqual(sync.pending_removals, ("keep.txt",))


if __name__ == "__main__":
    unittest.main()
