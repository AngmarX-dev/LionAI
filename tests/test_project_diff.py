import tempfile
import unittest
from pathlib import Path

from training.project_engine import ProjectPlan, ProjectStructureError
from training.project_diff import compare_project_plans


class ProjectDiffTests(unittest.TestCase):
    def plan(self, entries):
        return ProjectPlan.from_entries("demo", entries)

    def test_reports_added_removed_modified_and_unchanged_paths(self):
        current = self.plan([
            {"path": "src", "kind": "directory"},
            {"path": "src/keep.py", "kind": "file", "content": "old"},
            {"path": "src/change.py", "kind": "file", "content": "before"},
            {"path": "obsolete.txt", "kind": "file", "content": "gone"},
        ])
        desired = self.plan([
            {"path": "src", "kind": "directory"},
            {"path": "src/keep.py", "kind": "file", "content": "old"},
            {"path": "src/change.py", "kind": "file", "content": "after"},
            {"path": "src/new.py", "kind": "file", "content": "new"},
        ])

        diff = compare_project_plans(current, desired, compare_contents=True)
        self.assertEqual([item.path for item in diff.added], ["src/new.py"])
        self.assertEqual([item.path for item in diff.removed], ["obsolete.txt"])
        self.assertEqual([item.path for item in diff.modified], ["src/change.py"])
        self.assertEqual([item.path for item in diff.unchanged], ["src", "src/keep.py"])
        self.assertTrue(diff.has_changes)

    def test_structure_only_comparison_does_not_claim_content_equality(self):
        current = self.plan([
            {"path": "main.py", "kind": "file", "content": "current"},
        ])
        desired = self.plan([
            {"path": "main.py", "kind": "file", "content": "desired"},
        ])
        diff = compare_project_plans(current, desired)
        self.assertFalse(diff.content_compared)
        self.assertEqual([item.path for item in diff.unchanged], ["main.py"])
        self.assertEqual(diff.modified, ())

    def test_kind_conflict_is_reported_and_descendants_are_not_actioned(self):
        current = self.plan([
            {"path": "src", "kind": "file", "content": "file"},
        ])
        desired = self.plan([
            {"path": "src", "kind": "directory"},
            {"path": "src/main.py", "kind": "file", "content": "code"},
        ])
        diff = compare_project_plans(current, desired)
        self.assertEqual([item.path for item in diff.conflicts], ["src"])
        self.assertEqual(diff.added, ())
        self.assertEqual(diff.removed, ())

    def test_removed_entries_are_review_only(self):
        current = self.plan([
            {"path": "keep.txt", "kind": "file", "content": "keep"},
            {"path": "remove.txt", "kind": "file", "content": "remove"},
        ])
        desired = self.plan([
            {"path": "keep.txt", "kind": "file", "content": "keep"},
        ])
        diff = compare_project_plans(current, desired)
        self.assertIn("no deletion is performed", diff.removed[0].reason)
        self.assertIn("never auto-deleted", diff.preview())

    def test_project_names_must_match(self):
        current = ProjectPlan.from_entries("one", [])
        desired = ProjectPlan.from_entries("two", [])
        with self.assertRaises(ProjectStructureError):
            compare_project_plans(current, desired)

    def test_diff_is_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            marker = root / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            current = self.plan([{"path": "keep.txt", "kind": "file"}])
            desired = self.plan([{"path": "new.txt", "kind": "file"}])
            compare_project_plans(current, desired)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["keep.txt"])


if __name__ == "__main__":
    unittest.main()
