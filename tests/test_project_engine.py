import tempfile
import unittest
from pathlib import Path

from training.project_engine import ProjectPlan, ProjectStructureError


class ProjectPlanTests(unittest.TestCase):
    def make_plan(self):
        return ProjectPlan.from_entries(
            "flask-shop",
            [
                {"path": "app", "kind": "directory"},
                {"path": "app/templates", "kind": "directory"},
                {"path": "app/static", "kind": "directory"},
                {"path": "app/static/css", "kind": "directory"},
                {"path": "migrations", "kind": "directory"},
                {"path": "app/__init__.py", "kind": "file", "content": ""},
                {"path": "app/routes.py", "kind": "file", "content": "from flask import Flask\n"},
                {"path": "app/templates/base.html", "kind": "file", "content": "<html></html>"},
            ],
        )

    def test_preview_preserves_nested_and_empty_directories(self):
        preview = self.make_plan().preview()
        self.assertIn("├── app/", preview)
        self.assertIn("│   ├── templates/", preview)
        self.assertIn("└── migrations/", preview)

    def test_preview_does_not_touch_filesystem(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = self.make_plan()
            plan.preview()
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_apply_requires_explicit_confirmation(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ProjectStructureError):
                self.make_plan().apply(temp)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_apply_creates_files_and_empty_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.make_plan().apply(temp, confirmed=True)
            self.assertTrue((root / "app/routes.py").is_file())
            self.assertEqual((root / "app/routes.py").read_text(), "from flask import Flask\n")
            self.assertTrue((root / "migrations").is_dir())
            self.assertEqual(list((root / "migrations").iterdir()), [])

    def test_existing_project_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            existing = Path(temp) / "flask-shop"
            existing.mkdir()
            marker = existing / "keep.txt"
            marker.write_text("keep")
            with self.assertRaises(ProjectStructureError):
                self.make_plan().apply(temp, confirmed=True)
            self.assertEqual(marker.read_text(), "keep")

    def test_rejects_path_traversal(self):
        with self.assertRaises(ProjectStructureError):
            ProjectPlan.from_entries("demo", [{"path": "../outside.txt", "kind": "file"}])

    def test_rejects_duplicate_paths(self):
        with self.assertRaises(ProjectStructureError):
            ProjectPlan.from_entries(
                "demo",
                [
                    {"path": "src", "kind": "directory"},
                    {"path": "src", "kind": "directory"},
                ],
            )

    def test_rejects_file_parent_conflict(self):
        with self.assertRaises(ProjectStructureError):
            ProjectPlan.from_entries(
                "demo",
                [
                    {"path": "src", "kind": "file"},
                    {"path": "src/main.py", "kind": "file"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
