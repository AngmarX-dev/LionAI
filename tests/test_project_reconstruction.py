import tempfile
import unittest
from pathlib import Path

from training.project_engine import ProjectStructureError
from training.project_reconstruction import (
    parse_tree_text,
    reconstruct_project,
    scan_directory,
)


class ProjectReconstructionTests(unittest.TestCase):
    def test_parse_unicode_tree_and_preserve_empty_directories(self):
        plan = parse_tree_text(
            "flask-shop/\n"
            "├── app/\n"
            "│   ├── routes.py\n"
            "│   └── templates/\n"
            "├── migrations/\n"
            "└── README.md\n"
        )
        kinds = {entry.path: entry.kind for entry in plan.entries}
        self.assertEqual(plan.name, "flask-shop")
        self.assertEqual(kinds["app"], "directory")
        self.assertEqual(kinds["app/templates"], "directory")
        self.assertEqual(kinds["migrations"], "directory")
        self.assertEqual(kinds["app/routes.py"], "file")
        self.assertIn("└── migrations/", plan.preview())

    def test_unmarked_parent_with_children_is_inferred_as_directory(self):
        plan = parse_tree_text(
            "demo/\\n"
            "└── src\\n"
            "    └── main.py\\n"
        )
        kinds = {entry.path: entry.kind for entry in plan.entries}
        self.assertEqual(kinds["src"], "directory")
        self.assertEqual(kinds["src/main.py"], "file")

    def test_parse_tree_can_use_explicit_project_name(self):
        plan = parse_tree_text("├── src/\n│   └── main.py\n", project_name="tool")
        self.assertEqual(plan.name, "tool")
        self.assertIn("src/main.py", {entry.path for entry in plan.entries})

    def test_rejects_invalid_tree_indentation_and_traversal(self):
        with self.assertRaises(ProjectStructureError):
            parse_tree_text("demo/\n│       └── file.py\n")
        with self.assertRaises(ProjectStructureError):
            parse_tree_text("demo/\n└── ../escape.py\n")

    def test_scan_directory_is_structure_only_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            (root / "empty").mkdir(parents=True)
            (root / "src").mkdir()
            (root / "src" / "main.py").write_text("print('hello')", encoding="utf-8")
            plan = scan_directory(root)
            entries = {entry.path: entry for entry in plan.entries}
            self.assertIn("empty", entries)
            self.assertEqual(entries["empty"].kind, "directory")
            self.assertEqual(entries["src/main.py"].content, "")

    def test_content_capture_is_opt_in_bounded_and_skips_secret_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "main.py").write_text("print('safe')", encoding="utf-8")
            (root / ".env").write_text("API_TOKEN=secret", encoding="utf-8")
            (root / "large.txt").write_text("x" * 32, encoding="utf-8")
            plan = scan_directory(root, include_text_contents=True, max_file_bytes=16)
            entries = {entry.path: entry for entry in plan.entries}
            self.assertEqual(entries["main.py"].content, "print('safe')")
            self.assertEqual(entries[".env"].content, "")
            self.assertEqual(entries["large.txt"].content, "")

    def test_scan_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "root"
            root.mkdir()
            outside = Path(temp) / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            link = root / "link.txt"
            try:
                link.symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("Symlinks are unavailable on this platform.")
            with self.assertRaises(ProjectStructureError):
                scan_directory(root)

    def test_scan_enforces_entry_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "b.txt").write_text("b", encoding="utf-8")
            with self.assertRaises(ProjectStructureError):
                scan_directory(root, max_entries=1)

    def test_reconstruction_never_writes_to_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "existing"
            root.mkdir()
            original = root / "keep.txt"
            original.write_text("keep", encoding="utf-8")
            before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
            plan = reconstruct_project(root)
            after = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
            self.assertEqual(before, after)
            self.assertIn("keep.txt", {entry.path for entry in plan.entries})


if __name__ == "__main__":
    unittest.main()
