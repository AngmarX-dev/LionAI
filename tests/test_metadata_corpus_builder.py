import tempfile
import unittest
from pathlib import Path

from training.build_metadata_corpus import merge_corpora


class MetadataCorpusBuilderTests(unittest.TestCase):
    def write_rows(self, root, name, rows):
        path = root / name
        path.write_text("".join(
            __import__("json").dumps(row) + "\n" for row in rows
        ), encoding="utf-8")
        return path

    def test_merges_sources_and_skips_exact_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"topic": "Pictures", "filename": "a.jpg", "extension": "jpg", "label": "Images"}
            first = self.write_rows(root, "one.jsonl", [row])
            second = self.write_rows(root, "two.jsonl", [row, {
                "topic": "Work", "filename": "notes.pdf", "extension": "pdf", "label": "Documents"
            }])
            merged, report = merge_corpora([first, second])
            self.assertEqual(len(merged), 2)
            self.assertEqual(report["duplicate_rows_skipped"], 1)
            self.assertEqual(report["class_counts"]["Images"], 1)
            self.assertEqual(report["class_counts"]["Documents"], 1)
            self.assertFalse(report["file_contents_used"])

    def test_conflicting_duplicate_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.write_rows(root, "one.jsonl", [{
                "topic": "Downloads", "filename": "same.bin", "extension": "bin", "label": "Other/Review"
            }])
            second = self.write_rows(root, "two.jsonl", [{
                "topic": "downloads", "filename": "same.bin", "extension": "bin", "label": "Documents"
            }])
            with self.assertRaisesRegex(ValueError, "Conflicting labels"):
                merge_corpora([first, second])

    def test_same_filename_in_different_topics_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.write_rows(root, "one.jsonl", [
                {"topic": "School", "filename": "notes.txt", "extension": "txt", "label": "Documents"},
                {"topic": "Work", "filename": "notes.txt", "extension": "txt", "label": "Documents"},
            ])
            merged, report = merge_corpora([first])
            self.assertEqual(len(merged), 2)
            self.assertEqual(report["topics_total"], 2)


if __name__ == "__main__":
    unittest.main()
