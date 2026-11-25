import json
import tempfile
import unittest
from pathlib import Path

from training.evaluate_scratch import read_rows


class ScratchEvaluationDataTests(unittest.TestCase):
    def write_rows(self, rows):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "examples.jsonl"
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
        return path

    def test_accepts_metadata_only_rows(self):
        path = self.write_rows([{
            "topic": "موضوع آزمایشی", "filename": "notes.pdf",
            "extension": ".pdf", "label": "Documents",
        }])
        self.assertEqual(read_rows(path)[0]["filename"], "notes.pdf")

    def test_rejects_file_paths_in_filename_field(self):
        path = self.write_rows([{
            "topic": "test", "filename": "../notes.pdf",
            "extension": ".pdf", "label": "Documents",
        }])
        with self.assertRaises(ValueError):
            read_rows(path)

    def test_rejects_extra_fields_that_could_contain_file_content(self):
        path = self.write_rows([{
            "topic": "test", "filename": "notes.pdf", "extension": ".pdf",
            "label": "Documents", "content": "must not be accepted",
        }])
        with self.assertRaises(ValueError):
            read_rows(path)

    def test_rejects_unknown_labels(self):
        path = self.write_rows([{
            "topic": "test", "filename": "notes.pdf",
            "extension": ".pdf", "label": "UnknownClass",
        }])
        with self.assertRaises(ValueError):
            read_rows(path)


if __name__ == "__main__":
    unittest.main()
