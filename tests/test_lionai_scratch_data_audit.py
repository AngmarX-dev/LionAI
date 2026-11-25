import json
import tempfile
import unittest
from pathlib import Path

from training.audit_scratch_data import audit, load_dataset


class ScratchDatasetAuditTests(unittest.TestCase):
    def make_file(self, directory, name, rows):
        path = Path(directory) / name
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
        return path

    def test_rejects_windows_path_in_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_file(directory, "bad.jsonl", [{
                "topic": "test", "filename": r"..\\secret.txt",
                "extension": "txt", "label": "Documents",
            }])
            with self.assertRaises(ValueError):
                load_dataset(path)

    def test_rejects_extra_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_file(directory, "bad.jsonl", [{
                "topic": "test", "filename": "notes.txt",
                "extension": "txt", "label": "Documents", "contents": "not allowed",
            }])
            with self.assertRaises(ValueError):
                load_dataset(path)


    def test_rejects_mismatched_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_file(directory, "bad.jsonl", [{
                "topic": "test", "filename": "notes.pdf",
                "extension": "txt", "label": "Documents",
            }])
            with self.assertRaisesRegex(ValueError, "extension does not match"):
                load_dataset(path)


    def test_rejects_duplicate_evaluation_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = ["Images", "Documents", "Data", "Audio", "Video", "Archives",
                      "Source", "Config", "Installers", "Fonts", "Databases", "Reports", "Other/Review"]
            train = [{"topic": f"train topic {i}", "filename": f"train_{i}.dat",
                      "extension": "dat", "label": label} for i, label in enumerate(labels)]
            evaluation = [{"topic": f"eval topic {i}", "filename": f"eval_{i}.dat",
                           "extension": "dat", "label": label} for i, label in enumerate(labels)]
            evaluation.append(dict(evaluation[0]))
            train_path = self.make_file(directory, "train.jsonl", train)
            eval_path = self.make_file(directory, "eval.jsonl", evaluation)
            with self.assertRaisesRegex(ValueError, "Duplicate evaluation"):
                audit(train_path, eval_path)

    def test_rejects_train_eval_topic_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = ["Images", "Documents", "Data", "Audio", "Video", "Archives",
                      "Source", "Config", "Installers", "Fonts", "Databases", "Reports", "Other/Review"]
            train = [{"topic": "shared topic", "filename": f"train_{i}.dat",
                      "extension": "dat", "label": label} for i, label in enumerate(labels)]
            evaluation = [{"topic": "shared topic", "filename": f"eval_{i}.dat",
                           "extension": "dat", "label": label} for i, label in enumerate(labels)]
            train_path = self.make_file(directory, "train.jsonl", train)
            eval_path = self.make_file(directory, "eval.jsonl", evaluation)
            with self.assertRaisesRegex(ValueError, "topic leakage"):
                audit(train_path, eval_path)


if __name__ == "__main__":
    unittest.main()
