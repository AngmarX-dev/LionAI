import unittest

from training.import_hf_dataset import convert_records, map_label, normalize_record


class HuggingFaceImportTests(unittest.TestCase):
    def test_source_labels_map_to_lionai_classes(self):
        self.assertEqual(map_label("images", "jpg"), "Images")
        self.assertEqual(map_label("documents", "pdf"), "Documents")
        self.assertEqual(map_label("documents", "xlsx"), "Data")
        self.assertEqual(map_label("code", "yaml"), "Config")
        self.assertEqual(map_label("code", "py"), "Source")
        self.assertEqual(map_label("other", "msi"), "Installers")
        self.assertEqual(map_label("other", "woff2"), "Fonts")
        self.assertEqual(map_label("other", "sqlite"), "Databases")
        self.assertEqual(map_label("other", "unknown"), "Other/Review")

    def test_compound_extension_is_preserved(self):
        row = normalize_record({
            "file_name": "backup.tar.gz",
            "extension": "tar.gz",
            "folder_hint": "Backups",
            "label": "archives",
        }, 1)
        self.assertEqual(row["extension"], "tar.gz")

    def test_record_uses_only_metadata_and_plain_filename(self):
        row = normalize_record({
            "file_name": "notes.pdf",
            "extension": "pdf",
            "folder_hint": "Downloads/work",
            "label": "documents",
            "size_bytes": 1234,
        }, 1)
        self.assertEqual(row, {
            "topic": "Downloads/work",
            "filename": "notes.pdf",
            "extension": "pdf",
            "label": "Documents",
        })

    def test_duplicate_metadata_rows_are_removed(self):
        source = [
            {"file_name": "a.jpg", "extension": "jpg", "folder_hint": "Pictures", "label": "images"},
            {"file_name": "a.jpg", "extension": "jpg", "folder_hint": "Pictures", "label": "images"},
            {"file_name": "a.jpg", "extension": "jpg", "folder_hint": "Downloads", "label": "images"},
        ]
        rows, report = convert_records(source)
        self.assertEqual(len(rows), 2)
        self.assertEqual(report["duplicates_skipped"], 1)

    def test_conflicting_duplicate_labels_are_rejected(self):
        source = [
            {"file_name": "same.bin", "extension": "bin", "folder_hint": "Downloads", "label": "other"},
            {"file_name": "same.bin", "extension": "bin", "folder_hint": "downloads", "label": "documents"},
        ]
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            convert_records(source)

    def test_path_like_filename_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_record({
                "file_name": "../private.txt",
                "extension": "txt",
                "folder_hint": "Downloads",
                "label": "documents",
            }, 1)

    def test_control_characters_and_oversized_topics_are_rejected(self):
        base = {"file_name": "notes.txt", "extension": "txt", "label": "documents"}
        with self.assertRaisesRegex(ValueError, "control characters"):
            normalize_record({**base, "folder_hint": "Downloads\nPrivate"}, 1)
        with self.assertRaisesRegex(ValueError, "200 characters"):
            normalize_record({**base, "folder_hint": "x" * 201}, 1)


if __name__ == "__main__":
    unittest.main()
