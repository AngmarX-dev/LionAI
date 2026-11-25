import unittest
from unittest.mock import patch

from training.import_github_metadata import classify, collect_repo


class GitHubMetadataImporterTests(unittest.TestCase):
    def test_classifies_common_real_file_types(self):
        cases = {
            ("photo.png", "assets/photo.png"): ("png", "Images"),
            ("report.pdf", "reports/report.pdf"): ("pdf", "Reports"),
            ("track.wav", "audio/track.wav"): ("wav", "Audio"),
            ("clip.mp4", "video/clip.mp4"): ("mp4", "Video"),
            ("archive.zip", "dist/archive.zip"): ("zip", "Archives"),
            ("settings.yaml", "config/settings.yaml"): ("yaml", "Config"),
            ("main.py", "src/main.py"): ("py", "Source"),
            ("font.woff2", "fonts/font.woff2"): ("woff2", "Fonts"),
            ("catalog.sqlite", "db/catalog.sqlite"): ("sqlite", "Databases"),
            ("setup.msi", "release/setup.msi"): ("msi", "Installers"),
            ("unknown.xyz", "misc/unknown.xyz"): ("xyz", "Other/Review"),
        }
        for (filename, path), expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(classify(filename, path), expected)

    def test_rejects_sensitive_names_and_paths(self):
        self.assertIsNone(classify("credentials.json", "config/credentials.json"))
        self.assertIsNone(classify("private-key.pem", "certs/private-key.pem"))
        self.assertIsNone(classify("settings.yaml", "secrets/settings.yaml"))

    def test_collects_only_git_tree_metadata_from_permissive_repo(self):
        repo_info = {
            "license": {"spdx_id": "MIT"},
            "default_branch": "main",
        }
        tree = {
            "sha": "abc123",
            "truncated": False,
            "tree": [
                {"path": "assets/photo.png", "type": "blob", "mode": "100644"},
                {"path": "docs/report.pdf", "type": "blob", "mode": "100644"},
                {"path": "src/main.py", "type": "blob", "mode": "100644"},
                {"path": "secrets/credentials.json", "type": "blob", "mode": "100644"},
                {"path": "src/folder", "type": "tree", "mode": "040000"},
            ],
        }
        with patch("training.import_github_metadata.api_get", side_effect=[repo_info, tree]) as api:
            rows, manifest = collect_repo("example/project", "token", 100)
        self.assertEqual(api.call_count, 2)
        self.assertEqual({row["filename"] for row in rows}, {"photo.png", "report.pdf", "main.py"})
        self.assertTrue(all(row["topic"] == "github:example/project" for row in rows))
        self.assertTrue(manifest["metadata_only"])
        self.assertEqual(manifest["license"], "MIT")

    def test_skips_repository_without_allowlisted_license(self):
        with patch("training.import_github_metadata.api_get", return_value={
            "license": {"spdx_id": "GPL-3.0"}, "default_branch": "main"
        }) as api:
            rows, manifest = collect_repo("example/copyleft", "", 100)
        self.assertEqual(rows, [])
        self.assertEqual(api.call_count, 1)
        self.assertEqual(manifest["status"], "skipped_license")


if __name__ == "__main__":
    unittest.main()
