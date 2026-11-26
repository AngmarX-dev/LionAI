import tempfile
import unittest
from pathlib import Path

from training.project_engine import ProjectStructureError
from training.project_intent import ProjectIntent, build_project_plan


class ProjectIntentTests(unittest.TestCase):
    def test_flask_template_builds_expected_nested_tree(self):
        plan = build_project_plan(
            ProjectIntent(
                name="flask-shop",
                framework="flask",
                features=("tests", "templates", "static", "database"),
            )
        )
        preview = plan.preview()
        for expected in (
            "flask-shop/",
            "app/",
            "routes.py",
            "templates/",
            "base.html",
            "static/",
            "style.css",
            "main.js",
            "images/",
            "migrations/",
            "tests/",
            "requirements.txt",
            "run.py",
        ):
            self.assertIn(expected, preview)

    def test_fastapi_template_has_application_entrypoint(self):
        plan = build_project_plan(ProjectIntent("api-service", "fastapi"))
        paths = {entry.path for entry in plan.entries}
        self.assertIn("app/main.py", paths)
        self.assertIn("requirements.txt", paths)
        self.assertIn("tests/test_main.py", paths)

    def test_python_cli_template_has_pyproject_and_main(self):
        plan = build_project_plan(ProjectIntent("hello-cli", "python-cli"))
        paths = {entry.path for entry in plan.entries}
        self.assertIn("src/main.py", paths)
        self.assertIn("pyproject.toml", paths)

    def test_unknown_framework_is_rejected(self):
        with self.assertRaises(ProjectStructureError):
            build_project_plan(ProjectIntent("demo", "django"))

    def test_unknown_feature_is_rejected(self):
        with self.assertRaises(ProjectStructureError):
            build_project_plan(ProjectIntent("demo", "flask", ("tests", "unknown")))

    def test_duplicate_features_are_rejected(self):
        with self.assertRaises(ProjectStructureError):
            build_project_plan(ProjectIntent("demo", "flask", ("tests", "tests")))

    def test_authentication_is_rejected_for_cli(self):
        with self.assertRaises(ProjectStructureError):
            build_project_plan(ProjectIntent("demo", "python-cli", ("authentication",)))

    def test_planning_is_read_only_and_apply_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = build_project_plan(ProjectIntent("demo", "flask"))
            self.assertEqual(list(Path(temp).iterdir()), [])
            with self.assertRaises(ProjectStructureError):
                plan.apply(temp)
            self.assertEqual(list(Path(temp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
