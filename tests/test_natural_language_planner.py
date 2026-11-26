import unittest

from training.project_engine import ProjectStructureError
from training.project_intent import ProjectIntent, build_project_plan
from training.natural_language_planner import parse_project_request, plan_project_request


class NaturalLanguagePlannerTests(unittest.TestCase):
    def test_parses_persian_flask_shop_request(self):
        intent = parse_project_request(
            "یک فروشگاه با Flask و ورود کاربران و پایگاه داده و تست بساز",
            project_name="flask-shop",
        )
        self.assertEqual(intent.framework, "flask")
        self.assertEqual(intent.name, "flask-shop")
        self.assertEqual(set(intent.features), {"authentication", "database", "tests"})

    def test_parses_english_fastapi_request(self):
        intent = parse_project_request("Create a FastAPI service with database and tests")
        self.assertEqual(intent.framework, "fastapi")
        self.assertEqual(set(intent.features), {"database", "tests"})
        plan = plan_project_request("Create a FastAPI service with database and tests")
        paths = {entry.path for entry in plan.entries}
        self.assertIn("app/database.py", paths)
        requirements = next(entry.content for entry in plan.entries if entry.path == "requirements.txt")
        self.assertIn("SQLAlchemy", requirements)

    def test_parses_python_cli(self):
        plan = plan_project_request("یک ابزار خط فرمان Python با تست بساز")
        paths = {entry.path for entry in plan.entries}
        self.assertIn("src/main.py", paths)
        self.assertIn("tests/test_main.py", paths)

    def test_flask_defaults_are_rich_but_cli_defaults_are_valid(self):
        flask = build_project_plan(ProjectIntent("shop", "flask"))
        cli = build_project_plan(ProjectIntent("tool", "python-cli"))
        self.assertIn("app/auth/__init__.py", {entry.path for entry in flask.entries})
        self.assertIn("src/main.py", {entry.path for entry in cli.entries})

    def test_missing_framework_is_rejected(self):
        with self.assertRaises(ProjectStructureError):
            parse_project_request("یک پروژه فروشگاهی بساز")

    def test_multiple_frameworks_are_rejected(self):
        with self.assertRaises(ProjectStructureError):
            parse_project_request("Create a Flask and FastAPI project")

    def test_unsafe_project_name_is_rejected(self):
        with self.assertRaises(ProjectStructureError):
            parse_project_request("Create Flask app", project_name="../escape")

    def test_unknown_framework_is_not_guessed(self):
        with self.assertRaises(ProjectStructureError):
            parse_project_request("Create a Django project")

    def test_plan_is_read_only_and_never_auto_applies(self):
        plan = plan_project_request("Create a Flask app", project_name="safe-app")
        self.assertTrue(plan.preview().startswith("safe-app/"))


if __name__ == "__main__":
    unittest.main()
