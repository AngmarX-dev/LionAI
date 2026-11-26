"""Deterministic project intent schema and curated starter templates.

Natural-language parsing is intentionally outside this module. An AI planner or
UI may produce a ProjectIntent, but this layer validates it and selects a
reviewable template before any filesystem operation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

from training.project_engine import ProjectPlan, ProjectStructureError


SUPPORTED_FRAMEWORKS = {"flask", "fastapi", "python-cli"}
SUPPORTED_FEATURES = {
    "tests",
    "templates",
    "static",
    "database",
    "authentication",
}


@dataclass(frozen=True)
class ProjectIntent:
    name: str
    framework: str
    features: tuple[str, ...] = ("tests", "templates", "static", "database", "authentication")


def _entry(path: str, kind: str, content: str = "") -> dict[str, str]:
    result = {"path": path, "kind": kind}
    if kind == "file":
        result["content"] = content
    return result


def _validate(intent: ProjectIntent) -> None:
    if not isinstance(intent, ProjectIntent):
        raise ProjectStructureError("intent must be a ProjectIntent.")
    if not isinstance(intent.name, str) or not intent.name.strip():
        raise ProjectStructureError("Project name must be non-empty.")
    if not isinstance(intent.framework, str) or intent.framework not in SUPPORTED_FRAMEWORKS:
        raise ProjectStructureError(
            f"Unsupported framework {intent.framework!r}; choose one of "
            f"{', '.join(sorted(SUPPORTED_FRAMEWORKS))}."
        )
    if not isinstance(intent.features, tuple):
        raise ProjectStructureError("features must be a tuple of feature names.")
    if any(not isinstance(feature, str) for feature in intent.features):
        raise ProjectStructureError("Each feature must be a string.")
    if len(set(intent.features)) != len(intent.features):
        raise ProjectStructureError("Duplicate project features are not allowed.")
    unknown = set(intent.features) - SUPPORTED_FEATURES
    if unknown:
        raise ProjectStructureError(f"Unsupported features: {', '.join(sorted(unknown))}")
    if "authentication" in intent.features and intent.framework == "python-cli":
        raise ProjectStructureError("Authentication is not a supported Python CLI feature.")


def build_project_plan(intent: ProjectIntent) -> ProjectPlan:
    """Create a validated starter plan without touching the filesystem."""
    _validate(intent)
    entries: list[dict[str, str]] = []

    if intent.framework == "flask":
        entries.extend([
            _entry("app", "directory"),
            _entry("app/__init__.py", "file",
                   "from flask import Flask\n\ndef create_app():\n"
                   "    app = Flask(__name__)\n"
                   "    from .routes import main\n"
                   "    app.register_blueprint(main)\n"
                   "    return app\n"),
            _entry("app/routes.py", "file",
                   "from flask import Blueprint\n\nmain = Blueprint('main', __name__)\n\n"
                   "@main.get('/')\ndef index():\n    return {'message': 'Flask app is ready'}\n"),
            _entry("config.py", "file", "import os\n\nSECRET_KEY = os.environ.get('SECRET_KEY')\n"),
            _entry("requirements.txt", "file", "Flask>=3.0\n"),
            _entry("run.py", "file",
                   "from app import create_app\n\napp = create_app()\n\n"
                   "if __name__ == '__main__':\n    app.run()\n"),
            _entry("README.md", "file", f"# {intent.name}\n\nFlask starter project.\n"),
            _entry(".gitignore", "file", "venv/\n__pycache__/\n.env\n*.py[cod]\n"),
            _entry(".env.example", "file", "SECRET_KEY=replace-me-locally\n"),
        ])
        if "templates" in intent.features:
            entries.extend([
                _entry("app/templates", "directory"),
                _entry("app/templates/base.html", "file",
                       "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
                       "<title>{{ title or 'Flask App' }}</title></head><body>"
                       "{% block content %}{% endblock %}</body></html>\n"),
                _entry("app/templates/index.html", "file",
                       "{% extends 'base.html' %}\n{% block content %}<h1>Welcome</h1>{% endblock %}\n"),
                _entry("app/templates/login.html", "file",
                       "{% extends 'base.html' %}\n{% block content %}<h1>Login</h1>{% endblock %}\n"),
                _entry("app/templates/dashboard.html", "file",
                       "{% extends 'base.html' %}\n{% block content %}<h1>Dashboard</h1>{% endblock %}\n"),
            ])
        if "static" in intent.features:
            entries.extend([
                _entry("app/static", "directory"),
                _entry("app/static/css", "directory"),
                _entry("app/static/css/style.css", "file", "/* Add project styles here. */\n"),
                _entry("app/static/js", "directory"),
                _entry("app/static/js/main.js", "file", "'use strict';\n"),
                _entry("app/static/images", "directory"),
            ])
        if "database" in intent.features:
            entries.extend([
                _entry("app/models.py", "file", '"""Database models go here."""\n'),
                _entry("migrations", "directory"),
                _entry("instance", "directory"),
                _entry("instance/config.py", "file", "# Local instance configuration.\n"),
            ])
        if "authentication" in intent.features:
            entries.extend([
                _entry("app/forms.py", "file", '"""Authentication forms go here."""\n'),
                _entry("app/auth", "directory"),
                _entry("app/auth/__init__.py", "file", ""),
            ])
        if "tests" in intent.features:
            entries.extend([
                _entry("tests", "directory"),
                _entry("tests/__init__.py", "file", ""),
                _entry("tests/test_routes.py", "file",
                       "from app import create_app\n\ndef test_index_route():\n"
                       "    response = create_app().test_client().get('/')\n"
                       "    assert response.status_code == 200\n"),
                _entry("tests/test_models.py", "file",
                       "def test_models_module_is_available():\n    import app.models\n"),
            ])

    elif intent.framework == "fastapi":
        entries.extend([
            _entry("app", "directory"),
            _entry("app/__init__.py", "file", ""),
            _entry("app/main.py", "file",
                   "from fastapi import FastAPI\n\napp = FastAPI()\n\n"
                   "@app.get('/')\ndef read_root():\n    return {'message': 'FastAPI app is ready'}\n"),
            _entry("requirements.txt", "file", "fastapi>=0.110\nuvicorn[standard]>=0.27\n"),
            _entry("README.md", "file", f"# {intent.name}\n\nFastAPI starter project.\n"),
        ])
        if "tests" in intent.features:
            entries.extend([
                _entry("tests", "directory"),
                _entry("tests/test_main.py", "file",
                       "from fastapi.testclient import TestClient\nfrom app.main import app\n\n"
                       "def test_root():\n    assert TestClient(app).get('/').status_code == 200\n"),
            ])

    else:  # python-cli
        entries.extend([
            _entry("src", "directory"),
            _entry("src/__init__.py", "file", ""),
            _entry("src/main.py", "file",
                   "def main() -> int:\n    print('Hello from your Python project!')\n"
                   "    return 0\n\nif __name__ == '__main__':\n    raise SystemExit(main())\n"),
            _entry("README.md", "file", f"# {intent.name}\n\nPython CLI starter project.\n"),
            _entry("pyproject.toml", "file",
                   "[project]\nname = " + json.dumps(intent.name) + "\nversion = \"0.1.0\"\n"
                   "requires-python = \">=3.11\"\n"),
        ])
        if "tests" in intent.features:
            entries.extend([
                _entry("tests", "directory"),
                _entry("tests/test_main.py", "file",
                       "from src.main import main\n\ndef test_main():\n    assert main() == 0\n"),
            ])

    return ProjectPlan.from_entries(intent.name, entries)
