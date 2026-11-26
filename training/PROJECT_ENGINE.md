# LionAI Project Structure Engine

The first Project Engine milestone separates deterministic filesystem structure
handling from the metadata classifier.

## Capabilities

- Build a validated project plan from a manifest of relative paths.
- Preserve nested and empty directories.
- Render a stable tree preview without writing to disk.
- Scaffold a new project only after explicit confirmation.
- Refuse path traversal, duplicate paths, file/directory conflicts, symlink
  components, and existing project destinations.
- Never execute generated code and never overwrite existing files.

The deterministic intent layer in `project_intent.py` now validates structured
requests and builds starter manifests for Flask, FastAPI, and Python CLI projects.
It does not parse arbitrary natural-language prompts yet. A future language
planner will convert user requests into this schema. The resulting manifest must
be previewed and explicitly approved before applying it.

## Example

```python
from training.project_engine import ProjectPlan

plan = ProjectPlan.from_entries(
    "flask-shop",
    [
        {"path": "app", "kind": "directory"},
        {"path": "app/templates", "kind": "directory"},
        {"path": "app/routes.py", "kind": "file",
         "content": "from flask import Blueprint\n"},
        {"path": "migrations", "kind": "directory"},
        {"path": "README.md", "kind": "file",
         "content": "# Flask Shop\n"},
    ],
)

# Preview is read-only.
print(plan.preview())

# Apply only after the caller has obtained explicit user confirmation.
created = plan.apply("./workspace", confirmed=True)
print(created)
```

## Manifest contract

Each entry has a normalized relative POSIX path and a kind of `directory` or
`file`. File entries may include UTF-8 text content. Every parent directory
must be explicitly represented, which ensures empty directories are retained
and makes the plan auditable.

The caller must not set `confirmed=True` merely because a model proposed a
plan. It must represent a separate, explicit user approval.

## Next milestones

1. Add a project-intent schema (framework, features, constraints, target name).
2. Add curated templates for Flask, FastAPI, Python CLI, and common project types.
3. Convert natural-language requests into a proposed manifest, with validation.
4. Add a reconstruction importer for filesystem scans and textual tree input.
5. Add end-to-end fixtures that compare a requested tree with the generated tree.

## Structured project intent

```python
from training.project_intent import ProjectIntent, build_project_plan

intent = ProjectIntent(
    name="flask-shop",
    framework="flask",
    features=("tests", "templates", "static", "database"),
)
plan = build_project_plan(intent)
print(plan.preview())
# After the user reviews and explicitly approves:
# plan.apply("./workspace", confirmed=True)
```

Supported frameworks: `flask`, `fastapi`, and `python-cli`.

Supported optional features: `tests`, `templates`, `static`, `database`,
and `authentication` (Flask only). Unsupported combinations are rejected.
Planning is read-only; only the separately confirmed ProjectPlan can create files.


## Natural-language project requests

`training/natural_language_planner.py` adds a conservative Persian/English
parser for explicit Flask, FastAPI, and Python CLI requests. It recognizes
project names when written as `project name: ...`, `نام پروژه: ...`, or
`named ...`; callers may also pass `project_name` directly. Otherwise it
uses a predictable framework-based name.

Recognized feature phrases map to tests, templates, static assets, database,
and Flask authentication. Ambiguous or unsupported frameworks and unsafe names
are rejected instead of guessed. This parser is deterministic phrase matching,
not an LLM. `plan_project_request()` only returns a previewable plan and never
writes to disk. Filesystem creation still requires an explicit confirmation
through `ProjectPlan.apply(..., confirmed=True)`.

Default feature sets are framework-aware: Flask receives the full web starter,
while FastAPI and Python CLI receive tests by default. Authentication remains
an explicit Flask-only feature.
