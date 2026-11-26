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

The deterministic intent layer in `project_intent.py` validates structured
requests and builds starter manifests for Flask, FastAPI, and Python CLI projects.
The bilingual phrase parser in `natural_language_planner.py` converts supported
Persian and English requests into this schema. It is deliberately conservative,
not a general-purpose language model. The resulting manifest must be previewed
and explicitly approved before applying it.

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

1. Completed: project-intent schema with framework-specific feature validation.
2. Completed: curated starter templates for Flask, FastAPI, and Python CLI.
3. Completed: conservative Persian/English request parsing with safe validation.
4. Next: add a reconstruction importer for filesystem scans and textual tree input.
5. Next: add end-to-end fixtures that compare a requested tree with the generated tree.

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


## Project reconstruction

`training/project_reconstruction.py` reconstructs a structure-only ProjectPlan
from either a canonical Unicode tree listing or an existing directory.

- `parse_tree_text(text, project_name=...)` accepts tree output using
  `├──`, `└──`, and `│` indentation. A trailing slash marks directories;
  parents containing children are inferred as directories.
- `scan_directory(path)` walks an existing directory without following
  symlinks. Empty directories are retained.
- Both paths enforce entry limits and reject unsafe names or malformed input.
- File contents are omitted by default. Optional UTF-8 capture is size-bounded
  and skips common secret filenames such as .env, key, token, and credentials.
- Reconstruction is read-only. It returns a normal ProjectPlan for preview;
  creating a separate copy still requires explicit confirmation through
  ProjectPlan.apply and never overwrites an existing project.

Example:

```python
from training.project_reconstruction import parse_tree_text, scan_directory

tree_plan = parse_tree_text("demo/\\n├── src/\\n│   └── main.py\\n")
print(tree_plan.preview())

existing_plan = scan_directory("./existing-project")
print(existing_plan.preview())
```

The parser reconstructs structure, not missing source code or semantic intent.
For a bare leaf in a tree listing, the default interpretation is a file unless
it is marked with a trailing slash or contains children.


## Project diff

`training/project_diff.py` compares two validated `ProjectPlan` objects and
returns an immutable, read-only report.

- Added, removed, modified, unchanged, and file/directory type-conflict paths
  are reported separately.
- Structure-only comparison is the default. It does not claim that file
  contents are equal.
- `compare_contents=True` compares the supplied text snapshots; use it only
  when both plans contain complete, trustworthy content snapshots.
- A type conflict blocks automatic classification of descendants beneath that
  path and requires manual resolution.
- Removed paths are suggestions for review only. The diff engine has no apply
  or delete operation and never changes the filesystem.

Example:

```python
from training.project_diff import compare_project_plans
from training.project_reconstruction import scan_directory
from training.project_intent import ProjectIntent, build_project_plan

current = scan_directory("./existing-project")
desired = build_project_plan(ProjectIntent(name="existing-project", framework="flask"))
report = compare_project_plans(current, desired)
print(report.preview())
```

Review every reported removal and conflict manually. This stage only reports
differences; it does not synchronize the project.


## Project synchronization planner

`training/project_sync.py` converts a validated `ProjectDiff` into an
immutable, ordered synchronization proposal. It is a planner only: it does not
write, overwrite, rename, or delete files.

- New directories are proposed before their child files.
- New files and content updates are represented as separate operations.
- Updates are proposed only when the source diff compared text content and
  identified a changed file.
- Removals are excluded by default and listed as pending review.
- `include_deletions=True` includes explicitly marked destructive proposals,
  ordered deepest-first. These are still inert descriptions, not executable
  operations.
- File/directory type conflicts are surfaced for manual resolution and never
  converted into automatic actions.
- The planner validates that the supplied diff matches the exact current and
  desired plans, rejecting stale or unrelated diff results.

Example:

```python
from training.project_diff import compare_project_plans
from training.project_sync import build_sync_plan

diff = compare_project_plans(current, desired, compare_contents=True)
sync = build_sync_plan(current, desired, diff)
print(sync.preview())

# To include deletion proposals for review only:
review_with_deletions = build_sync_plan(
    current, desired, diff, include_deletions=True
)
print(review_with_deletions.preview())
```

No synchronization executor is included in this milestone. A future executor
must independently require explicit user approval, revalidate the target
filesystem immediately before each operation, and keep destructive actions
behind a separate confirmation step.


## Confirmed project synchronization executor

`training/project_sync_executor.py` adds a guarded executor for non-destructive
synchronization actions.

- Execution requires `confirmed=True`; truthy substitutes such as `1` or
  `"yes"` are rejected.
- The project root must already exist and must not be a symlink.
- Paths are checked for traversal, duplicate operations, and symlink
  components. Existing creation targets are rejected.
- Before any mutation, all update targets must still match the exact original
  text snapshot captured by the synchronization planner. If the current scan
  did not capture a file's content, an update cannot be executed safely and is
  rejected.
- New files use exclusive creation. Updated files use a same-directory
  temporary file followed by atomic replacement.
- If an operation fails, the executor attempts to restore updated files and
  remove files/directories created during that invocation.
- Pending removals are informational only. Delete actions are rejected even
  when marked destructive; deletion execution is intentionally deferred to a
  separate implementation with its own approval and rollback model.

Example:

```python
from training.project_sync_executor import execute_sync_plan

# Show sync.preview() and obtain an explicit user approval first.
result = execute_sync_plan(sync, "./workspace/demo", confirmed=True)
print(result)
```

The executor supports text files only. It does not execute generated code,
modify file permissions, or follow symlinks intentionally. Rollback is
best-effort if the underlying filesystem itself prevents restoration.
