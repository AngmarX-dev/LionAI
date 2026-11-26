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

This module is a deterministic executor, not a natural-language generator. A
future intent/planning layer will convert requests such as "create a Flask shop"
into a manifest. The manifest must be reviewed before applying it.

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
