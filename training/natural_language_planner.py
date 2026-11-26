"""Conservative Persian/English request parser for LionAI project scaffolding.

This is a deterministic parser, not a generative language model. It only maps
recognized framework and feature phrases into the validated ProjectIntent
schema. It never writes files; callers must preview and separately confirm.
"""
from __future__ import annotations

import re

from training.project_engine import ProjectStructureError, ProjectPlan
from training.project_intent import ProjectIntent, build_project_plan


_FRAMEWORK_PATTERNS = {
    "flask": (r"\bflask\b", r"فلا?سک"),
    "fastapi": (r"\bfast\s*api\b", r"\bfastapi\b", r"فست\s*ا?پی"),
    "python-cli": (
        r"\bpython\s*(?:cli|command[- ]line)\b",
        r"\bcli\b",
        r"خط\s*فرمان\s*(?:پایتون)?",
    ),
}

_FEATURE_PATTERNS = {
    "tests": (r"\btests?\b", r"\bpytest\b", r"تست", r"آزمون"),
    "templates": (r"\btemplates?\b", r"\bhtml\b", r"قالب", r"تمپلیت"),
    "static": (r"\bstatic\b", r"\bcss\b", r"\bjavascript\b", r"\bjs\b", r"فایل\s*های?\s*استاتیک"),
    "database": (r"\bdatabase\b", r"\bdb\b", r"\bsql(?:ite)?\b", r"پایگاه\s*داده", r"دیتابیس"),
    "authentication": (
        r"\bauth(?:entication)?\b", r"\blogin\b", r"\bsign[- ]?in\b",
        r"\buser\s*accounts?\b", r"ورود", r"احراز\s*هویت", r"ثبت\s*نام",
    ),
}

_NAME_PATTERNS = (
    r"(?:project\s+name|نام\s+پروژه)\s*[:=]?\s*([a-zA-Z0-9][a-zA-Z0-9._-]{0,63})",
    r"(?:named|به\s+نام|با\s+نام)\s+([a-zA-Z0-9][a-zA-Z0-9._-]{0,63})",
)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def parse_project_request(request: str, *, project_name: str | None = None) -> ProjectIntent:
    """Parse a short Persian/English request into a safe, structured intent."""
    if not isinstance(request, str) or not request.strip():
        raise ProjectStructureError("Project request must be a non-empty string.")
    if len(request) > 2000:
        raise ProjectStructureError("Project request is too long (maximum 2000 characters).")

    text = request.casefold()
    frameworks = [
        framework
        for framework, patterns in _FRAMEWORK_PATTERNS.items()
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    ]
    if not frameworks:
        raise ProjectStructureError(
            "Framework is unclear. Specify Flask, FastAPI, or Python CLI."
        )
    if len(frameworks) > 1:
        raise ProjectStructureError("Request mentions multiple frameworks; specify exactly one.")

    framework = frameworks[0]
    features = tuple(
        feature
        for feature, patterns in _FEATURE_PATTERNS.items()
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    )

    name = project_name
    if name is None:
        for pattern in _NAME_PATTERNS:
            match = re.search(pattern, request, re.IGNORECASE)
            if match:
                name = match.group(1)
                break
    if name is None:
        name = f"{framework}-project"
    if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name) or name in {".", ".."}:
        raise ProjectStructureError(
            "Project name must be 1-64 ASCII letters, digits, dots, underscores, or hyphens, "
            "and must start with a letter or digit."
        )

    return ProjectIntent(name=name, framework=framework, features=features)


def plan_project_request(request: str, *, project_name: str | None = None) -> ProjectPlan:
    """Return a validated, read-only scaffold plan for a natural-language request."""
    return build_project_plan(parse_project_request(request, project_name=project_name))
