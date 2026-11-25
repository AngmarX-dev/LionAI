#!/usr/bin/env python3
"""Collect real-world filename metadata from permissively licensed public GitHub repos.

Only Git tree metadata is requested. The importer never downloads or opens file blobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

API = "https://api.github.com"
ALLOWED_LICENSES = {
    "0BSD", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "CC0-1.0", "CC-BY-4.0",
    "ISC", "MIT", "OFL-1.1", "PSF-2.0", "Unlicense", "Zlib",
}
LABELS = {
    "Images", "Documents", "Data", "Audio", "Video", "Archives",
    "Source", "Config", "Installers", "Fonts", "Databases", "Reports",
    "Other/Review",
}
EXTENSIONS = {
    "Images": {"jpg", "jpeg", "png", "gif", "webp", "svg", "bmp", "tif", "tiff", "ico", "avif", "heic", "psd", "xcf"},
    "Documents": {"pdf", "doc", "docx", "odt", "rtf", "txt", "md", "rst", "html", "htm", "epub", "ppt", "pptx", "odp", "xls", "xlsx", "tex"},
    "Data": {"csv", "tsv", "parquet", "jsonl", "ndjson", "feather", "arrow", "geojson", "avro", "orc", "h5", "hdf5"},
    "Audio": {"mp3", "wav", "flac", "ogg", "m4a", "aac", "mid", "midi", "opus", "aiff"},
    "Video": {"mp4", "mkv", "mov", "avi", "webm", "mpeg", "mpg", "m4v", "ogv"},
    "Archives": {"zip", "tar", "gz", "bz2", "xz", "7z", "rar", "zst", "tgz", "whl"},
    "Installers": {"exe", "msi", "msix", "dmg", "pkg", "deb", "rpm", "appimage", "apk", "ipa"},
    "Fonts": {"ttf", "otf", "woff", "woff2", "eot"},
    "Databases": {"db", "sqlite", "sqlite3", "mdb", "accdb", "duckdb", "sql"},
    "Config": {"json", "xml", "yaml", "yml", "toml", "ini", "cfg", "conf", "env", "properties", "lock", "gitignore", "gitattributes", "editorconfig"},
    "Source": {"py", "pyi", "js", "jsx", "ts", "tsx", "java", "c", "h", "cc", "cpp", "hpp", "cs", "go", "rs", "rb", "php", "swift", "kt", "kts", "scala", "sh", "ps1", "bat", "lua", "r", "m", "mm", "dart", "vue", "svelte"},
}
REPORT_WORDS = {"report", "reports", "audit", "summary", "benchmark", "results"}
SENSITIVE_WORDS = {
    "secret", "secrets", "password", "passwords", "credential", "credentials",
    "private", "token", "tokens", "id_rsa", "id_ed25519",
}
SENSITIVE_SUFFIXES = {".pem", ".p12", ".pfx", ".key", ".keystore"}


def api_get(path: str, token: str, retries: int = 4) -> dict:
    url = API + path
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "LionAI-metadata-collector",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(retries):
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code in (403, 429) and attempt + 1 < retries:
                retry_after = int(exc.headers.get("Retry-After", "0") or 0)
                time.sleep(max(retry_after, 2 ** attempt))
                continue
            raise RuntimeError(f"GitHub API {exc.code} for {path}") from exc
        except (URLError, TimeoutError) as exc:
            if attempt + 1 >= retries:
                raise RuntimeError(f"GitHub API request failed for {path}: {exc}") from exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"GitHub API retries exhausted for {path}")


def classify(filename: str, path: str) -> tuple[str, str] | None:
    name = filename.casefold()
    lower_path = path.casefold()
    if any(part in SENSITIVE_WORDS for part in re.split(r"[^a-z0-9]+", lower_path)):
        return None
    if any(name.endswith(suffix) for suffix in SENSITIVE_SUFFIXES):
        return None
    if len(filename) > 180 or not filename or filename in {".", ".."}:
        return None
    if any(ord(ch) < 32 or ch in '<>:"|?*\\/' for ch in filename):
        return None

    suffix = Path(filename).suffix.casefold().lstrip(".")
    if not suffix and filename.startswith("."):
        suffix = filename[1:].casefold()
    if not suffix:
        return None

    if any(word in name for word in REPORT_WORDS) and suffix in (
        EXTENSIONS["Documents"] | EXTENSIONS["Data"]
    ):
        label = "Reports"
    else:
        label = next((category for category, values in EXTENSIONS.items() if suffix in values), "Other/Review")
    return suffix, label


def reservoir_sample(items, limit: int, seed: str):
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))
    chosen = []
    for index, item in enumerate(items):
        if index < limit:
            chosen.append(item)
        else:
            replacement = rng.randint(0, index)
            if replacement < limit:
                chosen[replacement] = item
    return chosen


def collect_repo(repo: str, token: str, max_files: int) -> tuple[list[dict[str, str]], dict]:
    info = api_get(f"/repos/{repo}", token)
    license_info = info.get("license") or {}
    license_id = license_info.get("spdx_id") or "NOASSERTION"
    if license_id not in ALLOWED_LICENSES:
        return [], {"repository": repo, "status": "skipped_license", "license": license_id}

    default_branch = info.get("default_branch")
    if not default_branch:
        return [], {"repository": repo, "status": "skipped_no_branch", "license": license_id}

    branch_info = api_get(
        f"/repos/{repo}/branches/{quote(default_branch, safe='')}",
        token,
    )
    commit_sha = branch_info["commit"]["sha"]
    tree_sha = branch_info["commit"]["commit"]["tree"]["sha"]
    tree = api_get(
        f"/repos/{repo}/git/trees/{tree_sha}?recursive=1",
        token,
    )
    # Recursive Git tree responses can be truncated for large repositories.
    # Never treat a partial tree as a complete benchmark or training source.
    if tree.get("truncated"):
        return [], {
            "repository": repo,
            "status": "skipped_truncated_tree",
            "license": license_id,
            "revision": default_branch,
            "commit": commit_sha,
            "tree_sha": tree.get("sha"),
            "tree_truncated": True,
            "metadata_only": True,
            "source_url": f"https://github.com/{repo}",
        }

    entries = tree.get("tree") or []
    candidates = []
    seen_names = set()
    for entry in entries:
        if entry.get("type") != "blob" or entry.get("mode") not in ("100644", "100755"):
            continue
        path = entry.get("path", "")
        if not path or path.startswith("/") or "\\" in path:
            continue
        parts = path.split("/")
        filename = parts[-1]
        if any(part.casefold() in SENSITIVE_WORDS for part in parts):
            continue
        classified = classify(filename, path)
        if classified is None:
            continue
        extension, label = classified
        key = filename.casefold()
        if key in seen_names:
            continue
        seen_names.add(key)
        candidates.append((path, {
            "topic": f"github:{repo}",
            "filename": filename,
            "extension": extension,
            "label": label,
        }))

    selected = reservoir_sample(candidates, max_files, repo)
    rows = [row for _, row in sorted(selected, key=lambda item: item[0])]
    return rows, {
        "repository": repo,
        "status": "imported",
        "license": license_id,
        "revision": info.get("default_branch"),
        "commit": commit_sha,
        "tree_sha": tree.get("sha"),
        "tree_truncated": bool(tree.get("truncated")),
        "tree_entries": len(entries),
        "candidate_files": len(candidates),
        "rows_written": len(rows),
        "metadata_only": True,
        "source_url": f"https://github.com/{repo}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="training/github_metadata_sources.txt")
    parser.add_argument("--output", default="training/external/lionai_github_metadata.jsonl")
    parser.add_argument("--report", default="training/external/lionai_github_metadata_report.json")
    parser.add_argument("--max-files-per-repo", type=int, default=5000)
    parser.add_argument("--max-repos", type=int, default=0, help="0 imports every listed repository")
    args = parser.parse_args()
    if args.max_files_per_repo < 1 or args.max_repos < 0:
        parser.error("file/repository limits must be positive (max-repos may be zero)")

    token = os.environ.get("GITHUB_TOKEN", "")
    repos = [
        line.strip() for line in Path(args.sources).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if args.max_repos:
        repos = repos[:args.max_repos]

    all_rows = []
    manifests = []
    for repo in repos:
        try:
            rows, manifest = collect_repo(repo, token, args.max_files_per_repo)
        except Exception as exc:
            manifest = {"repository": repo, "status": "error", "error": str(exc)}
            rows = []
        all_rows.extend(rows)
        manifests.append(manifest)
        print(json.dumps(manifest, ensure_ascii=False))

    # Repository topic is the leakage boundary. Keep the largest set of unique
    # (topic, filename) records and reject any accidental label conflicts.
    unique = {}
    for row in all_rows:
        key = (row["topic"].casefold(), row["filename"].casefold())
        prior = unique.get(key)
        if prior is not None and prior["label"] != row["label"]:
            raise SystemExit(f"Conflicting labels for {key}")
        unique.setdefault(key, row)
    rows = sorted(unique.values(), key=lambda row: (row["topic"], row["filename"].casefold()))
    if not rows:
        raise SystemExit("No eligible metadata rows were imported; inspect repository license/API reports.")

    counts = Counter(row["label"] for row in rows)
    topic_counts = defaultdict(set)
    for row in rows:
        topic_counts[row["label"]].add(row["topic"])
    report = {
        "source_type": "public GitHub repository trees",
        "repositories_requested": len(repos),
        "repositories_imported": sum(item.get("status") == "imported" for item in manifests),
        "repositories_skipped_or_failed": sum(item.get("status") != "imported" for item in manifests),
        "rows_written": len(rows),
        "class_counts": {label: counts.get(label, 0) for label in sorted(LABELS)},
        "topics_per_class": {label: len(topic_counts[label]) for label in sorted(LABELS)},
        "licenses_allowed": sorted(ALLOWED_LICENSES),
        "repositories": manifests,
        "metadata_only": True,
        "file_contents_downloaded": False,
        "note": "Labels are inferred from filename extension and report-like naming; audit class semantics before training.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temp.replace(output)
    finally:
        if temp.exists():
            temp.unlink()
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "repositories"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
