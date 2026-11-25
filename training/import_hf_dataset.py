#!/usr/bin/env python3
"""Import metadata-only examples from a Hugging Face Hub dataset into LionAI's scratch schema.

This script downloads dataset records, never user files or file contents. The imported
file is an external supplemental source and must be audited before it is merged into
the official topic-disjoint training/evaluation splits.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DATASET = "AnodemTheGr8t/file_organiser_metadata"
LABELS = {
    "Images", "Documents", "Data", "Audio", "Video", "Archives",
    "Source", "Config", "Installers", "Fonts", "Databases", "Reports", "Other/Review",
}
DATA_EXTENSIONS = {
    "csv", "tsv", "xlsx", "xls", "parquet", "jsonl", "ndjson",
    "feather", "arrow", "geojson",
}
CONFIG_EXTENSIONS = {"json", "xml", "yaml", "yml", "toml", "ini", "cfg", "conf", "env"}
INSTALLER_EXTENSIONS = {"exe", "msi", "msix", "dmg", "iso", "apk", "deb", "rpm", "appimage"}
FONT_EXTENSIONS = {"ttf", "otf", "woff", "woff2"}
DATABASE_EXTENSIONS = {"db", "sqlite", "sqlite3", "sql"}


def normalized_extension(filename: str, source_extension: Any) -> str:
    """Return the final suffix expected by LionAI's current metadata schema."""
    suffix = Path(filename).suffix.casefold().lstrip(".")
    if suffix:
        return suffix
    value = str(source_extension or "").casefold().strip().lstrip(".")
    return value.split(".")[-1] if value else ""


def map_label(source_label: Any, extension: str) -> str:
    """Map the source's seven labels to LionAI's thirteen classes conservatively."""
    source = str(source_label or "").strip().casefold()
    ext = extension.casefold()
    if source == "images":
        return "Images"
    if source == "documents":
        return "Data" if ext in DATA_EXTENSIONS else "Documents"
    if source == "audio":
        return "Audio"
    if source == "video":
        return "Video"
    if source == "archives":
        return "Archives"
    if source == "code":
        if ext in CONFIG_EXTENSIONS:
            return "Config"
        if ext in DATA_EXTENSIONS:
            return "Data"
        return "Source"
    if source == "other":
        if ext in INSTALLER_EXTENSIONS:
            return "Installers"
        if ext in FONT_EXTENSIONS:
            return "Fonts"
        if ext in DATABASE_EXTENSIONS:
            return "Databases"
        return "Other/Review"
    raise ValueError(f"Unsupported source label: {source_label!r}")


def normalize_record(record: dict[str, Any], row_number: int) -> dict[str, str]:
    required = ("file_name", "extension", "folder_hint", "label")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"row {row_number}: missing fields: {', '.join(missing)}")

    filename = record["file_name"]
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError(f"row {row_number}: file_name must be non-empty text")
    filename = filename.strip()
    if (
        filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or any(ord(char) < 32 or char in '<>:"|?*' for char in filename)
    ):
        raise ValueError(f"row {row_number}: file_name must be a plain filename")

    extension = normalized_extension(filename, record["extension"])
    folder_hint = record["folder_hint"]
    topic = str(folder_hint).strip() if folder_hint is not None else ""
    if not topic:
        topic = "Downloads"
    if len(topic) > 200 or any(ord(char) < 32 for char in topic):
        raise ValueError(f"row {row_number}: folder_hint must be at most 200 characters without control characters")
    label = map_label(record["label"], extension)
    if label not in LABELS:
        raise ValueError(f"row {row_number}: mapped label is invalid")
    return {
        "topic": topic,
        "filename": filename,
        "extension": extension,
        "label": label,
    }


def convert_records(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    converted: list[dict[str, str]] = []
    seen: dict[tuple[str, str], str] = {}
    counts: Counter[str] = Counter()
    duplicates = 0
    for index, record in enumerate(records, 1):
        row = normalize_record(dict(record), index)
        key = (row["topic"].casefold(), row["filename"].casefold())
        previous_label = seen.get(key)
        if previous_label is not None:
            if previous_label != row["label"]:
                raise ValueError(
                    f"row {index}: conflicting labels for duplicate topic/filename "
                    f"({previous_label!r} vs {row['label']!r})"
                )
            duplicates += 1
            continue
        seen[key] = row["label"]
        converted.append(row)
        counts[row["label"]] += 1
    report = {label: counts.get(label, 0) for label in sorted(LABELS)}
    report["duplicates_skipped"] = duplicates
    return converted, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset repository ID")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", default="main", help="Pin this to a commit hash for reproducible imports")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional positive limit; uses streaming mode")
    parser.add_argument(
        "--output",
        default="training/external/lionai_file_organiser_metadata.jsonl",
    )
    args = parser.parse_args()
    if args.max_rows < 0:
        parser.error("--max-rows must be zero (unlimited) or a positive integer")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with: python -m pip install -r training/requirements-scratch.txt"
        ) from exc

    dataset = load_dataset(
        args.dataset,
        split=args.split,
        revision=args.revision,
        trust_remote_code=False,
        streaming=bool(args.max_rows),
    )
    records = islice(dataset, args.max_rows) if args.max_rows else dataset
    rows, counts = convert_records(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(json.dumps({
        "source_dataset": args.dataset,
        "revision": args.revision,
        "split": args.split,
        "max_rows": args.max_rows or None,
        "rows_written": len(rows),
        "output": str(output),
        "mapped_class_counts": counts,
        "metadata_only": True,
        "remote_code_execution": False,
        "source_license": "CC0-1.0 (verify the dataset card before redistribution)",
        "warning": (
            "This synthetic English dataset is supplemental only. Its folder_hint values "
            "become topic metadata; do not merge it into the official topic-disjoint split "
            "until group leakage and class coverage are audited."
        ),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
