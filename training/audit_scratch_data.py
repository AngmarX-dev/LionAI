#!/usr/bin/env python3
"""Audit LionAI JSONL metadata datasets without opening user files."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ["Images", "Documents", "Data", "Audio", "Video", "Archives",
          "Source", "Config", "Installers", "Fonts", "Databases", "Reports", "Other/Review"]
REQUIRED = {"topic", "filename", "extension", "label"}


def load_dataset(path: Path) -> list[dict[str, str]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict) or set(row) != REQUIRED:
            raise ValueError(f"{path}:{line_number}: rows must contain exactly {sorted(REQUIRED)}")
        if not all(isinstance(row[key], str) for key in REQUIRED):
            raise ValueError(f"{path}:{line_number}: every field must be a string")
        if not row["topic"].strip() or not row["filename"].strip():
            raise ValueError(f"{path}:{line_number}: topic and filename cannot be empty")
        name = row["filename"]
        if name in {".", ".."} or "/" in name or "\\\\" in name or any(ord(ch) < 32 for ch in name):
            raise ValueError(f"{path}:{line_number}: filename must be a plain filename")
        if row["label"] not in LABELS:
            raise ValueError(f"{path}:{line_number}: unknown label {row['label']!r}")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: dataset is empty")
    return rows


def audit(train_path: Path, eval_path: Path) -> dict:
    train = load_dataset(train_path)
    evaluation = load_dataset(eval_path)
    train_topics = {row["topic"].casefold().strip() for row in train}
    eval_topics = {row["topic"].casefold().strip() for row in evaluation}
    overlap = sorted(train_topics & eval_topics)
    if overlap:
        raise ValueError("Train/evaluation topic leakage: " + ", ".join(overlap))

    counts = Counter(row["label"] for row in train)
    eval_counts = Counter(row["label"] for row in evaluation)
    missing_train = sorted(set(LABELS) - set(counts))
    missing_eval = sorted(set(LABELS) - set(eval_counts))
    if missing_train or missing_eval:
        raise ValueError(f"Missing classes; train={missing_train}, evaluation={missing_eval}")

    seen = {}
    duplicates = []
    for index, row in enumerate(train, 1):
        key = (row["topic"].casefold().strip(), row["filename"].casefold())
        if key in seen:
            duplicates.append({"first_line": seen[key], "duplicate_line": index, "filename": row["filename"]})
        else:
            seen[key] = index
    if duplicates:
        raise ValueError(f"Duplicate topic/filename examples: {duplicates[:5]}")

    return {
        "train_examples": len(train),
        "evaluation_examples": len(evaluation),
        "train_topics": len(train_topics),
        "evaluation_topics": len(eval_topics),
        "train_class_counts": dict(sorted(counts.items())),
        "evaluation_class_counts": dict(sorted(eval_counts.items())),
        "topic_overlap": 0,
        "duplicate_train_examples": 0,
        "metadata_schema": sorted(REQUIRED),
        "file_contents_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="training/lionai_scratch_train.jsonl")
    parser.add_argument("--evaluation", default="training/lionai_scratch_eval.jsonl")
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.train), Path(args.evaluation)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
