#!/usr/bin/env python3
"""Merge and audit metadata-only JSONL corpora before any training use.

This tool never opens referenced files. It only reads the JSONL metadata rows.
The output stays under training/external by default and is ignored by Git.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from training.audit_scratch_data import LABELS, load_dataset


def merge_corpora(inputs: list[Path]) -> tuple[list[dict[str, str]], dict]:
    if not inputs:
        raise ValueError("At least one --input JSONL file is required.")

    merged: list[dict[str, str]] = []
    seen: dict[tuple[str, str], tuple[str, int]] = {}
    duplicate_count = 0
    source_report = []

    for source in inputs:
        rows = load_dataset(source)
        added = 0
        duplicates = 0
        for line_number, row in enumerate(rows, 1):
            key = (row["topic"].casefold().strip(), row["filename"].casefold())
            previous = seen.get(key)
            if previous is not None:
                previous_label, _ = previous
                if previous_label != row["label"]:
                    raise ValueError(
                        f"Conflicting labels for topic/filename {key!r}: "
                        f"{previous_label!r} vs {row['label']!r} in {source}:{line_number}"
                    )
                duplicate_count += 1
                duplicates += 1
                continue
            seen[key] = (row["label"], len(merged))
            merged.append(row)
            added += 1
        source_report.append({
            "source": source.as_posix(),
            "rows_read": len(rows),
            "rows_added": added,
            "duplicates_skipped": duplicates,
        })

    counts = Counter(row["label"] for row in merged)
    topic_counts = Counter(row["topic"].casefold().strip() for row in merged)
    class_counts = {label: counts.get(label, 0) for label in LABELS}
    report = {
        "rows_total": len(merged),
        "topics_total": len(topic_counts),
        "duplicate_rows_skipped": duplicate_count,
        "class_counts": class_counts,
        "classes_missing": [label for label in LABELS if class_counts[label] == 0],
        "classes_below_10": [label for label in LABELS if 0 < class_counts[label] < 10],
        "sources": source_report,
        "metadata_only": True,
        "file_contents_used": False,
        "split_policy": (
            "Do not randomly split rows. Keep all rows sharing a normalized topic in "
            "one split; reserve separate topics for train, validation, calibration, and test."
        ),
    }
    return merged, report


def atomic_write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True,
                        help="Input LionAI metadata JSONL; repeat for multiple sources")
    parser.add_argument("--output", type=Path,
                        default=Path("training/external/lionai_metadata_corpus.jsonl"))
    parser.add_argument("--report", type=Path,
                        default=Path("training/external/lionai_metadata_corpus_report.json"))
    args = parser.parse_args()

    rows, report = merge_corpora(args.input)
    atomic_write_jsonl(args.output, rows)
    report["output"] = args.output.as_posix()
    report["report"] = args.report.as_posix()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(args.report.name + ".tmp")
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(args.report)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
