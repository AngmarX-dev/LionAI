#!/usr/bin/env python3
"""Validate LionAI JSONL examples and enforce file/path safety invariants."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import PurePosixPath


def fail(source: str, line: int, message: str) -> None:
    raise ValueError(f"{source}:{line}: {message}")


def validate(path: str) -> tuple[int, int]:
    examples = files_total = 0
    with open(path, encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                fail(path, line_number, f"invalid JSONL record: {exc}")
            for key in ("instruction", "input", "output"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    fail(path, line_number, f"{key} must be a non-empty string")
            match = re.search(r"files=(.*)$", row["input"])
            if not match:
                fail(path, line_number, "input must include a files= list")
            names = [part.strip() for part in match.group(1).split(",") if part.strip()]
            if not names or len(names) != len(set(names)):
                fail(path, line_number, "input filenames must be non-empty and unique")
            try:
                output = json.loads(row["output"])
            except json.JSONDecodeError as exc:
                fail(path, line_number, f"output is not JSON: {exc}")
            if not isinstance(output, dict) or output.get("schema_version") != "1.0":
                fail(path, line_number, "output schema_version must be 1.0")
            folders, mappings = output.get("folders"), output.get("files")
            if not isinstance(folders, list) or not all(isinstance(x, str) for x in folders):
                fail(path, line_number, "folders must be an array of strings")
            if not isinstance(mappings, list) or not all(isinstance(x, dict) for x in mappings):
                fail(path, line_number, "files must be an array of objects")
            mapped_names = []
            parents = set()
            for item in mappings:
                for key in ("original_name", "suggested_path", "confidence", "needs_review"):
                    if key not in item:
                        fail(path, line_number, f"file mapping missing {key}")
                name, target = item["original_name"], item["suggested_path"]
                if not isinstance(name, str) or not isinstance(target, str):
                    fail(path, line_number, "file names and paths must be strings")
                p = PurePosixPath(target)
                if p.is_absolute() or ".." in p.parts or "\\" in target or not p.parts:
                    fail(path, line_number, f"unsafe relative path: {target!r}")
                if p.name != name:
                    fail(path, line_number, f"filename changed or extension not preserved: {name!r}")
                confidence = item["confidence"]
                if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                    fail(path, line_number, "confidence must be numeric in [0, 1]")
                if not isinstance(item["needs_review"], bool):
                    fail(path, line_number, "needs_review must be boolean")
                mapped_names.append(name)
                parents.add(str(p.parent))
            if len(mapped_names) != len(set(mapped_names)):
                fail(path, line_number, "duplicate file mappings")
            if sorted(mapped_names) != sorted(names):
                fail(path, line_number, "mapped filenames must exactly match input filenames")
            if set(folders) != parents:
                fail(path, line_number, "folders must exactly match mapping parent directories")
            examples += 1
            files_total += len(names)
    if examples == 0:
        raise ValueError(f"{path}: no examples found")
    print(f"{path}: {examples} examples, {files_total} file mappings validated")
    return examples, files_total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    args = parser.parse_args()
    totals = [validate(path) for path in args.datasets]
    print(f"TOTAL: {sum(x[0] for x in totals)} examples; {sum(x[1] for x in totals)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
