#!/usr/bin/env python3
"""Optional Python launcher for the native LionAI executable.

The file organization logic remains in lionai.c. Python is not required when
running the compiled native executable directly.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_native_binary() -> Path | None:
    root = Path(__file__).resolve().parent
    names = ("lionai.exe", "lionai") if os.name == "nt" else ("lionai", "lionai.exe")
    candidates = [root / name for name in names]
    candidates.extend(root / "build" / name for name in names)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def main() -> int:
    binary = find_native_binary()
    if binary is None:
        print("LionAI native executable was not found.", file=sys.stderr)
        print("Compile lionai.c first; see README.md for build commands.", file=sys.stderr)
        print("Python is only a launcher: it does not replace the native C engine.", file=sys.stderr)
        return 127

    try:
        completed = subprocess.run([str(binary), *sys.argv[1:]], check=False)
        return completed.returncode
    except OSError as exc:
        print(f"Could not start LionAI: {exc}", file=sys.stderr)
        return 126


if __name__ == "__main__":
    raise SystemExit(main())
