# LionAI

LionAI is a lightweight, offline file organizer. The organizing engine is written in portable C; Python is an optional launcher only.

## Two ways to run

### 1. Native mode — no Python required

Compile the C engine once, then run the resulting executable directly.

**Windows (Visual Studio Developer Command Prompt):**
```bat
cl /std:c11 /W4 lionai.c
lionai.exe --help
```

**Linux / LionOS toolchain:**
```sh
cc -std=c11 -O2 -Wall -Wextra -pedantic lionai.c -o lionai
./lionai --help
```

The native executable does not need Python, a Python interpreter, network access, or an external AI service.

### 2. Python launcher mode — Python 3.10+

After compiling the native executable into the repository directory (or its `build/` directory), run:

```sh
python lionai.py --help
python lionai.py create "./Workspace" "Photography"
python lionai.py organize "./Downloads" "Personal files"
python lionai.py organize "./Downloads" "Personal files" --apply
```

On Windows, use `py lionai.py ...` if that is how Python is installed. The launcher forwards all arguments and the exit code to the native executable. It does not duplicate the file-management logic.

## Commands

```text
lionai create <directory> <topic>
lionai organize <directory> <topic> [--apply]
```

- `create` creates a topic-based folder structure.
- `organize` scans files in the target directory's top level and prints a proposed organization.
- Add `--apply` to move files into extension-based category folders and assign sequential names.
- The preview is the default. Review it before using `--apply`.
- File contents are not rewritten. Existing destination names are not intentionally overwritten.

## Current limitations

- Classification currently uses file extensions, not semantic understanding of file contents. This MVP is rule-based and should not be described as a full semantic AI model.
- Organization is currently non-recursive (top-level files only).
- A rollback/undo feature is not implemented yet; keep a backup before applying changes.
- The current Windows implementation uses ANSI file APIs, so some non-ASCII paths may not work reliably yet.

## Resource profile

The C engine has no runtime dependency on Python and is designed to use little CPU and memory. Actual performance and compatibility should be measured on each target OS.
