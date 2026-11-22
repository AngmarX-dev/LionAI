# LionAI

LionAI is a lightweight, offline file organizer with a single native CPU text-classification model. Its core is written in C. Python is optional and is not required at inference/runtime.

## Native build (no Python)

**Windows (Visual Studio Developer Command Prompt):**
```bat
cl /std:c11 /W4 lionai.c
lionai.exe --help
```

**Linux / LionOS-compatible toolchain:**
```sh
cc -std=c11 -O2 -Wall -Wextra -pedantic lionai.c -o lionai -lm
./lionai --help
```

## Train the single CPU model

The starter bilingual dataset is `training/lionai_train.tsv`. Each row contains a class label, a tab, and a short text sample. Labels must match the built-in class names exactly.

```sh
./lionai train training/lionai_train.tsv lionai.model
./lionai predict lionai.model "family vacation photo camera"
```

Training and prediction are implemented in C and run on CPU. The model is a compact hashed-feature multinomial Naive Bayes text classifier; it is not a large neural language model. It learns associations between words/tokens and file categories. The starter dataset is only a smoke-test baseline, not a production-quality or broadly validated training corpus. Add carefully labeled examples from the target users' real naming patterns, then evaluate on a separate held-out test set before relying on predictions.

The model file is generated locally and is not committed by default. Retrain after changing the dataset. Keep a backup of any model you want to preserve.

## Optional Python launcher

After compiling the native executable into this directory (or `build/`), use:

```sh
python lionai.py --help
python lionai.py predict lionai.model "project source code"
```

The Python script only forwards commands to the native executable; it does not contain or duplicate the AI model.

## File organization commands

```text
lionai create <directory> <topic>
lionai organize <directory> <topic> [--apply]
```

- `create` creates a topic-based folder structure.
- `organize` currently scans only top-level files and previews extension-based moves.
- Add `--apply` to perform the moves. Review the preview first.
- File contents are not rewritten.
- Current organization still uses the existing extension rules; model-driven organization is a later integration step.

## Current limitations and safety

- The new model predicts among the built-in file categories from text supplied to `predict`; it is not yet connected to the `organize` command.
- The starter dataset is small and partly hand-authored. Do not treat its output as reliable until tested against a separate, representative evaluation set.
- Organization is non-recursive (top-level files only).
- Rollback/undo is not implemented yet; back up files before applying moves.
- The Windows file APIs currently use ANSI paths, so some non-ASCII paths may not work reliably.
- No file contents are read by the current organizer or model training command; training reads only the supplied TSV text dataset.

## Runtime profile

The classifier is designed for CPU-only inference and does not require Python, a GPU, internet access, or an external AI service at runtime.
