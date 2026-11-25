# LionAI

LionAI is an offline, task-specific file-organization classifier. The selected model is trained **from random initialization** with a custom UTF-8 byte tokenizer and a compact Transformer. No pretrained Qwen, DeepSeek, or other model weights are used.

## Current model contract

- **Architecture:** compact Transformer encoder classifier.
- **Initialization:** random weights; deterministic seed.
- **Tokenizer:** LionAI UTF-8 byte tokenizer.
- **Inputs:** user topic, plain filename, and extension only. File contents are never opened or sent to the model.
- **Outputs:** one of 13 classes: Images, Documents, Data, Audio, Video, Archives, Source, Config, Installers, Fonts, Databases, Reports, and Other/Review.
- **Authority boundary:** the model predicts a class only. Trusted native code validates paths; preview and explicit confirmation are required before any future apply operation.
- **Training status:** the starter data is small and synthetic. A trained checkpoint is generated locally; no trained checkpoint is currently committed to this repository.

## Install and train

    python -m venv .venv
    # Windows: .venv\Scripts\activate
    # Linux: source .venv/bin/activate
    python -m pip install -r training/requirements-scratch.txt

    python training/audit_scratch_data.py
    python training/train_scratch.py --epochs 30 --patience 6 --history-output artifacts/training-history.json
    python training/predict_scratch.py --topic "مدارک مدرسه" --filename "biology_notes.pdf"
    python training/evaluate_scratch.py --test training/lionai_scratch_eval.jsonl --output artifacts/evaluation.json

Training uses CUDA when available and otherwise CPU. A completed training run does not establish production accuracy. Evaluate on a sufficiently large, representative, topic-disjoint test set with per-class precision/recall before enabling model predictions in a real organizer.

## Hugging Face Datasets integration

The project uses the official Hugging Face Datasets library (https://github.com/huggingface/datasets) to load and normalize external metadata datasets. The library is a data-loading/processing toolkit, **not itself a training dataset**.

A first supplemental source is Smart Downloads Organiser Metadata (https://huggingface.co/datasets/AnodemTheGr8t/file_organiser_metadata). Its card describes synthetic English metadata such as filename, extension, folder hint, size, and a seven-class label, and declares CC0-1.0. Its card and viewer currently report different row counts, so the importer reports the actual loaded count rather than assuming one.

    python -m pip install -r training/requirements-scratch.txt
    python training/import_hf_dataset.py

The importer:
- loads records through datasets.load_dataset;
- reads metadata fields only, never file contents;
- maps the source's seven labels to LionAI's 13 classes conservatively;
- rejects path-like filenames and removes duplicate topic/filename records;
- writes a local supplemental file to training/external/ (not committed by the importer).

**Important:** imported rows are supplemental, not yet merged into the official train/evaluation split. The source is synthetic, English-only, and its labels are largely extension-derived. Its folder_hint becomes the topic field, so group leakage and class coverage must be audited before combining it with LionAI's topic-disjoint data.

For a reproducible import, pass a specific Hugging Face dataset commit hash with --revision instead of relying on the moving main revision.

## Repository components

- lionai.c: native organizer and experimental baseline.
- lionai_hybrid.py: metadata-only preview engine; never reads or moves files.
- lionai_scratch.py: custom tokenizer and randomly initialized Transformer.
- training/train_scratch.py: topic-disjoint training split, validation-based checkpoint selection, scheduler, early stopping, and atomic checkpoint output.
- training/predict_scratch.py: metadata-only inference with confidence and ambiguity review.
- training/evaluate_scratch.py: per-class precision/recall/F1, confusion matrix, ECE, and Brier score.
- training/calibrate_scratch.py and training/calibration.py: optional temperature calibration using a dedicated calibration split.
- training/audit_scratch_data.py: schema, class coverage, duplicates, filename/extension consistency, and topic-leakage audit.
- training/import_hf_dataset.py: Hugging Face metadata import and 7-to-13-class mapping.
- tests/: unit and safety-contract tests.

## Safety requirements

- Only metadata supplied by the user or an explicitly configured metadata dataset may reach the model. Never inspect file contents.
- The model cannot choose arbitrary destination paths or perform filesystem operations.
- Validate classes, filenames, paths, traversal, absolute paths, and destination collisions in trusted native code.
- Default to preview-only and require explicit user confirmation before applying changes.
- Never delete or overwrite existing files. Ambiguous predictions go to Other/Review.
