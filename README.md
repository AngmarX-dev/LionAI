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

The project uses the official Hugging Face Datasets library (https://github.com/huggingface/datasets) to load and normalize external metadata datasets. The library is a data-loading/processing toolkit, **not itself a training dataset**. LionAI installs the released Python package from PyPI; it does not clone or vendor the full library source tree.

A first supplemental source is Smart Downloads Organiser Metadata (https://huggingface.co/datasets/AnodemTheGr8t/file_organiser_metadata). Its card describes synthetic English metadata such as filename, extension, folder hint, size, and a seven-class label, and declares CC0-1.0. Its card and viewer have reported different row counts, so the importer reports the actual loaded count.

    python training/import_hf_dataset.py
    # Optional bounded streaming import:
    python training/import_hf_dataset.py --max-rows 5000

The importer loads metadata with remote code execution disabled, validates plain filenames, maps source labels to LionAI classes, deduplicates rows, and writes to ignored local directory training/external/. Imported data remains supplemental until reviewed.

## Build a metadata corpus

After importing a source and preparing any additional compatible LionAI JSONL metadata files, merge and audit them:

    python training/build_metadata_corpus.py \
      --input training/external/lionai_file_organiser_metadata.jsonl \
      --input training/my_curated_metadata.jsonl

The builder writes training/external/lionai_metadata_corpus.jsonl and a class/source report. It skips identical topic/filename duplicates and rejects conflicting labels. It does **not** split or train on the result automatically.

Keep all records for a normalized topic together when creating train, validation, calibration, and test sets. Never split randomly by row: that can leak near-identical topic context across splits. The report highlights missing and underrepresented classes; resolve those gaps with curated examples before training. The external source is synthetic and English-only, so it is not a production benchmark by itself.

For reproducible Hugging Face imports, pin --revision to a dataset commit hash rather than the moving main branch.

## Safety requirements

- Only metadata supplied by the user or an explicitly configured metadata dataset may reach the model. Never inspect file contents.
- The model cannot choose arbitrary destination paths or perform filesystem operations.
- Validate classes, filenames, paths, traversal, absolute paths, and destination collisions in trusted native code.
- Default to preview-only and require explicit user confirmation before applying changes.
- Never delete or overwrite existing files. Ambiguous predictions go to Other/Review.
