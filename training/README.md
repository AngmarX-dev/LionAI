# LionAI training and data pipeline

## Selected model: LionAI Scratch

LionAI's selected architecture is a compact Transformer classifier initialized with random weights and paired with a custom UTF-8 byte tokenizer. No pretrained language-model checkpoint is downloaded or fine-tuned.

## Install

    python -m pip install -r training/requirements-scratch.txt

This installs PyTorch and Hugging Face datasets. The latter is used only to load and process metadata datasets from the Hub; it does not provide pretrained model weights.

## Import a supplemental Hugging Face dataset

    python training/import_hf_dataset.py

Default source: AnodemTheGr8t/file_organiser_metadata.

The importer uses datasets.load_dataset, validates plain filenames, deduplicates (folder_hint, file_name), maps the source's seven labels to LionAI's 13 categories, and writes:

    training/external/lionai_file_organiser_metadata.jsonl

Optional arguments:

    python training/import_hf_dataset.py --dataset AnodemTheGr8t/file_organiser_metadata --split train --revision DATASET_COMMIT_SHA --output training/external/lionai_file_organiser_metadata.jsonl

Pin --revision to a commit SHA for reproducibility. Do not commit downloaded/generated dataset files automatically.

## Source limitations and split policy

The source dataset is synthetic, English-only, and its card says labels are largely determined by file extensions. It is useful for parser and mapping tests, but it is not sufficient as a production benchmark. The card declares CC0-1.0; verify the source card before redistributing derived data. The card and dataset viewer currently disagree on row count; the importer reports the number actually loaded.

The source has folder_hint, not LionAI's user-entered topic. The importer maps that field into topic to preserve contextual metadata. Consequently, the output is kept separate from official training/evaluation files until group leakage, class coverage, and topic-disjoint split behavior are reviewed. Do not use the same source rows in training, calibration, and final test sets.

## Train and evaluate the scratch model

    python training/audit_scratch_data.py
    python training/train_scratch.py --epochs 30 --patience 6
    python training/evaluate_scratch.py --test training/lionai_scratch_eval.jsonl --output artifacts/evaluation.json

Use a dedicated calibration split for temperature fitting; never fit calibration on the final test set.

## Safety boundary

The pipeline handles metadata only. It must never read file contents, create or apply destination paths, delete files, or overwrite files. All filesystem operations remain outside the model and require native validation plus explicit user confirmation.
