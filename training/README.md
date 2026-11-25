# LionAI training and data pipeline

## Selected model: LionAI Scratch

LionAI's selected architecture is a compact Transformer classifier initialized with random weights and paired with a custom UTF-8 byte tokenizer. No pretrained language-model checkpoint is downloaded or fine-tuned.

## Install

    python -m pip install -r training/requirements-scratch.txt

This installs PyTorch and Hugging Face datasets. The latter is used only to load and process metadata datasets from the Hub; it does not provide pretrained model weights.

## Import a supplemental Hugging Face dataset

    python training/import_hf_dataset.py
    python training/import_hf_dataset.py --max-rows 5000

Default source: AnodemTheGr8t/file_organiser_metadata. The importer uses datasets.load_dataset with remote code execution disabled, validates plain filenames and topic metadata, maps seven source labels to LionAI's 13 classes, and writes:

    training/external/lionai_file_organiser_metadata.jsonl

Pin a dataset commit hash with --revision for reproducibility. Downloaded corpora and trained checkpoints are not committed automatically.

### Run the import on GitHub Actions

Open the repository's **Actions** tab, select **LionAI metadata dataset import**, choose **Run workflow**, and leave `max_rows` at `0` to import the complete available split. The workflow validates and audits the converted metadata, then uploads a 14-day artifact named `lionai-metadata-dataset` containing the source JSONL, merged JSONL, and class/source report. Download that artifact from the completed workflow run. This keeps external training data out of Git history.

## Merge and audit metadata corpora

Repeat --input for each compatible LionAI JSONL file:

    python training/build_metadata_corpus.py \
      --input training/external/lionai_file_organiser_metadata.jsonl \
      --input training/my_curated_metadata.jsonl

Outputs:
- training/external/lionai_metadata_corpus.jsonl
- training/external/lionai_metadata_corpus_report.json

The builder validates each input against the LionAI metadata schema, skips identical normalized topic/filename duplicates, rejects conflicting labels, and reports class counts, missing classes, underrepresented classes, and per-source row totals. It never reads file contents. The generated external directory is ignored by Git.

The builder intentionally does not create train/validation/calibration/test splits and does not start training. Split by normalized topic groups, not by individual rows, to avoid topic leakage. Ensure every class has enough independent topic groups for every intended split before generating those splits.

## Source limitations and split policy

The first source is synthetic and English-only, and its card says labels are largely determined by file extensions. It is useful for parser and mapping tests, but it is not sufficient as a production benchmark. The card declares CC0-1.0; verify the source card before redistributing derived data. The card and dataset viewer have reported different row counts; the importer reports the number actually loaded.

The source has folder_hint, not LionAI's user-entered topic. The importer maps that field into topic to preserve contextual metadata. Consequently, imported rows remain separate from official train/evaluation files until group leakage, class coverage, and topic-disjoint split behavior are reviewed. Do not reuse the same source rows in training, calibration, and final test sets.

## Train and evaluate the scratch model

    python training/audit_scratch_data.py
    python training/train_scratch.py --epochs 30 --patience 6
    python training/evaluate_scratch.py --test training/lionai_scratch_eval.jsonl --output artifacts/evaluation.json

Use a dedicated calibration split for temperature fitting; never fit calibration on the final test set.

## Safety boundary

The pipeline handles metadata only. It must never read file contents, create or apply destination paths, delete files, or overwrite files. All filesystem operations remain outside the model and require native validation plus explicit user confirmation.
