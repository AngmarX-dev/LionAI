# LionAI training and data pipeline

## Selected model: LionAI Scratch

LionAI's selected architecture is a compact Transformer classifier initialized with random weights and paired with a custom UTF-8 byte tokenizer. No pretrained language-model checkpoint is downloaded or fine-tuned.

## Install

    python -m pip install -r training/requirements-scratch.txt

This installs PyTorch and Hugging Face datasets. The latter is used only to load and process metadata datasets from the Hub; it does not provide pretrained model weights.

## Import the complete supplemental Hugging Face dataset

    python training/import_hf_dataset.py
    python training/import_hf_dataset.py --max-rows 5000

Default source: AnodemTheGr8t/file_organiser_metadata. The importer uses datasets.load_dataset with remote code execution disabled, validates plain filenames and topic metadata, maps seven source labels to LionAI's 13 classes, and writes:

    training/external/lionai_file_organiser_metadata.jsonl

The default `--max-rows 0` imports all available rows. Pin a dataset commit hash with `--revision` for reproducibility. Downloaded corpora and trained checkpoints are not committed automatically.

## Import real-world GitHub filename metadata

    python training/import_github_metadata.py --max-files-per-repo 1500

The source list is maintained in `training/github_metadata_sources.txt`. The collector queries only public repository metadata and recursive Git trees, accepts repositories with an allowlisted SPDX license, records the license and tree revision in a provenance report, and never fetches file blobs or contents. It samples up to 1,500 distinct basenames per repository to bound API and training costs. Each repository is one topic group so files from the same project cannot leak across train/validation/test. Labels are inferred from extensions and report-like filenames; review the source report and per-class counts because extension-based labels are weak supervision, not human annotations.

## Merge, audit, and split the maximum available corpus

    python training/build_metadata_corpus.py \
      --input training/lionai_scratch_train.jsonl \
      --input training/external/lionai_file_organiser_metadata.jsonl

Then create deterministic topic-disjoint splits:

    python training/split_metadata_corpus.py \
      --input training/external/lionai_metadata_corpus.jsonl \
      --output-dir training/external/splits \
      --validation-fraction 0.10 \
      --test-fraction 0.10

The splitter uses every accepted row exactly once, keeps all records sharing a normalized topic in one split, preserves every class in train/validation/test, and optimizes class and total row balance. Because topic groups vary greatly in size, exact 80/10/10 ratios are not guaranteed. It reports actual counts and class coverage in `split-report.json`. It refuses to split a class with fewer than three independent topics.

## Train and evaluate

    python training/train_scratch.py \
      --data training/external/splits/train.jsonl \
      --validation-data training/external/splits/validation.jsonl \
      --epochs 20 --patience 5

    python training/evaluate_scratch.py \
      --checkpoint artifacts/lionai-scratch.pt \
      --test training/external/splits/test.jsonl \
      --train training/external/splits/train.jsonl \
      --output artifacts/evaluation-report.json

The test split is not used for weight selection. Training defaults to deterministic square-root class-aware sampling (`--sampling-power 0.5`): this reduces domination by common classes without giving the rarest classes the same total sampling mass as the largest classes. Use `--sampling-power 0` for ordinary uniform-row sampling or `1` for equal expected class mass. The sampler uses a seeded generator and replacement sampling, while validation and test distributions remain untouched. Report accuracy, per-class Precision/Recall/F1, macro F1, calibration error, Brier score, and confusion matrix. Compare the new checkpoint against the previous run on the same untouched test split; do not claim improvement unless macro F1 and rare-class recall actually improve. Do not fit temperature calibration on the final test split.

## GitHub Actions

The **LionAI metadata dataset import and full-corpus training** workflow imports all available Hugging Face rows plus up to 1,500 metadata records per eligible GitHub repository, audits the merged corpus, creates topic-disjoint splits, trains from random initialization, evaluates once on the untouched test split, and uploads the dataset, split report, model, history, and evaluation report as a 14-day artifact. The artifact keeps large generated data and model weights out of Git history.

## Source limitations

The Hugging Face source is synthetic and English-only. The GitHub tree source contains real public filenames from permissively licensed repositories, but is biased toward software projects and its extension-derived labels are weak supervision. Its labels are largely determined by file extensions, and its `folder_hint` field becomes topic metadata. It is useful for engineering tests, but it is not a realistic production benchmark. The source card declares CC0-1.0; verify the card before redistribution. The card and viewer have shown inconsistent row counts, so always trust the importer's actual `rows_written` count.

The 13-class mapping is imbalanced: rare classes such as Reports, Fonts, Installers, and Databases have few independent topics. Class-aware sampling mitigates training bias but does not create new information or replace collecting more independently licensed, relevant, human-verified examples. The GitHub labels remain weakly supervised, so improvements on this corpus are not proof of production accuracy.

## Safety boundary

The pipeline handles metadata only. It must never read file contents, create or apply destination paths, delete files, or overwrite files. All filesystem operations remain outside the model and require native validation plus explicit user confirmation.
