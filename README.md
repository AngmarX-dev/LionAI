# LionAI

LionAI is an offline, task-specific file-organization model. The selected direction is a **fully from-scratch model**: its neural weights start randomly initialized, its tokenizer is custom, and it is trained on LionAI-owned metadata examples. No Qwen, DeepSeek, or other pretrained model weights are part of this design.

## From-scratch model (current development)

- **Architecture:** compact Transformer encoder classifier for file-role prediction.
- **Tokenizer:** deterministic UTF-8 byte tokenizer; no downloaded vocabulary or pretrained tokenizer.
- **Inputs:** user topic, filename and extension only. File contents are never read.
- **Outputs:** one of 13 organization classes. The native safety/organization layer remains responsible for constructing and validating paths.
- **Initialization:** random weights, reproducible seed; no pretrained checkpoint loading.
- **Training data:** `training/lionai_scratch_train.jsonl`, now expanded to 149 curated/synthetic metadata examples across all 13 classes. It remains a small development dataset, not enough to claim production accuracy.
- **Checkpoint:** `artifacts/lionai-scratch.pt` (generated locally; model weights are not committed to Git).

Install and train:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux: source .venv/bin/activate
python -m pip install -r training/requirements-scratch.txt
python training/audit_scratch_data.py
python training/train_scratch.py --epochs 30 --patience 6 --history-output artifacts/training-history.json

# After a local training run, classify one filename (metadata only):
python training/predict_scratch.py --topic "مدارک مدرسه" --filename "biology_notes.pdf"

# Evaluate on a separate test set; overlapping topics are rejected:
python training/evaluate_scratch.py --test training/lionai_scratch_eval.jsonl --output artifacts/evaluation.json

# Fit confidence calibration on a dedicated split (never reuse the final test set):
python training/calibrate_scratch.py --calibration training/lionai_scratch_calibration.jsonl

# Predict with calibrated confidence and ambiguity review:
python training/predict_scratch.py --topic "مدارک مدرسه" --filename "biology_notes.pdf" --calibration artifacts/lionai-calibration.json

# Evaluate calibration quality on a separate untouched test set:
python training/evaluate_scratch.py --test path/to/untouched-test.jsonl --calibration artifacts/lionai-calibration.json --output artifacts/evaluation-calibrated.json
```

The training script uses CUDA when available and otherwise falls back to CPU. Start with a small epoch count when testing. A successful training run does not establish generalization: expand the dataset and evaluate on a separate, topic-disjoint test set with per-class precision/recall before enabling model predictions in the organizer. The current validation split and 26-example evaluation set are development checks, not final benchmarks. The expanded training examples include synthetic metadata and must be supplemented with representative, consented real-world filename/topic pairs before drawing quality conclusions.

## Existing components

- `lionai.c`: native organizer and experimental Naive Bayes baseline.
- `lionai_hybrid.py`: metadata-only preview engine with deterministic classification and a strictly validated optional model hook. It never reads or moves files.
- `lionai_scratch.py`: custom byte tokenizer and randomly initialized Transformer classifier.
- `training/train_scratch.py`: supervised training pipeline with deterministic topic-disjoint splitting, validation-loss checkpoint selection, learning-rate scheduling, early stopping, atomic checkpoint writes and a per-epoch JSON history report.
- `training/predict_scratch.py`: safe local inference; emits a category and confidence only, never a destination path or filesystem action.
- `training/evaluate_scratch.py`: independent evaluation with per-class precision/recall/F1, confusion matrix, expected calibration error (ECE) and multiclass Brier score; optionally evaluates temperature-calibrated probabilities.
- `training/calibrate_scratch.py` and `training/calibration.py`: fit and validate scalar temperature scaling using a dedicated topic-disjoint calibration split; never use the final test set for fitting.
- Calibrated inference also checks the gap between the top two class probabilities; low confidence or a small margin routes the item to `Other/Review`.
- `training/audit_scratch_data.py`: audits dataset schemas, class coverage, duplicate samples in both datasets, filename/extension consistency, plain filenames and train/evaluation topic separation.
- `training/train_scratch.py` now uses the same strict dataset loader and a deterministic topic-disjoint split that keeps every class in both partitions.
- `training/lionai_scratch_eval.jsonl`: a small topic-disjoint starter evaluation set (26 synthetic metadata examples; not a production benchmark).
- `tests/test_lionai_scratch_evaluation.py`: validates evaluation data boundaries and metadata-only schema.
- `tests/test_lionai_scratch_inference.py`: verifies the inference safety contract.
- `training/lionai_scratch_train.jsonl`: curated starter examples containing only topic, filename, extension and label.
- `training/download_model.py` and `training/train_qwen3_qlora.py`: legacy Qwen experiment scripts; they are not used by the selected from-scratch architecture.
- Native inference integration, model calibration, independent evaluation, and safe apply/rollback remain future work.

## Safety requirements

- Only user-supplied topic and file metadata may reach the model. Never inspect file contents.
- The model predicts a category; it does not get authority to perform filesystem operations.
- Validate all predicted classes and generated paths in trusted native code.
- Reject path traversal, absolute paths, invalid filenames and destination collisions.
- Default to preview-only. Require explicit user confirmation before any apply operation.
- Never delete or overwrite files. Ambiguous predictions must be sent to `Other/Review`.
