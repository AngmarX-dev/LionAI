# LionAI

LionAI is an offline, task-specific file-organization model. The selected direction is a **fully from-scratch model**: its neural weights start randomly initialized, its tokenizer is custom, and it is trained on LionAI-owned metadata examples. No Qwen, DeepSeek, or other pretrained model weights are part of this design.

## From-scratch model (current development)

- **Architecture:** compact Transformer encoder classifier for file-role prediction.
- **Tokenizer:** deterministic UTF-8 byte tokenizer; no downloaded vocabulary or pretrained tokenizer.
- **Inputs:** user topic, filename and extension only. File contents are never read.
- **Outputs:** one of 13 organization classes. The native safety/organization layer remains responsible for constructing and validating paths.
- **Initialization:** random weights, reproducible seed; no pretrained checkpoint loading.
- **Training data:** `training/lionai_scratch_train.jsonl`, a small curated starter set. It is a development dataset, not enough to claim production accuracy.
- **Checkpoint:** `artifacts/lionai-scratch.pt` (generated locally; model weights are not committed to Git).

Install and train:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux: source .venv/bin/activate
python -m pip install -r training/requirements-scratch.txt
python training/train_scratch.py --epochs 30
```

The training script uses CUDA when available and otherwise falls back to CPU. Start with a small epoch count when testing. A successful training run does not establish generalization: expand the dataset, create a topic-disjoint held-out test set, and evaluate per-class precision/recall before enabling model predictions in the organizer.

## Existing components

- `lionai.c`: native organizer and experimental Naive Bayes baseline.
- `lionai_hybrid.py`: metadata-only preview engine with deterministic classification and a strictly validated optional model hook. It never reads or moves files.
- `lionai_scratch.py`: custom byte tokenizer and randomly initialized Transformer classifier.
- `training/train_scratch.py`: supervised training pipeline for the from-scratch model.
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
