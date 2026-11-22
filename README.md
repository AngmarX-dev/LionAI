# LionAI

LionAI is an offline file-organization assistant. The selected model is **Qwen3-1.7B** with memory-conscious 4-bit QLoRA fine-tuning. The earlier Qwen2.5-0.5B experiment remains a legacy baseline.

## Selected model

- **Model:** [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B).
- **License:** Apache-2.0.
- **Training:** 4-bit NF4 QLoRA. Base weights remain frozen; only small LoRA adapters are trained.
- **LoRA scope:** rank 8 on `q_proj` and `v_proj` by default.
- **Download location:** `models/Qwen3-1.7B/`. Model weights are intentionally not committed to Git.
- **Inference direction:** GGUF + llama.cpp for local CPU/GPU inference after fine-tuning and independent evaluation.

## Download and train

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux: source .venv/bin/activate
python -m pip install -r training/requirements-qwen3.txt
python training/download_model.py --model Qwen/Qwen3-1.7B --destination models/Qwen3-1.7B
```

The BF16 checkpoint is roughly 3.4 GB; reserve additional disk space for the download cache and training artifacts. Training requires a CUDA GPU with at least 5.5 GiB VRAM in this conservative profile. A 6 GB GPU is close to the minimum and may still run out of memory depending on other GPU workloads.

## Current implementation status

- `lionai.c`: existing native organizer and experimental Naive Bayes baseline; it is not the Qwen3 model.
- `lionai.py`: optional launcher for the native C executable.
- `training/download_model.py`: downloads and verifies Qwen3-1.7B.
- `training/train_qwen3_qlora.py`: trains only LoRA adapters; base model weights remain frozen.
- `training/lionai_sft.jsonl`: starter fine-tuning data.
- **LLM inference is not yet integrated into the native `organize` command.**
- GGUF export, quantization, native runtime integration, and LionOS integration remain future steps.

## Safety and limitations

- The model receives only the user-supplied topic and file metadata such as names/extensions; it must not inspect file contents.
- Parse and validate every JSON response. Reject malformed output, paths outside the selected root, duplicate destinations, and unsupported operations.
- Default behavior must be preview-only. Never delete or overwrite files; preserve filenames, extensions, and contents.
- Ambiguous filenames must be sent to `Other/Review` and require user review.
- The existing native organizer is still an MVP and does not yet provide rollback/undo. Back up files before applying file moves.
- Training loss is not proof of useful file organization. Run independent held-out evaluation before considering GGUF export or release.
