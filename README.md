# LionAI

LionAI is an offline file-organization assistant. The new large-model direction is **Qwen3-8B** with 4-bit QLoRA fine-tuning, while the earlier Qwen2.5-0.5B experiment is retained as a legacy baseline.

## Selected model

- **Model:** [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), 8.2B parameters.
- **License:** Apache-2.0.
- **Language support:** 100+ languages, including Persian.
- **Training:** CUDA-based 4-bit QLoRA; the training workflow requires a self-hosted Linux GPU runner with at least 16 GiB VRAM.
- **Download location:** `models/Qwen3-8B/`. Model weights are large and are intentionally not committed to Git.
- **Inference direction:** GGUF + llama.cpp for local CPU/GPU inference after fine-tuning and independent evaluation.

The model card documents 32,768 native context tokens and multilingual capabilities. See the [official model card](https://huggingface.co/Qwen/Qwen3-8B).

## Download and train Qwen3-8B

From the repository root:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux: source .venv/bin/activate
python -m pip install -r training/requirements-qwen3.txt
python training/download_model.py --model Qwen/Qwen3-8B --destination models/Qwen3-8B
```

The download script verifies that the checkpoint is Qwen3 and that its SafeTensors weights are present. The full BF16 checkpoint is approximately 16–17 GB; reserve at least 22 GB of free disk space.

Training is started from **Actions → Train LionAI Qwen3-8B (QLoRA) → Run workflow** after configuring a self-hosted Linux GPU runner with the `gpu` label and at least 16 GiB of VRAM. The workflow downloads the checkpoint into `models/Qwen3-8B/` on that runner, validates the dataset, and trains a LoRA adapter. It does not silently fall back to CPU.

## Current implementation status

- `lionai.c`: existing native organizer and experimental Naive Bayes baseline; it is not the Qwen3 model.
- `lionai.py`: optional launcher for the native C executable.
- `training/download_model.py`: downloads and verifies the Qwen3-8B checkpoint.
- `training/train_qwen3_qlora.py`: CUDA 4-bit QLoRA pipeline.
- `training/lionai_sft.jsonl`: current starter fine-tuning data.
- **LLM inference is not yet integrated into the native `organize` command.**
- GGUF export, quantization, native runtime integration, and LionOS integration remain future steps.

## Safety and limitations

- The model receives only the user-supplied topic and file metadata such as names/extensions; it must not inspect file contents.
- Parse and validate every JSON response. Reject malformed output, paths outside the selected root, duplicate destinations, and unsupported operations.
- Default behavior must be preview-only. Never delete or overwrite files; preserve filenames, extensions, and contents.
- Ambiguous filenames must be sent to `Other/Review` and require user review.
- The existing native organizer is still an MVP and does not yet provide rollback/undo. Back up files before applying file moves.
- Training loss is not proof of useful file organization. Run the independent held-out evaluation before considering GGUF export or release.
