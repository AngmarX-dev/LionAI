# LionAI

LionAI is an offline file-organization assistant project. Its selected AI direction is **one fine-tuned LLM**: `Qwen/Qwen2.5-0.5B-Instruct`, adapted for cautious file organization using LoRA.

## Selected AI architecture

- **Single model:** Qwen2.5-0.5B-Instruct (0.49B parameters; Apache-2.0).
- **Training:** Python + PyTorch + Transformers + PEFT/LoRA, configured for CPU-only FP32 training.
- **Training data:** `training/lionai_sft.jsonl` (starter examples; expand and validate before production use).
- **Native inference direction:** export the trained model to GGUF and run it with a native C/C++ runtime such as llama.cpp. Python is not required on the target OS once the model and runtime are packaged.
- **File safety:** the model only receives user-supplied topics and file metadata such as names/extensions. It must not modify files directly. Every proposed operation must pass strict validation and user preview/approval.

## Train the LLM

See [training/README.md](training/README.md).

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux: source .venv/bin/activate
python -m pip install -r training/requirements.txt
python training/train_llm.py --data training/lionai_sft.jsonl --epochs 3
```

Training is a CPU-only experiment and can take considerable time. It has not been run or validated automatically by this repository update. The starter dataset is not sufficient for production quality; collect diverse, permissioned examples and maintain a separate test set.

## Current implementation status

- `lionai.c`: existing native file organizer and an experimental Naive Bayes baseline. The Naive Bayes commands are **not** the selected LLM.
- `lionai.py`: optional launcher for the existing native C executable.
- `training/train_llm.py`: CPU LoRA fine-tuning pipeline for the selected Qwen LLM.
- `training/lionai_sft.jsonl`: starter instruction dataset.
- LLM inference has not yet been integrated into `organize`.
- GGUF conversion, quantization, native runtime packaging, and LionOS integration remain future steps.

## Safety and limitations

- The LLM's JSON output must be parsed and schema-validated. Reject malformed output, paths outside the selected root, duplicate destinations, and unsupported operations.
- Default behavior must be preview-only. Never delete or overwrite existing files; preserve extensions and file contents.
- Ambiguous file names must be marked for user review.
- The current organizer scans only top-level files and has no rollback/undo yet. Back up files before applying existing organizer operations.
- Do not claim semantic file understanding based on file contents: current training examples use names and metadata only.
