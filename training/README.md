# LionAI LLM training

## Model choice

Base model: `Qwen/Qwen2.5-0.5B-Instruct` (0.49B parameters, Apache-2.0). This is the single LLM selected for LionAI's first fine-tuning experiment.

## CPU-only LoRA fine-tuning

Use Python only on the development/training machine. Training runs on CPU in FP32; it can be slow. The 0.5B model is selected to keep the experiment modest, but actual RAM/time depend on the OS, PyTorch build, sequence length, and dataset size.

From the repository root:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux: source .venv/bin/activate
python -m pip install -r training/requirements.txt
python training/train_llm.py --data training/lionai_sft.jsonl --epochs 3
```

The script creates a train/evaluation split and saves a LoRA adapter and evaluation loss under `artifacts/lionai-qwen-lora/`. Do not treat training loss as proof of useful behavior; test on a separate hand-reviewed set that was not included in training.

## Important limitations

- The committed JSONL file is a small starter dataset for validating the pipeline, not enough to create a dependable specialized assistant.
- Expand it with varied, permissioned examples and hold out test examples before relying on it.
- The model sees filenames and user-provided descriptions only. Do not feed private document contents into training.
- The LLM is not yet connected to file operations. Its output must be parsed against a strict schema, path-validated, previewed, and explicitly approved before any move/rename.
- The current C Naive Bayes experiment is a separate legacy baseline; it is not the LLM and is not used by this LoRA training script.

## Native LionOS direction

For inference without Python, use a GGUF export and link a native C/C++ runtime such as `llama.cpp`. `llama.cpp` supports CPU inference and GGUF. The LionOS integration should package the model file and runtime with the OS build; Python is not needed on the target system. This is a native runtime dependency, not a single self-contained C source file.

Do not convert the adapter until training and evaluation are satisfactory. The conversion/quantization step should use the matching `llama.cpp` conversion tools and be tested against the exact pinned runtime version.
