# LionAI LLM training

## Model choice

Base model: `Qwen/Qwen2.5-0.5B-Instruct` (0.49B parameters, Apache-2.0). This is the single LLM selected for LionAI's first fine-tuning experiment.

## Download the base model with GitHub Actions

The repository includes `.github/workflows/download-model.yml`. To run it:

1. Open the LionAI repository on GitHub and select **Actions**.
2. Select **Download LionAI base model**.
3. Choose **Run workflow**; keep revision as `main` unless a specific model revision is required.
4. Wait for the `download-model` job to finish successfully.
5. Open that workflow run's **Artifacts** section and download `lionai-qwen2.5-0.5b-instruct`.

The workflow downloads the model from the official Hugging Face repository into a dedicated folder, `models/Qwen2.5-0.5B-Instruct/`. It verifies the configuration and SafeTensors weights, then uploads the parent `models/` directory so the artifact preserves this folder structure:

```text
lionai-qwen2.5-0.5b-instruct.zip
└── Qwen2.5-0.5B-Instruct/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

After extracting the artifact's `Qwen2.5-0.5B-Instruct/` folder into the repository's `models/` directory, the model path will be `models/Qwen2.5-0.5B-Instruct/`. The artifact is retained for 7 days. GitHub Actions also caches the downloaded files between compatible runs. The large model weights are deliberately not committed to Git.

Then pass that local directory to the training script with `--model` (see below). The model download workflow only downloads and packages the base model; it does not start training automatically.

## CPU-only LoRA fine-tuning

Use Python only on the development/training machine. Training runs on CPU in FP32; it can be slow. The 0.5B model is selected to keep the experiment modest, but actual RAM/time depend on the OS, PyTorch build, sequence length, and dataset size.

From the repository root:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux: source .venv/bin/activate
python -m pip install -r training/requirements.txt
python training/train_llm.py --model models/Qwen2.5-0.5B-Instruct --data training/lionai_sft.jsonl --epochs 3
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
