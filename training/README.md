# LionAI model training

## Selected large model: Qwen3-8B

The new candidate is the official [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) checkpoint:

- 8.2B parameters, Apache-2.0 license.
- Multilingual model supporting 100+ languages.
- 32,768-token native context.
- BF16 Hugging Face checkpoint for fine-tuning.

The previous Qwen2.5-0.5B CPU LoRA run is retained only as a legacy baseline. It is not the selected large model.

## Download into models/

Install the Qwen3-specific requirements and download the official checkpoint:

```bash
python -m pip install -r training/requirements-qwen3.txt
python training/download_model.py --model Qwen/Qwen3-8B --destination models/Qwen3-8B
```

The downloader verifies `config.json`, tokenizer files, Qwen3 model type, and the presence/size of SafeTensors weights. The full checkpoint is approximately 16–17 GB. Keep at least 22 GB of free disk space before starting. The large model files are not committed to Git.

The same download step is included in the Qwen3 training workflow, so a configured runner receives the model under `models/Qwen3-8B/` before training.

## QLoRA training requirements

An 8B model is not suitable for the previous CPU-only FP32 workflow. LionAI now uses 4-bit NF4 QLoRA through bitsandbytes.

The workflow `.github/workflows/train-qwen3-8b.yml` is manually started from GitHub Actions. It requires a **self-hosted Linux x64 GPU runner** carrying the `gpu` label and at least **16 GiB of GPU VRAM**. It checks CUDA and VRAM before loading the model; it will stop with a clear error rather than trying an unsafe CPU fallback.

The current development laptop profile available in this project context has 6 GB dedicated GPU memory, so it does not meet this training workflow's 16 GiB minimum. It can be used for testing smaller quantized inference builds, but not for this QLoRA training profile.

For direct training on a suitable GPU runner:

```bash
python -m pip install -r training/requirements-qwen3.txt
python training/download_model.py --model Qwen/Qwen3-8B --destination models/Qwen3-8B
python training/validate_dataset.py training/lionai_sft.jsonl
python training/train_qwen3_qlora.py --model models/Qwen3-8B --data training/lionai_sft.jsonl --epochs 3
```

The adapter and training metrics are written to `artifacts/lionai-qwen3-8b-qlora/`. Training loss is only a diagnostic; it does not establish file-organization accuracy.

## Dataset and evaluation

The current JSONL dataset is still a small starter set. Expand it with diverse, permissioned examples and maintain a separate held-out set. Evaluation must measure at least:

- Exact and semantic destination-path quality.
- Correct file-to-role assignment.
- Filename coverage and extension preservation.
- JSON/schema validity and path safety.
- Appropriate `Other/Review` decisions for ambiguous inputs.

Do not export GGUF until independent evaluation demonstrates meaningful improvement.

## Safety and native integration

- The model receives only a user-supplied topic and file metadata such as names/extensions. Never send file contents to the model.
- The model cannot directly move or delete files.
- Validate every generated path in native code, reject traversal/absolute paths and collisions, and show a preview before any change.
- Applying a plan must require explicit user confirmation. Never overwrite or delete existing files.
- Native inference is a later step: merge/export the adapter to GGUF, test it with a pinned llama.cpp version, and then integrate the runtime into the C/C++ application.
