# LionAI model training

## Selected model: Qwen3-1.7B

The selected candidate is the official [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) checkpoint.

- 1.7B parameters; Apache-2.0 license.
- BF16 Hugging Face checkpoint for fine-tuning.
- 4-bit NF4 QLoRA: base model weights are frozen; only LoRA adapters are updated.
- Default adapter: rank 8, target modules `q_proj` and `v_proj`.

The previous Qwen2.5-0.5B CPU LoRA run is retained only as a legacy baseline.

## Download into models/

```bash
python -m pip install -r training/requirements-qwen3.txt
python training/download_model.py --model Qwen/Qwen3-1.7B --destination models/Qwen3-1.7B
```

The downloader verifies the Qwen3 architecture, tokenizer, and SafeTensors files. The BF16 checkpoint is roughly 3.4 GB. Model weights are not committed to Git.

## QLoRA training requirements

The workflow `.github/workflows/train-qwen3-1.7b.yml` is manually started from GitHub Actions. It requires a self-hosted Linux x64 CUDA GPU runner with at least 5.5 GiB VRAM. A 6 GB GPU is near the minimum and may still run out of memory depending on drivers and other GPU workloads. The script fails clearly instead of silently falling back to CPU.

```bash
python -m pip install -r training/requirements-qwen3.txt
python training/download_model.py --model Qwen/Qwen3-1.7B --destination models/Qwen3-1.7B
python training/validate_dataset.py training/lionai_sft.jsonl training/lionai_eval.jsonl
python training/train_qwen3_qlora.py --model models/Qwen3-1.7B --data training/lionai_sft.jsonl --output artifacts/lionai-qwen3-1.7b-qlora --epochs 3
```

Only adapter parameters are trained; the base checkpoint is not rewritten. Adapter and metrics are written to `artifacts/lionai-qwen3-1.7b-qlora/`.

## Dataset and evaluation

Maintain a separate held-out set and evaluate semantic destination-path quality, file-to-role assignment, filename coverage, extension preservation, JSON/schema validity, path safety, and appropriate `Other/Review` decisions. Do not export GGUF until independent evaluation demonstrates meaningful improvement.

## Safety and native integration

- The model receives only user-supplied topic and file metadata such as names/extensions. Never send file contents to the model.
- Validate every generated path in native code; reject traversal, absolute paths, and collisions.
- Show a preview and require explicit user confirmation before applying changes.
- Never overwrite or delete existing files.
- Native inference is a later step: export/quantize the adapter-compatible model to GGUF, test with a pinned llama.cpp version, then integrate the runtime into the C/C++ application.
