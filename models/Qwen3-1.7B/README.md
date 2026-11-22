# Qwen3-1.7B base model

LionAI uses the official `Qwen/Qwen3-1.7B` checkpoint.

- Parameters: 1.7B
- License: Apache-2.0
- Official model card: https://huggingface.co/Qwen/Qwen3-1.7B

The model weights are intentionally not committed to Git. Download them into this directory from the repository root:

```bash
python -m pip install -r training/requirements-qwen3.txt
python training/download_model.py --model Qwen/Qwen3-1.7B --destination models/Qwen3-1.7B
```

The BF16 checkpoint is roughly 3.4 GB. This directory contains the actual weights only after the download script completes.
