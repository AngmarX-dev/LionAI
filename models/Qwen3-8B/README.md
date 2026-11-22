# Qwen3-8B base model

LionAI's new large-model candidate is the official `Qwen/Qwen3-8B` instruction model.

- Parameters: 8.2B
- License: Apache-2.0
- Native context: 32,768 tokens
- Multilingual support: 100+ languages
- Official model card: https://huggingface.co/Qwen/Qwen3-8B

The model weights are intentionally not committed to Git. Download them into this exact directory from the repository root:

```bash
python -m pip install -r training/requirements-qwen3.txt
python training/download_model.py --model Qwen/Qwen3-8B --destination models/Qwen3-8B
```

The full BF16 checkpoint is large (approximately 16–17 GB). Keep enough additional disk space for download/cache and training artifacts. This folder contains the model after the download script completes; the README alone is not the model weights.
