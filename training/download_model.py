#!/usr/bin/env python3
"""Download and verify the official Qwen3-1.7B checkpoint into models/."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--destination", default="models/Qwen3-1.7B")
    parser.add_argument("--minimum-free-gb", type=float, default=6.0)
    args = parser.parse_args()

    destination = Path(args.destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(destination.parent).free / (1024 ** 3)
    if free_gb < args.minimum_free_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GiB free; need at least {args.minimum_free_gb:.1f} GiB."
        )

    print(f"Downloading {args.model}@{args.revision} to {destination}")
    snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        local_dir=str(destination),
        allow_patterns=ALLOW_PATTERNS,
    )

    config_path = destination / "config.json"
    tokenizer_path = destination / "tokenizer.json"
    if not config_path.is_file() or not tokenizer_path.is_file():
        raise RuntimeError("Download is incomplete: config.json/tokenizer.json missing.")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        raise RuntimeError(f"Unexpected model_type={config.get('model_type')!r}; expected 'qwen3'.")
    if config.get("num_hidden_layers", 0) > 30:
        raise RuntimeError("Unexpected model size: expected the Qwen3-1.7B architecture.")

    weight_files = sorted(destination.glob("*.safetensors"))
    total_bytes = sum(path.stat().st_size for path in weight_files)
    if not weight_files or total_bytes < 2 * (1024 ** 3):
        raise RuntimeError(f"Checkpoint weights look incomplete ({total_bytes / (1024 ** 3):.2f} GiB).")

    print(f"Verified model directory: {destination}")
    print(f"Safetensors files: {len(weight_files)}")
    print(f"Total weight size: {total_bytes / (1024 ** 3):.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
