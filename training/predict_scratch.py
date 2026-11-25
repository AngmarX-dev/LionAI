#!/usr/bin/env python3
"""Run LionAI's from-scratch classifier on topic + filename metadata only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lionai_scratch import LionAIConfig, LionAITokenizer, build_model
from training.calibration import load_temperature

CONFIDENCE_THRESHOLD = 0.75


def load_checkpoint(path: Path, device: torch.device):
    # weights_only=True avoids arbitrary Python object deserialization.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"state_dict", "config", "labels", "tokenizer", "initialized_from"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Invalid LionAI scratch checkpoint schema.")
    if checkpoint["tokenizer"] != "utf8-byte-v1" or checkpoint["initialized_from"] != "random":
        raise ValueError("Checkpoint does not match the from-scratch LionAI format.")
    config = LionAIConfig(**checkpoint["config"])
    labels = checkpoint["labels"]
    if not isinstance(labels, list) or len(labels) != config.num_classes or not all(isinstance(x, str) for x in labels):
        raise ValueError("Checkpoint labels do not match model configuration.")
    model = build_model(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, labels, config


def predict(topic: str, filename: str, checkpoint_path: str | Path,
            threshold: float = CONFIDENCE_THRESHOLD,
            calibration_path: str | Path | None = None,
            margin_threshold: float = 0.10) -> dict:
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("topic must be a non-empty string")
    if not isinstance(filename, str) or not filename.strip() or "/" in filename or "\\" in filename:
        raise ValueError("filename must be a plain filename, not a path")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not 0.0 <= margin_threshold <= 1.0:
        raise ValueError("margin_threshold must be between 0 and 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, labels, config = load_checkpoint(Path(checkpoint_path), device)
    extension = Path(filename).suffix.casefold().lstrip(".")
    tokenizer = LionAITokenizer()
    ids, mask = tokenizer.batch(
        [{"topic": topic, "filename": filename, "extension": extension}],
        max_length=config.max_length,
    )
    temperature = load_temperature(Path(calibration_path), labels) if calibration_path else 1.0
    with torch.inference_mode():
        logits = model(ids.to(device), mask.to(device))[0]
        probabilities = torch.softmax(logits / temperature, dim=-1)
    top_probabilities, top_indices = probabilities.topk(k=2)
    confidence = top_probabilities[0]
    score = float(confidence.item())
    margin = float((top_probabilities[0] - top_probabilities[1]).item())
    label = labels[int(top_indices[0].item())]
    needs_review = score < threshold or margin < margin_threshold or label == "Other/Review"
    if needs_review:
        label = "Other/Review"
    return {
        "topic": topic,
        "original_name": filename,
        "extension": extension,
        "predicted_category": label,
        "confidence": round(score, 6),
        "confidence_margin": round(margin, 6),
        "temperature": round(temperature, 6),
        "calibrated": calibration_path is not None,
        "needs_review": needs_review,
        "model": "lionai-scratch-transformer-v1",
        "safety": {
            "metadata_only": True,
            "file_contents_read": False,
            "filesystem_modified": False,
            "path_generation_authority": False,
            "preview_only": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/lionai-scratch.pt")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--margin-threshold", type=float, default=0.10)
    parser.add_argument("--calibration", default="", help="Optional temperature calibration JSON file")
    args = parser.parse_args()
    result = predict(args.topic, args.filename, args.checkpoint, args.threshold,
                     args.calibration or None, args.margin_threshold)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
