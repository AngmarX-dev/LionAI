#!/usr/bin/env python3
"""Fit confidence temperature scaling using a dedicated metadata-only split."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lionai_scratch import LionAITokenizer
from training.audit_scratch_data import load_dataset
from training.calibration import fit_temperature, write_calibration
from training.predict_scratch import load_checkpoint


def calibrate(checkpoint_path: Path, calibration_file: Path,
              train_file: Path | None, output: Path) -> float:
    rows = load_dataset(calibration_file)
    if len(rows) < 2:
        raise ValueError("At least two calibration examples are required.")
    if train_file is not None:
        train_topics = {row["topic"].casefold().strip() for row in load_dataset(train_file)}
        calibration_topics = {row["topic"].casefold().strip() for row in rows}
        overlap = sorted(train_topics & calibration_topics)
        if overlap:
            raise ValueError("Topic leakage between training and calibration sets: " + ", ".join(overlap))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, labels, config = load_checkpoint(checkpoint_path, device)
    tokenizer = LionAITokenizer()
    logits_rows = []
    targets = []
    model.eval()
    with torch.inference_mode():
        for row in rows:
            ids, mask = tokenizer.batch([row], max_length=config.max_length)
            logits_rows.append(model(ids.to(device), mask.to(device))[0].cpu())
            targets.append(labels.index(row["label"]))
    logits = torch.stack(logits_rows)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    temperature = fit_temperature(logits, target_tensor)
    write_calibration(output, temperature, labels, len(rows))
    return temperature


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/lionai-scratch.pt")
    parser.add_argument("--calibration", required=True,
                        help="Dedicated calibration JSONL; do not reuse the final test set.")
    parser.add_argument("--train", default="training/lionai_scratch_train.jsonl")
    parser.add_argument("--output", default="artifacts/lionai-calibration.json")
    args = parser.parse_args()
    temperature = calibrate(
        Path(args.checkpoint), Path(args.calibration),
        Path(args.train) if args.train else None, Path(args.output),
    )
    print(f"temperature={temperature:.6f}")
    print(f"calibration_file={args.output}")
    print("Use a separate, untouched test set to measure calibration quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
