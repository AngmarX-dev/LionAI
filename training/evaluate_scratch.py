#!/usr/bin/env python3
"""Evaluate a LionAI scratch checkpoint on an independent metadata-only JSONL set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lionai_scratch import LionAITokenizer
from training.predict_scratch import load_checkpoint
from training.audit_scratch_data import LABELS, load_dataset


def read_rows(path: Path) -> list[dict[str, str]]:
    return load_dataset(path)


def evaluate(checkpoint: Path, test_file: Path, train_file: Path | None = None) -> dict:
    rows = read_rows(test_file)
    if train_file:
        train_topics = {row["topic"].casefold().strip() for row in read_rows(train_file)}
        overlap = sorted(train_topics & {row["topic"].casefold().strip() for row in rows})
        if overlap:
            raise ValueError("Topic leakage between training and evaluation sets: " + ", ".join(overlap))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, labels, config = load_checkpoint(checkpoint, device)
    tokenizer = LionAITokenizer()
    matrix = [[0 for _ in labels] for _ in labels]
    model.eval()
    with torch.inference_mode():
        for row in rows:
            ids, mask = tokenizer.batch([row], max_length=config.max_length)
            predicted_id = int(model(ids.to(device), mask.to(device)).argmax(dim=-1).item())
            actual_id = labels.index(row["label"])
            matrix[actual_id][predicted_id] += 1

    per_class = {}
    f1_values = []
    for index, label in enumerate(labels):
        tp = matrix[index][index]
        fp = sum(matrix[actual][index] for actual in range(len(labels)) if actual != index)
        fn = sum(matrix[index][predicted] for predicted in range(len(labels)) if predicted != index)
        support = sum(matrix[index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            f1_values.append(f1)
        per_class[label] = {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "support": support,
        }
    accuracy = sum(matrix[i][i] for i in range(len(labels))) / len(rows)
    return {
        "model": "lionai-scratch-transformer-v1",
        "evaluation_examples": len(rows),
        "accuracy": round(accuracy, 4),
        "macro_f1_present_classes": round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "rows_actual_columns_predicted": matrix},
        "safety": {"metadata_only": True, "file_contents_read": False, "filesystem_modified": False},
        "warning": "Metrics on a small test set are descriptive only; do not treat them as production readiness.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/lionai-scratch.pt")
    parser.add_argument("--test", required=True)
    parser.add_argument("--train", default="training/lionai_scratch_train.jsonl")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = evaluate(Path(args.checkpoint), Path(args.test), Path(args.train) if args.train else None)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
