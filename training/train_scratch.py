#!/usr/bin/env python3
"""Train LionAI's task-specific Transformer from random initialization."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lionai_scratch import LionAIConfig, LionAITokenizer, build_model
from training.audit_scratch_data import load_dataset

LABELS = ["Images", "Documents", "Data", "Audio", "Video", "Archives",
          "Source", "Config", "Installers", "Fonts", "Databases", "Reports", "Other/Review"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def split_by_topic(rows, seed: int = 2026):
    """Create deterministic, topic-disjoint splits with every label in both sets."""
    topic_groups = {}
    for row in rows:
        topic_groups.setdefault(row["topic"].casefold().strip(), []).append(row)
    topics_by_label = {label: [] for label in LABELS}
    for topic, group in topic_groups.items():
        for label in {row["label"] for row in group}:
            topics_by_label[label].append(topic)
    for label, topics_for_label in topics_by_label.items():
        if len(topics_for_label) < 2:
            raise ValueError(f"At least two distinct topics are required for class {label!r}.")
    rng = random.Random(seed)
    validation_topics = set()
    for label in LABELS:
        candidates = topics_by_label[label][:]
        rng.shuffle(candidates)
        validation_topics.add(candidates[0])
    all_topics = list(topic_groups)
    rng.shuffle(all_topics)
    target = max(len(validation_topics), min(len(all_topics) - 1, round(len(all_topics) * 0.2)))
    for topic in all_topics:
        if len(validation_topics) >= target:
            break
        validation_topics.add(topic)
    train_rows = [row for topic, group in topic_groups.items()
                  if topic not in validation_topics for row in group]
    valid_rows = [row for topic, group in topic_groups.items()
                  if topic in validation_topics for row in group]
    for split_name, split_rows in (("train", train_rows), ("validation", valid_rows)):
        present = {row["label"] for row in split_rows}
        missing = set(LABELS) - present
        if missing:
            raise ValueError(f"{split_name} split is missing classes: " + ", ".join(sorted(missing)))
    return train_rows, valid_rows


class MetadataDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows, self.tokenizer = rows, tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        ids, mask = self.tokenizer.batch([row])
        return ids[0], mask[0], LABEL_TO_ID[row["label"]]


def collate(batch):
    ids, masks, labels = zip(*batch)
    width = max(item.numel() for item in ids)
    padded = torch.zeros((len(ids), width), dtype=torch.long)
    mask = torch.zeros((len(ids), width), dtype=torch.bool)
    for i, (tokens, valid) in enumerate(zip(ids, masks)):
        padded[i, :tokens.numel()] = tokens
        mask[i, :valid.numel()] = valid
    return padded, mask, torch.tensor(labels, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="training/lionai_scratch_train.jsonl")
    parser.add_argument("--output", default="artifacts/lionai-scratch.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--history-output", default="artifacts/training-history.json")
    parser.add_argument("--validation-data", default=None,
                        help="Optional independent topic-disjoint validation JSONL; when set, train on all rows in --data")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size and patience must be positive; learning-rate must be greater than zero")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = load_dataset(Path(args.data))
    if len(rows) < 20:
        raise SystemExit("At least 20 metadata examples are required before training.")
    if args.validation_data:
        train_rows = rows
        valid_rows = load_dataset(Path(args.validation_data))
        train_topics = {row["topic"].casefold().strip() for row in train_rows}
        valid_topics = {row["topic"].casefold().strip() for row in valid_rows}
        overlap = sorted(train_topics & valid_topics)
        if overlap:
            raise SystemExit("Train/validation topic leakage: " + ", ".join(overlap[:10]))
        for split_name, split_rows in (("train", train_rows), ("validation", valid_rows)):
            present = {row["label"] for row in split_rows}
            missing = set(LABELS) - present
            if missing:
                raise SystemExit(f"{split_name} split is missing classes: " + ", ".join(sorted(missing)))
    else:
        train_rows, valid_rows = split_by_topic(rows, args.seed)
    validation_topics = len({row["topic"].casefold().strip() for row in valid_rows})
    print(f"train_examples={len(train_rows)} validation_examples={len(valid_rows)} validation_topics={validation_topics}")
    tokenizer = LionAITokenizer()
    train_loader = DataLoader(MetadataDataset(train_rows, tokenizer), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(MetadataDataset(valid_rows, tokenizer), batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate)

    config = LionAIConfig(num_classes=len(LABELS))
    model = build_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    class_counts = torch.tensor(
        [sum(row["label"] == label for row in train_rows) for label in LABELS],
        dtype=torch.float32, device=device,
    )
    if torch.any(class_counts == 0):
        missing = [LABELS[i] for i, count in enumerate(class_counts.tolist()) if count == 0]
        raise SystemExit("Training data is missing classes: " + ", ".join(missing))
    # Mild inverse-square-root weighting reduces domination by frequent synthetic labels
    # without letting tiny classes produce extreme gradients.
    class_weights = torch.sqrt(class_counts.sum() / (len(LABELS) * class_counts))
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    best_validation_loss = float("inf")
    best_accuracy = 0.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for ids, mask, labels in train_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(ids, mask), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        correct = count = 0
        validation_loss_total = 0.0
        with torch.inference_mode():
            for ids, mask, labels in valid_loader:
                logits = model(ids.to(device), mask.to(device))
                validation_loss_total += criterion(logits, labels.to(device)).item() * labels.numel()
                correct += (logits.argmax(dim=1).cpu() == labels).sum().item()
                count += labels.numel()
        validation_loss = validation_loss_total / max(1, count)
        accuracy = correct / count if count else 0.0
        scheduler.step(validation_loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(1, len(train_loader)), 6),
            "validation_loss": round(validation_loss, 6),
            "validation_accuracy": round(accuracy, 6),
            "learning_rate": learning_rate,
        }
        history.append(epoch_record)
        history_path = Path(args.history_output)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_tmp = history_path.with_name(history_path.name + ".tmp")
        history_tmp.write_text(json.dumps({
            "model": "lionai-scratch-transformer-v1",
            "seed": args.seed,
            "initialized_from": "random",
            "history": history,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        history_tmp.replace(history_path)
        print(
            f"epoch={epoch} train_loss={total_loss / max(1, len(train_loader)):.4f} "
            f"validation_loss={validation_loss:.4f} validation_accuracy={accuracy:.3f} "
            f"learning_rate={learning_rate:.2e}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_accuracy = accuracy
            epochs_without_improvement = 0
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_state = {key: value.detach().cpu().clone()
                                for key, value in model.state_dict().items()}
            checkpoint = {
                "state_dict": checkpoint_state,
                "config": config.__dict__,
                "labels": LABELS,
                "tokenizer": "utf8-byte-v1",
                "initialized_from": "random",
                "training_examples": len(train_rows),
                "training_class_counts": {label: int(class_counts[i].item()) for i, label in enumerate(LABELS)},
                "validation_data": args.validation_data,
                "validation_examples": len(valid_rows),
                "validation_loss": best_validation_loss,
                "validation_accuracy": best_accuracy,
                "seed": args.seed,
                "epoch": epoch,
            }
            checkpoint_tmp = output.with_name(output.name + ".tmp")
            torch.save(checkpoint, checkpoint_tmp)
            checkpoint_tmp.replace(output)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early_stopping=1 epoch={epoch} patience={args.patience}")
                break

    print(f"best_validation_loss={best_validation_loss:.4f}")
    print(f"best_validation_accuracy={best_accuracy:.3f}")
    print("Reminder: validate on a separate, topic-disjoint test set before using predictions.")


if __name__ == "__main__":
    main()
