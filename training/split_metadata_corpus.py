#!/usr/bin/env python3
"""Create deterministic topic-disjoint train/validation/test splits from metadata JSONL."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.audit_scratch_data import LABELS, load_dataset


def split_rows(rows, seed=2026, validation_fraction=0.1, test_fraction=0.1):
    """Split whole normalized topics while preserving every class in all splits."""
    if (validation_fraction <= 0 or test_fraction <= 0
            or validation_fraction + test_fraction >= 0.5):
        raise ValueError("validation/test fractions must be positive and sum to less than 0.5")

    groups = defaultdict(list)
    for row in rows:
        groups[row["topic"].casefold().strip()].append(row)
    topic_labels = {
        topic: {row["label"] for row in group}
        for topic, group in groups.items()
    }
    label_topics = {
        label: {topic for topic, labels in topic_labels.items() if label in labels}
        for label in LABELS
    }
    absent = [label for label, topics in label_topics.items() if len(topics) < 3]
    if absent:
        raise ValueError(
            "At least three distinct topics per class are required: " + ", ".join(absent)
        )

    rng = random.Random(seed)
    topic_order = {topic: rng.random() for topic in groups}
    group_sizes = {topic: len(group) for topic, group in groups.items()}
    total_rows = len(rows)
    assignment = {}

    def can_assign(topic):
        # Keep at least one remaining topic per label for the training partition.
        for label in topic_labels[topic]:
            remaining = sum(
                1 for candidate in label_topics[label]
                if candidate not in assignment and candidate != topic
            )
            if remaining < 1:
                return False
        return True

    # Reserve coverage for every class in both held-out partitions. Prefer
    # small groups so broad folders cannot consume an entire split.
    for split in ("test", "validation"):
        for label in sorted(LABELS, key=lambda item: (len(label_topics[item]), item)):
            if any(
                label in topic_labels[topic]
                for topic, assigned_split in assignment.items()
                if assigned_split == split
            ):
                continue
            candidates = [
                topic for topic in label_topics[label]
                if topic not in assignment and can_assign(topic)
            ]
            if not candidates:
                raise ValueError(
                    f"Cannot reserve a {split} topic for {label} without losing train coverage"
                )
            chosen = min(
                candidates,
                key=lambda topic: (
                    group_sizes[topic],
                    -sum(1 / len(label_topics[item]) for item in topic_labels[topic]),
                    topic_order[topic],
                ),
            )
            assignment[chosen] = split

    # Optimize per-class example counts and total rows rather than topic counts:
    # folder_hint groups can differ by orders of magnitude in size.
    class_totals = Counter(row["label"] for row in rows)
    topic_counts = {
        topic: Counter(row["label"] for row in group)
        for topic, group in groups.items()
    }

    def objective(split, fraction):
        selected = [
            topic for topic, assigned_split in assignment.items()
            if assigned_split == split
        ]
        actual = Counter()
        total = 0
        for topic in selected:
            actual.update(topic_counts[topic])
            total += group_sizes[topic]
        score = 0.25 * (
            (total - total_rows * fraction) ** 2 / max(total_rows * fraction, 1)
        )
        for label in LABELS:
            target = class_totals[label] * fraction
            score += (actual[label] - target) ** 2 / max(target, 1)
        return score

    for split, fraction in (
        ("test", test_fraction),
        ("validation", validation_fraction),
    ):
        while True:
            before = objective(split, fraction)
            candidates = []
            for topic in groups:
                if topic in assignment or not can_assign(topic):
                    continue
                assignment[topic] = split
                after = objective(split, fraction)
                del assignment[topic]
                improvement = before - after
                if improvement > 1e-9:
                    candidates.append((improvement, topic_order[topic], topic))
            if not candidates:
                break
            assignment[max(candidates)[2]] = split

    for topic in groups:
        assignment.setdefault(topic, "train")

    result = {name: [] for name in ("train", "validation", "test")}
    for topic, group in groups.items():
        result[assignment[topic]].extend(group)

    for split, split_rows_ in result.items():
        missing = sorted(set(LABELS) - {row["label"] for row in split_rows_})
        if missing:
            raise ValueError(f"{split} split missing classes: {missing}")

    topic_sets = {
        split: {row["topic"].casefold().strip() for row in split_rows_}
        for split, split_rows_ in result.items()
    }
    for first, second in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if topic_sets[first] & topic_sets[second]:
            raise AssertionError("Topic leakage detected between splits")

    report = {
        "seed": seed,
        "source_examples": len(rows),
        "source_topics": len(groups),
        "fractions_by_topics": {
            split: round(len(topic_sets[split]) / len(groups), 4)
            for split in result
        },
        "fractions_by_examples": {
            split: round(len(result[split]) / len(rows), 4)
            for split in result
        },
        "splits": {},
        "topic_overlap": 0,
        "metadata_only": True,
        "file_contents_used": False,
        "warning": (
            "The source is synthetic; these metrics describe this corpus distribution, "
            "not real-world readiness."
        ),
    }
    for split, split_rows_ in result.items():
        report["splits"][split] = {
            "examples": len(split_rows_),
            "topics": len(topic_sets[split]),
            "class_counts": dict(sorted(Counter(
                row["label"] for row in split_rows_
            ).items())),
        }
    return result, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="training/external/splits")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()

    rows = load_dataset(Path(args.input))
    splits, report = split_rows(
        rows, args.seed, args.validation_fraction, args.test_fraction
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        destination = output_dir / f"{name}.jsonl"
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in items
            ),
            encoding="utf-8",
        )
        temporary.replace(destination)

    report["output_dir"] = str(output_dir)
    report_path = output_dir / "split-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
