import unittest
from collections import Counter

from training.audit_scratch_data import LABELS
from training.split_metadata_corpus import split_rows


def balanced_rows(topics_per_class=4):
    rows = []
    for label_index, label in enumerate(LABELS):
        for topic_index in range(topics_per_class):
            topic = f"topic-{label_index}-{topic_index}"
            rows.append({
                "topic": topic,
                "filename": f"sample-{label_index}-{topic_index}.bin",
                "extension": "bin",
                "label": label,
            })
    return rows


class MetadataSplitTests(unittest.TestCase):
    def test_uses_every_row_and_keeps_topics_disjoint(self):
        rows = balanced_rows()
        splits, report = split_rows(rows, seed=2026)
        all_rows = [row for split in splits.values() for row in split]
        self.assertCountEqual(all_rows, rows)
        topic_sets = {
            name: {row["topic"] for row in items}
            for name, items in splits.items()
        }
        self.assertFalse(topic_sets["train"] & topic_sets["validation"])
        self.assertFalse(topic_sets["train"] & topic_sets["test"])
        self.assertFalse(topic_sets["validation"] & topic_sets["test"])
        for items in splits.values():
            self.assertEqual({row["label"] for row in items}, set(LABELS))
        self.assertEqual(report["topic_overlap"], 0)
        self.assertEqual(sum(report["splits"][name]["examples"] for name in splits), len(rows))

    def test_split_is_deterministic(self):
        first, first_report = split_rows(balanced_rows(), seed=77)
        second, second_report = split_rows(balanced_rows(), seed=77)
        self.assertEqual(first, second)
        self.assertEqual(first_report, second_report)

    def test_rejects_classes_with_fewer_than_three_topics(self):
        rows = balanced_rows(topics_per_class=3)
        # Remove one independent topic for a single class.
        rows = [row for row in rows if not (
            row["label"] == LABELS[0] and row["topic"] == f"topic-0-2"
        )]
        with self.assertRaisesRegex(ValueError, "three distinct topics"):
            split_rows(rows)

    def test_report_contains_class_counts_for_every_partition(self):
        splits, report = split_rows(balanced_rows(topics_per_class=5))
        for split_name, rows_in_split in splits.items():
            counts = Counter(row["label"] for row in rows_in_split)
            self.assertEqual(report["splits"][split_name]["class_counts"], dict(sorted(counts.items())))


if __name__ == "__main__":
    unittest.main()
