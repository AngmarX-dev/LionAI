import unittest
from collections import Counter

from training.audit_scratch_data import LABELS
from training.split_metadata_corpus import split_rows


class MetadataSplitTests(unittest.TestCase):
    def make_rows(self):
        rows = []
        for label_index, label in enumerate(LABELS):
            for topic_index in range(3):
                topic = f"topic-{label_index}-{topic_index}"
                extension = f"x{label_index}{topic_index}"
                rows.append({
                    "topic": topic,
                    "filename": f"sample.{extension}",
                    "extension": extension,
                    "label": label,
                })
        return rows

    def test_all_rows_are_preserved_and_topics_do_not_leak(self):
        rows = self.make_rows()
        splits, report = split_rows(rows, seed=42)
        self.assertEqual(sum(len(part) for part in splits.values()), len(rows))
        self.assertEqual(report["topic_overlap"], 0)
        topic_sets = {
            name: {row["topic"].casefold().strip() for row in part}
            for name, part in splits.items()
        }
        self.assertFalse(topic_sets["train"] & topic_sets["validation"])
        self.assertFalse(topic_sets["train"] & topic_sets["test"])
        self.assertFalse(topic_sets["validation"] & topic_sets["test"])
        for part in splits.values():
            self.assertEqual({row["label"] for row in part}, set(LABELS))

    def test_split_is_deterministic(self):
        rows = self.make_rows()
        first, _ = split_rows(rows, seed=2026)
        second, _ = split_rows(rows, seed=2026)
        self.assertEqual(first, second)

    def test_missing_class_topic_coverage_is_rejected(self):
        rows = self.make_rows()
        rows = [row for row in rows if row["label"] != "Reports" or row["topic"] == "topic-11-0"]
        with self.assertRaisesRegex(ValueError, "three distinct topics"):
            split_rows(rows)


if __name__ == "__main__":
    unittest.main()
