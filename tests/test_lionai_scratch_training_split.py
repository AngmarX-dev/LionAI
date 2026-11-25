import unittest

from training.audit_scratch_data import LABELS, load_dataset
from training.train_scratch import split_by_topic


class ScratchTrainingSplitTests(unittest.TestCase):
    def test_split_is_deterministic_topic_disjoint_and_class_complete(self):
        rows = load_dataset(__import__("pathlib").Path("training/lionai_scratch_train.jsonl"))
        train_a, valid_a = split_by_topic(rows, seed=2026)
        train_b, valid_b = split_by_topic(rows, seed=2026)
        self.assertEqual(train_a, train_b)
        self.assertEqual(valid_a, valid_b)
        train_topics = {row["topic"].casefold().strip() for row in train_a}
        valid_topics = {row["topic"].casefold().strip() for row in valid_a}
        self.assertFalse(train_topics & valid_topics)
        self.assertEqual({row["label"] for row in train_a}, set(LABELS))
        self.assertEqual({row["label"] for row in valid_a}, set(LABELS))

    def test_rejects_class_with_only_one_topic(self):
        rows = [{"topic": "only topic", "filename": "one.jpg",
                 "extension": "jpg", "label": label} for label in LABELS]
        with self.assertRaisesRegex(ValueError, "two distinct topics"):
            split_by_topic(rows)


if __name__ == "__main__":
    unittest.main()
