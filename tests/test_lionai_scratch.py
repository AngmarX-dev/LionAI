import unittest

from lionai_scratch import LionAITokenizer, SEP_ID, VOCAB_SIZE


class ScratchTokenizerTests(unittest.TestCase):
    def test_utf8_bytes_and_separator_are_in_vocabulary(self):
        tokenizer = LionAITokenizer()
        ids = tokenizer.encode("پروژه عکس", "تصویر_01.png", ".png")
        self.assertTrue(ids)
        self.assertTrue(all(0 < token < VOCAB_SIZE for token in ids))
        self.assertIn(SEP_ID, ids)

    def test_different_metadata_produces_different_tokens(self):
        tokenizer = LionAITokenizer()
        first = tokenizer.encode("school", "notes.pdf", ".pdf")
        second = tokenizer.encode("music", "notes.pdf", ".pdf")
        self.assertNotEqual(first, second)

    def test_input_types_are_checked(self):
        with self.assertRaises(TypeError):
            LionAITokenizer().encode("topic", None, ".txt")

    def test_max_length_is_respected(self):
        tokens = LionAITokenizer().encode("x" * 100, "y" * 100, ".txt", max_length=32)
        self.assertEqual(len(tokens), 32)


if __name__ == "__main__":
    unittest.main()
