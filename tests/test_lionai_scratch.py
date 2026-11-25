import unittest

from lionai_scratch import LionAIConfig, LionAITokenizer, SEP_ID, VOCAB_SIZE, build_model


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

    def test_v2_defaults_use_compact_regularized_architecture(self):
        config = LionAIConfig()
        self.assertEqual(config.d_model, 128)
        self.assertEqual(config.n_heads, 4)
        self.assertEqual(config.n_layers, 2)
        self.assertEqual(config.feedforward, 384)
        self.assertEqual(config.max_length, 192)
        self.assertEqual(config.dropout, 0.25)

    def test_model_is_randomly_initialized_and_has_expected_output_shape(self):
        import torch
        config = LionAIConfig(d_model=32, n_heads=4, n_layers=1, feedforward=64,
                              max_length=32, num_classes=13, dropout=0.0)
        model = build_model(config)
        ids, mask = LionAITokenizer().batch([
            {"topic": "school", "filename": "notes.pdf", "extension": ".pdf"},
            {"topic": "music", "filename": "track.mp3", "extension": ".mp3"},
        ], max_length=32)
        logits = model(ids, mask)
        self.assertEqual(tuple(logits.shape), (2, 13))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_model_rejects_sequences_longer_than_config(self):
        import torch
        config = LionAIConfig(d_model=32, n_heads=4, n_layers=1, feedforward=64,
                              max_length=4, num_classes=13, dropout=0.0)
        model = build_model(config)
        with self.assertRaises(ValueError):
            model(torch.ones((1, 5), dtype=torch.long), torch.ones((1, 5), dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
