import tempfile
import unittest
from pathlib import Path

import torch

from lionai_scratch import LionAIConfig, build_model
from training.predict_scratch import predict


class ScratchInferenceTests(unittest.TestCase):
    def test_prediction_is_metadata_only_and_never_generates_a_path(self):
        config = LionAIConfig(d_model=32, n_heads=4, n_layers=1, feedforward=64,
                              max_length=128, num_classes=13, dropout=0.0)
        model = build_model(config)
        labels = ["Images", "Documents", "Data", "Audio", "Video", "Archives",
                  "Source", "Config", "Installers", "Fonts", "Databases", "Reports", "Other/Review"]
        checkpoint = {
            "state_dict": model.state_dict(),
            "config": config.__dict__,
            "labels": labels,
            "tokenizer": "utf8-byte-v1",
            "initialized_from": "random",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(checkpoint, path)
            result = predict("مدارک مدرسه", "notes.pdf", path, threshold=0.0)

        self.assertEqual(result["original_name"], "notes.pdf")
        self.assertIn(result["predicted_category"], labels)
        self.assertNotIn("suggested_path", result)
        self.assertTrue(result["safety"]["metadata_only"])
        self.assertFalse(result["safety"]["filesystem_modified"])
        self.assertFalse(result["safety"]["path_generation_authority"])

    def test_invalid_filename_path_is_rejected(self):
        with self.assertRaises(ValueError):
            predict("school", "../notes.pdf", "missing.pt")


if __name__ == "__main__":
    unittest.main()
