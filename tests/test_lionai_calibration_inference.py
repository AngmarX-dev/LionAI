import tempfile
import unittest
from pathlib import Path

import torch

from lionai_scratch import LionAIConfig, build_model
from training.calibration import write_calibration
from training.predict_scratch import predict


class CalibratedInferenceTests(unittest.TestCase):
    def test_temperature_and_margin_are_reported(self):
        config = LionAIConfig(d_model=32, n_heads=4, n_layers=1, feedforward=64,
                              max_length=128, num_classes=13, dropout=0.0)
        model = build_model(config)
        labels = ["Images", "Documents", "Data", "Audio", "Video", "Archives",
                  "Source", "Config", "Installers", "Fonts", "Databases", "Reports", "Other/Review"]
        checkpoint = {"state_dict": model.state_dict(), "config": config.__dict__,
                      "labels": labels, "tokenizer": "utf8-byte-v1", "initialized_from": "random"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path, calibration_path = root / "model.pt", root / "calibration.json"
            torch.save(checkpoint, checkpoint_path)
            write_calibration(calibration_path, 1.5, labels, 20)
            result = predict("school", "notes.pdf", checkpoint_path, threshold=0.0,
                             calibration_path=calibration_path, margin_threshold=0.0)
        self.assertTrue(result["calibrated"])
        self.assertEqual(result["temperature"], 1.5)
        self.assertGreaterEqual(result["confidence_margin"], 0.0)


if __name__ == "__main__":
    unittest.main()
