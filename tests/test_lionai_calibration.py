import tempfile
import unittest
from pathlib import Path

import torch

from training.calibration import fit_temperature, load_temperature, write_calibration


class TemperatureCalibrationTests(unittest.TestCase):
    def test_fits_finite_positive_temperature(self):
        logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [2.0, 0.2], [0.1, 2.0]])
        targets = torch.tensor([0, 1, 0, 1])
        temperature = fit_temperature(logits, targets)
        self.assertTrue(0.05 <= temperature <= 20.0)

    def test_round_trip_and_label_binding(self):
        labels = ["Images", "Documents"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            write_calibration(path, 1.4, labels, 12)
            self.assertEqual(load_temperature(path, labels), 1.4)
            with self.assertRaisesRegex(ValueError, "labels do not match"):
                load_temperature(path, ["Documents", "Images"])

    def test_rejects_invalid_temperature(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            with self.assertRaises(ValueError):
                write_calibration(path, float("nan"), ["Images"], 5)

    def test_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            fit_temperature(torch.zeros((3, 2)), torch.zeros((2,), dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
