import math
import unittest

from training.sampling import compute_sample_weights


class ClassAwareSamplingTests(unittest.TestCase):
    def test_square_root_sampling_moderates_imbalance(self):
        rows = (
            [{"label": "Source"}] * 100
            + [{"label": "Reports"}] * 4
        )
        weights = compute_sample_weights(rows, power=0.5)
        source_mass = sum(weights[:100])
        report_mass = sum(weights[100:])
        self.assertAlmostEqual(source_mass / report_mass, 5.0)

    def test_power_zero_is_uniform_per_row(self):
        rows = [{"label": "Source"}, {"label": "Reports"}, {"label": "Source"}]
        self.assertEqual(compute_sample_weights(rows, power=0), [1.0, 1.0, 1.0])

    def test_power_one_equalizes_expected_class_mass(self):
        rows = [{"label": "Source"}] * 20 + [{"label": "Reports"}] * 2
        weights = compute_sample_weights(rows, power=1)
        self.assertAlmostEqual(sum(weights[:20]), sum(weights[20:]))

    def test_rejects_invalid_power_and_rows(self):
        with self.assertRaises(ValueError):
            compute_sample_weights([{"label": "Source"}], power=math.nan)
        with self.assertRaises(ValueError):
            compute_sample_weights([], power=0.5)
        with self.assertRaises(ValueError):
            compute_sample_weights([{"label": ""}], power=0.5)


if __name__ == "__main__":
    unittest.main()
