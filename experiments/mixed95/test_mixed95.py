"""Contracts for the paper-configured source labels and held-out target input."""

import math
import unittest
from unittest.mock import patch

from data_provider.data_split_recorder import split_recorder
from experiments.mixed95.evaluate import build_target_input
from experiments.mixed95.prepare import (
    capacity_denominator, label_file_and_key, life_from_soh, source_folder,
)
from experiments.mixed95.train import _source_metrics


class SourceLabelTests(unittest.TestCase):
    def test_first_observed_crossing_and_100_cycle_boundary(self):
        self.assertEqual(life_from_soh([1.0] * 100 + [0.95, 0.94]), ("observed", 101))
        self.assertEqual(life_from_soh([1.0] * 99 + [0.95, 0.94]),
                         ("life_at_or_before_100", None))
        self.assertEqual(life_from_soh([1.0] * 100 + [0.950001, 0.95]),
                         ("observed", 102))

    def test_paper_extrapolation_band_and_censoring(self):
        near = [1.0 - 0.04 * i / 199 for i in range(200)]
        status, life = life_from_soh(near)
        self.assertEqual(status, "extrapolated")
        self.assertGreater(life, 200)
        self.assertEqual(life_from_soh([0.98] * 200), ("not_reached", None))
        boundary = [1.0 - 0.025 * i / 199 for i in range(200)]
        self.assertEqual(life_from_soh(boundary)[0], "extrapolated")
        self.assertEqual(life_from_soh([math.nan] * 200), ("invalid_terminal_soh", None))

    def test_dataset_denominator_and_label_mapping(self):
        rwth = capacity_denominator({"nominal_capacity_in_Ah": 9,
                                     "SOC_interval": [0.2, 0.8]}, "RWTH_016.pkl")
        snl = capacity_denominator({"nominal_capacity_in_Ah": 9,
                                    "SOC_interval": [0.2, 0.8]}, "SNL_18650_NCA_25C_20-80_a.pkl")
        self.assertEqual(rwth[0], 1.85)
        self.assertEqual(snl[0], 3.2)
        self.assertAlmostEqual(rwth[1], 0.6)
        self.assertAlmostEqual(snl[1], 0.6)
        self.assertEqual(source_folder("UL-PUR_example.pkl"), "UL_PUR")
        self.assertEqual(label_file_and_key("Tongji1_CY35-05_1--2.pkl"),
                         ("Tongji_labels.json", "Tongji1_CY35-05_1-#2.pkl"))

    def test_mixed_split_is_disjoint(self):
        splits = [set(split_recorder.MIX_large_train_files),
                  set(split_recorder.MIX_large_val_files),
                  set(split_recorder.MIX_large_test_files)]
        self.assertTrue(all(splits))
        self.assertFalse(splits[0] & splits[1])
        self.assertFalse(splits[0] & splits[2])
        self.assertFalse(splits[1] & splits[2])
        self.assertFalse({"C1", "C2"} & set.union(*splits))

    def test_loader_skips_missing_95_label(self):
        try:
            from data_provider.data_loader import Dataset_original
        except ModuleNotFoundError as exc:
            self.skipTest(f"original loader dependency not installed: {exc}")
        loader = Dataset_original.__new__(Dataset_original)
        loader.read_cell_data_according_to_prefix = lambda _: (None, None)
        self.assertEqual(loader.read_cell_df("missing.pkl"), (None,) * 6)

    def test_original_training_log_parser(self):
        line = ("Best model performance: Test MAE: 10.0 | Test RMSE: 12.0 | "
                "Test MAPE: 0.1234 | Test 15%-accuracy: 70.0 | "
                "Test 10%-accuracy: 60.0 | Val MAE: 8.0")
        self.assertEqual(_source_metrics(line)["mape"], 0.1234)


class TargetInputTests(unittest.TestCase):
    @staticmethod
    def curves():
        trace = {"charge": [(4.0, 183.0, 0.0), (4.1, 183.0, 1.0)],
                 "discharge": [(4.0, -183.0, 0.0), (3.0, -183.0, 1.0)]}
        return {cycle: trace for cycle in range(2, 101)}

    def test_100_physical_cycles_rpt_gap_and_nominal_feature_scale(self):
        with patch("experiments.mixed95.evaluate.read_early_curves", return_value=self.curves()):
            features, mask = build_target_input(None, "C1")
        self.assertEqual(features.shape, (100, 3, 300))
        self.assertEqual(mask.shape, (100,))
        self.assertEqual(mask[0], 0)
        self.assertTrue((mask[1:] == 1).all())
        self.assertAlmostEqual(float(features[1, 1, 0]), 1.0)
        self.assertAlmostEqual(float(features[1, 2, 149]), 1.0 / 183.0, places=6)

    def test_future_cycle_rejected(self):
        curves = self.curves()
        curves[101] = curves[100]
        with patch("experiments.mixed95.evaluate.read_early_curves", return_value=curves):
            with self.assertRaises(ValueError):
                build_target_input(None, "C2")


if __name__ == "__main__":
    unittest.main()
