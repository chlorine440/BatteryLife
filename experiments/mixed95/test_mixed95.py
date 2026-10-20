"""Contracts for the paper-configured source labels and held-out target input."""

import math
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from data_provider.data_split_recorder import split_recorder
from experiments.mixed95.evaluate import build_target_input, evaluate, prepare_comac
from experiments.mixed95.comac_input import (
    PREPROCESSING_VERSION, TARGET_DIRECTORY, cycle_frame, load_target_input,
    original_loader_features, read_early_cycle_records,
)
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
        trace = [
            (0., 4., 183., 100., 200.),
            (1., 4.03, 183., 100.5, 200.),
            (2., 4.1, 183., 101., 200.),
            (3., 4.12, 36.8, 101.5, 200.),
            (4., 4.08, 0., 101.5, 200.),
            (5., 4., -183., 101.5, 200.),
            (6., 3., -183., 101.5, 201.),
            (7., 3.1, 0., 101.5, 201.),
        ]
        return {cycle: trace for cycle in range(2, 101)}

    def test_100_physical_cycles_rpt_gap_and_nominal_feature_scale(self):
        with patch("experiments.mixed95.evaluate.read_early_cycle_records", return_value=self.curves()):
            features, mask = build_target_input(None, "C1")
        self.assertEqual(features.shape, (100, 3, 300))
        self.assertEqual(mask.shape, (100,))
        self.assertEqual(mask[0], 0)
        self.assertTrue((mask[1:] == 1).all())
        self.assertAlmostEqual(float(features[1, 1, 0]), 1.0)
        self.assertAlmostEqual(float(features[1, 2, 149]), 1.0 / 183.0, places=6)
        # Original convention: the last positive-current point starts the
        # discharge segment. Do not 'fix' the source method in an adapter.
        self.assertAlmostEqual(float(features[1, 1, 150]), 36.8 / 183., places=6)
        self.assertAlmostEqual(float(features[1, 0, 149]), 4.1 / 4.12, places=6)

    def test_future_cycle_rejected(self):
        curves = self.curves()
        curves[101] = curves[100]
        with patch("experiments.mixed95.evaluate.read_early_cycle_records", return_value=curves):
            with self.assertRaises(ValueError):
                build_target_input(None, "C2")

    def test_exact_c_rate_cutoffs_follow_original_inclusive_then_strict_filters(self):
        trace = self.curves()[2].copy()
        trace[3] = (3., 4.12, 1.83, 101.5, 200.)
        trace[5] = (5., 4., -1.83, 101.5, 200.)
        features, _ = original_loader_features({n: trace for n in range(2, 101)})
        # >= .01 chooses the charging endpoint; > .01 subsequently removes
        # the exact-threshold points from the discharge segment.
        self.assertEqual(float(features[1, 1, 150]), -1.)
        self.assertAlmostEqual(float(features[1, 2, 150]), 1. / 183., places=6)

    def test_missing_cycle_rejected_without_compressing_physical_indices(self):
        curves = self.curves()
        del curves[50]
        with self.assertRaises(ValueError):
            original_loader_features(curves)

    def test_rest_inside_charge_and_exact_original_feature_equivalence(self):
        from data_provider.data_loader import Dataset_original
        import pandas as pd

        trace = self.curves()[2].copy()
        trace.insert(1, (.5, 3.8, 0., 100., 200.))
        curves = {n: trace for n in range(2, 101)}
        actual, _ = original_loader_features(curves)
        frames = []
        for n, records in curves.items():
            a = np.asarray(records)
            frames.append(pd.DataFrame({"cycle_number": n,
                "voltage_in_V": a[:, 1], "current_in_A": a[:, 2],
                "charge_capacity_in_Ah": a[:, 3] - 100.,
                "discharge_capacity_in_Ah": a[:, 4] - 200.,
                "time_in_s": a[:, 0] * 86400}))
        loader = Dataset_original.__new__(Dataset_original)
        loader.charge_discharge_len = 300
        loader.ZN_coin_charge_first_file_names = []
        expected = loader.get_charge_discharge_curves("COMAC_C1.pkl",
            pd.concat(frames), 100, 183.).astype(np.float32)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue((actual[1, 1, :150] < .01).any())

    def test_lifetime_capacity_conversion_keeps_all_charge_steps(self):
        frame = cycle_frame(2, self.curves()[2])
        np.testing.assert_array_equal(frame.charge_capacity_in_Ah,
                                     [0., .5, 1., 1.5, 1.5, 1.5, 1.5, 1.5])
        np.testing.assert_array_equal(frame.discharge_capacity_in_Ah,
                                     [0., 0., 0., 0., 0., 0., 1., 1.])
        self.assertEqual(len(frame), 8)  # Rest points retained for the loader.

    def test_record_reader_merges_continuation_and_excludes_rpt_and_future(self):
        trace = self.curves()[2]
        header = dict(enumerate(["循环序号", "绝对时间", "电压(V)",
            "电流(A)", "累计充电容量(Ah)", "累计放电容量(Ah)"]))
        def row(cycle, p):
            return dict(enumerate([str(cycle), *map(str, p)]))
        class FakeBook:
            sheets = {"详细数据1": None}
            def __init__(self, path):
                self.continuation = path.name.endswith("-2.xlsx")
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def rows(self, _):
                yield header
                yield row(1, trace[0])
                for p in (trace[3:] if self.continuation else trace[:5]):
                    yield row(2, p)
                yield row(101, trace[-1])
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "Cell01"
            folder.mkdir()
            for part in (1, 2):
                (folder / f"C1-100-{part}.xlsx").touch()
            with patch("experiments.mixed95.comac_input.XlsxRows", FakeBook):
                records = read_early_cycle_records(Path(temp), "C1")
        self.assertEqual(records, {2: trace})

    @staticmethod
    def save_input(path, **overrides):
        values = dict(features=np.zeros((100, 3, 300), dtype=np.float32),
                      mask=np.r_[0., np.ones(99)].astype(np.float32),
                      physical_cycles=np.arange(1, 101),
                      preprocessing_version=np.asarray(PREPROCESSING_VERSION))
        values.update(overrides)
        np.savez_compressed(path, **values)

    def test_versioned_input_rejects_legacy_future_nan_and_hidden_rpt(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.npz"
            self.save_input(path)
            features, _ = load_target_input(path)
            for overrides in (
                {"preprocessing_version": np.asarray("legacy")},
                {"physical_cycles": np.arange(2, 102)},
                {"mask": np.ones(100, dtype=np.float32)},
                {"features": features + np.nan},
                {"features": features + 1},
            ):
                with self.subTest(overrides=list(overrides)):
                    self.save_input(path, **overrides)
                    with self.assertRaises(ValueError):
                        load_target_input(path)
            np.savez_compressed(path, features=features)
            with self.assertRaises(ValueError):
                load_target_input(path)

    def test_preparation_refuses_to_overwrite_corrected_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / TARGET_DIRECTORY).mkdir()
            with self.assertRaises(FileExistsError):
                prepare_comac(None, Path(temp))

    def test_corrected_evaluation_preserves_legacy_result_and_guards_second_run(self):
        import torch
        class FakeModel:
            def __call__(self, x, mask):
                return torch.zeros((1, 1))
        class FakeScaler:
            def inverse_transform(self, x):
                return np.asarray(x) + 400.
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            run = output / "runs" / "fake_source_run"
            run.mkdir(parents=True)
            legacy = run / "held_out_comac_evaluation_95.json"
            legacy.write_text("preserve legacy result")
            (run / "manifest.json").write_text(json.dumps({"smoke_only": False,
                "seeds": {str(s): {"status": "complete", "checkpoint": "fake",
                    "source_test": {}} for s in (2021, 42, 2024)}}))
            target = output / TARGET_DIRECTORY
            target.mkdir()
            (target / "COMAC_proxy_labels_95.json").write_text(json.dumps({"C1": 379, "C2": 419}))
            for cell in ("C1", "C2"):
                self.save_input(target / f"{cell}_input.npz")
            with patch("experiments.mixed95.evaluate._load_paper_model",
                       return_value=(FakeModel(), FakeScaler())) as load:
                result = evaluate(output, run_id="fake_source_run")
                self.assertEqual(load.call_count, 3)
                with self.assertRaises(FileExistsError):
                    evaluate(output, run_id="fake_source_run")
            self.assertEqual(legacy.read_text(), "preserve legacy result")
            self.assertEqual(len(result["comac"]), 6)
            self.assertEqual(result["target_preprocessing_version"], PREPROCESSING_VERSION)
            self.assertTrue(result["prior_evaluation_preserved"])


if __name__ == "__main__":
    unittest.main()
