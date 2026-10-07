"""eVTOL RPT/mission separation and leakage checks."""

import unittest
from pathlib import Path

import pandas as pd

from experiments.comac95.evtol import (REQUIRED_COLUMNS, _stratified_splits,
                                       classify_runs, extract_cell)
from experiments.comac95.run import _normalized_curve


class EvtolTests(unittest.TestCase):
    def test_lifetime_cumulative_capacity_is_rebased_per_leg(self):
        charge = [(3.5, 1.0, 500.0), (4.2, 1.0, 501.0)]
        discharge = [(4.1, -1.0, 600.0), (2.5, -1.0, 601.0)]
        curve = _normalized_curve(charge, discharge, nominal_ah=2.0)
        self.assertAlmostEqual(float(curve[2, 0]), 0.0)
        self.assertAlmostEqual(float(curve[2, 150]), 0.0)
        self.assertAlmostEqual(float(curve[2, 149]), 0.5)
        self.assertAlmostEqual(float(curve[2, 299]), 0.5)

    def test_repeating_local_cycle_numbers_do_not_merge_runs(self):
        rows = []
        for local, step, count, current, qd in (
                (0, 3, 1000, -600, 600), (0, 5, 1000, -600, 1800),
                (0, 7, 100, -600, 3000), (0, 8, 30, 0, 3000),
                (1, 3, 110, -3000, 1000),
                (0, 3, 1000, -600, 600), (0, 5, 1000, -600, 1800),
                (0, 7, 100, -600, 2800), (0, 8, 30, 0, 2800)):
            rows.extend((len(rows) + i, 3.7, current, 1500, qd, local, step)
                        for i in range(count))
        frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
        _, summary = classify_runs(frame)
        self.assertEqual(len(summary), 3)
        self.assertEqual(summary.rpt_protocol.tolist(), [True, False, True])
        self.assertEqual(summary.rpt_complete.tolist(), [True, False, True])

    def test_stratified_splits_are_cell_disjoint(self):
        labels = {f"VAH{i:02d}": life for i, life in enumerate(
            [150] * 3 + [200] * 10 + [250] * 9, start=1)}
        splits = _stratified_splits(labels)
        self.assertEqual([len(splits[key]) for key in ("train", "val", "test")],
                         [12, 5, 5])
        sets = [set(splits[key]) for key in ("train", "val", "test")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(set.union(*sets), set(labels))
        self.assertFalse({"C1", "C2"} & set(labels))

    def test_real_first_100_window_and_rpt_if_available(self):
        folder = Path("/Users/curtischan/Datasets/comac/raw/eVTOL")
        if not folder.is_dir():
            self.skipTest("local eVTOL CSV files not installed")
        for cell, expected in (("VAH01", 250), ("VAH09", 150)):
            with self.subTest(cell=cell):
                life, features, mask, rpt, audit, _ = extract_cell(folder / f"{cell}.csv")
                self.assertEqual(life, expected)
                self.assertEqual(features.shape, (100, 3, 300))
                self.assertGreaterEqual(mask.sum(), 98)
                self.assertEqual(mask[49], 0)  # age-50 RPT is masked
                self.assertGreater(audit["complete_rpt_runs"], 2)
                self.assertTrue(all(row["label_source"] == "RPT_C_over_5_full_discharge"
                                    for row in rpt))


if __name__ == "__main__":
    unittest.main()
