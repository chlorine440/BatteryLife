"""Tests for the 95% experiment's source-independent label contract."""

import unittest
from pathlib import Path

from experiments.comac95.data import (
    NOMINAL_AH, SOC_SPAN, deduplicate_trace, first_crossing, global_cycle, proxy_soh,
    read_capacity_history, read_early_curves,
)


class LabelTests(unittest.TestCase):
    def test_continuation_file_duplicate_records(self):
        first = (1.0, 3.9, -183.0, 0.1)
        second = (2.0, 3.8, -183.0, 0.2)
        self.assertEqual(deduplicate_trace([second, first, second]),
                         [(3.9, -183.0, 0.1), (3.8, -183.0, 0.2)])

    def test_xjtu_split_is_cell_disjoint_and_comac_is_held_out(self):
        from data_provider.data_split_recorder import split_recorder

        splits = (set(split_recorder.XJTU_train_files),
                  set(split_recorder.XJTU_val_files),
                  set(split_recorder.XJTU_test_files))
        self.assertTrue(all(splits))
        self.assertFalse(splits[0] & splits[1])
        self.assertFalse(splits[0] & splits[2])
        self.assertFalse(splits[1] & splits[2])
        self.assertFalse({"C1", "C2"} & set.union(*splits))

    def test_physical_cycle_mapping(self):
        self.assertEqual(global_cycle("C1", 100, 1), 1)
        self.assertEqual(global_cycle("C1", 200, 1), 101)
        self.assertEqual(global_cycle("C2", 500, 100), 500)
        with self.assertRaises(ValueError):
            global_cycle("C3", 200, 1)

    def test_proxy_normalization(self):
        self.assertAlmostEqual(proxy_soh(NOMINAL_AH * SOC_SPAN), 1.0)
        self.assertAlmostEqual(proxy_soh(NOMINAL_AH * SOC_SPAN * .95), .95)
        with self.assertRaises(ValueError):
            proxy_soh(0)

    def test_first_crossing_and_censoring(self):
        self.assertEqual(first_crossing([(1, 1.0), (2, .95001), (3, .95), (4, .94)]), 3)
        self.assertIsNone(first_crossing([(1, 1.0), (2, .95001)]))
        self.assertEqual(first_crossing([(2, .94), (1, .97)]), 2)

    def test_real_comac_if_available(self):
        root = Path("/Users/curtischan/Datasets/comac/raw/商飞800VSOH-tight/商飞800VSOH-tight")
        if not root.is_dir():
            self.skipTest("local COMAC Excel files not installed")
        for cell, expected in (("C1", 379), ("C2", 419)):
            with self.subTest(cell=cell):
                ordinary, rpt, crossing, issues = read_capacity_history(root, cell)
                self.assertEqual(crossing, expected)
                self.assertEqual(len(ordinary), 495)
                self.assertEqual([row["cycle"] for row in rpt], [1, 101, 201, 301, 401])
                self.assertFalse({row["cycle"] for row in ordinary} & {row["cycle"] for row in rpt})
                self.assertEqual(issues, [])

    def test_real_early_window_if_available(self):
        root = Path("/Users/curtischan/Datasets/comac/raw/商飞800VSOH-tight/商飞800VSOH-tight")
        if not root.is_dir():
            self.skipTest("local COMAC Excel files not installed")
        for cell in ("C1", "C2"):
            with self.subTest(cell=cell):
                curves = read_early_curves(root, cell)
                self.assertEqual(set(curves), set(range(2, 101)))
                self.assertTrue(all(curves[cycle]["charge"] and curves[cycle]["discharge"]
                                    for cycle in curves))
        with self.assertRaises(ValueError):
            read_early_curves(root, "C1", max_physical_cycle=101)


if __name__ == "__main__":
    unittest.main()
