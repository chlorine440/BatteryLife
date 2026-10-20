import tempfile
import unittest
import json
import csv
from pathlib import Path

import numpy as np

from experiments.mixed95.audit_benchmark_results import (
    calculate, compare, source_arrays, summarize, audit, digest, ROOT, MODELS, SEEDS, csv_matches_json,
    failure_reasons,
)


class IndependentNumericalAuditTests(unittest.TestCase):
    def test_failed_tasks_require_explicit_reasons_without_scores(self):
        entry = {"seeds": {"2021": {"status": "failed", "reason": "metric mismatch"},
                           "42": {"status": "complete"}, "2024": {"status": "pending"}},
                 "target": {"status": "pending"}}
        self.assertEqual(failure_reasons(entry), "metric mismatch")
        entry["target"] = {"status": "blocked_incomplete", "reason": "three seeds required"}
        self.assertEqual(failure_reasons(entry), "metric mismatch; three seeds required")
        for status in ("failed", "blocked_smoke", "blocked_incomplete"):
            for reason in (None, "", "  ", 0):
                bad = dict(entry, target={"status": status, "reason": reason})
                with self.assertRaisesRegex(ValueError, "missing its explicit reason"):
                    failure_reasons(bad)

    def test_final_report_cannot_omit_failed_trial_reason(self):
        baseline = ROOT/"experiments/mixed95/benchmark_baseline.json"
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp)
            manifest = {"identity": {"baseline_sha256": digest(baseline), "code_sha256": {}},
                "models": {m: {"seeds": {str(s): {"status": "failed", "reason": "recorded failure"}
                                         for s in SEEDS},
                               "target": {"status": "blocked_incomplete", "reason": "incomplete source"}}
                           for m in MODELS}}
            published = {"models": {m: {} for m in MODELS}, "summary": [
                {"model": m, "source_status": json.dumps({"failed": 3}),
                 "target_status": "blocked_incomplete", "failure_reason": ""} for m in MODELS]}
            (suite/"manifest.json").write_text(json.dumps(manifest))
            (suite/"report.json").write_text(json.dumps(published))
            with self.assertRaisesRegex(ValueError, "failure reason disagrees"):
                audit(suite, suite/"cp", live=False)

    def test_csv_serialization_distinguishes_zero_missing_and_duplicate_ids(self):
        expected = [{"model": "a", "mae": 0.0, "time": None},
                    {"model": "b", "mae": 12.5, "time": 10}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"comparison.csv"
            def write(rows):
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["model", "mae", "time"])
                    writer.writeheader(); writer.writerows(rows)
            write(expected)
            self.assertEqual(csv_matches_json(path, expected, ("model",))["rows"], 2)
            for rows in ([dict(expected[0], time=0), expected[1]],
                         [dict(expected[0], mae=""), expected[1]],
                         [expected[0], expected[0]]):
                write(rows)
                with self.assertRaises(ValueError):
                    csv_matches_json(path, expected, ("model",))

    def test_metrics_units_boundary_and_rmse(self):
        values = calculate([85, 130], [100, 100])
        self.assertEqual(values["mae_cycles"], 22.5)
        self.assertAlmostEqual(values["rmse_cycles"], np.sqrt((225+900)/2))
        self.assertAlmostEqual(values["mape"], .225)
        self.assertEqual(values["acc15"], 50)
        self.assertEqual(values["acc10"], 0)

    def test_invalid_arrays_and_nonfinite_metric_rejected(self):
        for p, y in (([], []), ([1], [0]), ([float("nan")], [1]), ([1, 2], [1])):
            with self.assertRaises(ValueError):
                calculate(p, y)
        for x in (float("nan"), 50.01):
            with self.assertRaises(ValueError):
                compare({"acc15": 50}, {"acc15": x})

    def test_only_three_seeds_population_std(self):
        with self.assertRaises(ValueError):
            summarize([{"mae_cycles": 1}])
        summary = summarize([{"mae_cycles": x} for x in (1, 2, 3)])
        self.assertEqual(summary["mae_cycles"]["mean"], 2)
        self.assertAlmostEqual(summary["mae_cycles"]["std_population"], np.sqrt(2/3))

    def test_final_audit_rejects_running_or_pending_source(self):
        baseline = ROOT/"experiments/mixed95/benchmark_baseline.json"
        for status in ("pending", "running"):
            with tempfile.TemporaryDirectory() as tmp:
                suite = Path(tmp)
                manifest = {"identity": {"baseline_sha256": digest(baseline), "code_sha256": {}},
                    "models": {m: {"seeds": {str(s): {"status": status} for s in SEEDS}}
                               for m in MODELS}}
                (suite/"manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "all 39 source tasks terminal"):
                    audit(suite, suite/"cp", live=False)

    def test_windows_truth_order_and_flagged_cell(self):
        cells = [{"cell": str(i), "life": 200+i} for i in range(96)]
        truth = np.repeat([r["life"] for r in cells], 100).astype("float32")
        windows = np.tile(np.arange(1, 101), 96)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"source.npz"
            np.savez(path, predicted=truth, truth=truth, observation_cycles=windows)
            values = source_arrays(path, cells, "0")
            self.assertEqual(values["all_windows"]["acc15"], 100)
            self.assertEqual(values["sensitivity_without_flagged_cell"]["mae_cycles"], 0)
            windows[0] = 101
            np.savez(path, predicted=truth, truth=truth, observation_cycles=windows)
            with self.assertRaises(ValueError):
                source_arrays(path, cells, "0")


if __name__ == "__main__":
    unittest.main()
