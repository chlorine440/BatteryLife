"""Portable independent arithmetic on cached arrays/JSON; no NN or sklearn.

math.fsum calculations are separate from both production NumPy metric helpers
and the remote auditor. Scientific CSV files are read, never rewritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (2021, 42, 2024)
KEYS = ("mae_cycles", "rmse_cycles", "mape", "acc15", "acc10")


def read(path):
    return json.loads(Path(path).read_text())


def calculate(p, y):
    p, y = list(map(float, p)), list(map(float, y))
    if len(p) != len(y) or not p or not all(math.isfinite(x) for x in p + y) or min(y) <= 0:
        raise ValueError("matching finite arrays and positive truth required")
    absolute = [abs(a - b) for a, b in zip(p, y)]
    relative = [a / b for a, b in zip(absolute, y)]
    n = len(p)
    return {"mae_cycles": math.fsum(absolute) / n,
            "rmse_cycles": math.sqrt(math.fsum(x * x for x in absolute) / n),
            "mape": math.fsum(relative) / n,
            "acc15": sum(x <= .15 for x in relative) / n * 100,
            "acc10": sum(x <= .1 for x in relative) / n * 100}


def summary(values):
    if len(values) != 3:
        raise ValueError("exactly three seeds required")
    result = {}
    for key in KEYS:
        mean = math.fsum(v[key] for v in values) / 3
        result[key] = {"mean": mean, "std_population": math.sqrt(
            math.fsum((v[key] - mean) ** 2 for v in values) / 3)}
    return result


def differences(actual, expected, tolerance=1e-9):
    values = []
    for key, value in expected.items():
        if isinstance(value, dict):
            values.extend(differences(actual[key], value, tolerance))
        else:
            delta = abs(float(actual[key]) - float(value))
            if not math.isfinite(float(value)) or delta > tolerance:
                raise ValueError(f"independent arithmetic mismatch: {key} / {delta}")
            values.append(delta)
    return values


def csv_check(path, expected, identity):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows, headers = list(reader), reader.fieldnames
    key = lambda row: tuple(str(row[k]) for k in identity)
    lookup = {key(row): row for row in expected}
    if len(lookup) != len(expected) or len(rows) != len(expected) or len(set(map(key, rows))) != len(rows):
        raise ValueError("CSV row count/uniqueness differs")
    if set(headers) != {k for row in expected for k in row}:
        raise ValueError("CSV header differs")
    for row in rows:
        ref = lookup[key(row)]
        for field, value in row.items():
            original = ref.get(field)
            if original is None:
                ok = value == ""
            elif isinstance(original, (float, int)) and not isinstance(original, bool):
                ok = math.isclose(float(value), original, abs_tol=1e-9, rel_tol=0)
            else:
                ok = value == str(original)
            if not ok:
                raise ValueError(f"CSV/JSON disagreement: {key(row)} / {field}")
    return {"passed": True, "rows": len(rows), "columns": len(headers)}


def audit(suite):
    manifest, report = read(suite / "manifest.json"), read(suite / "report.json")
    baseline = read(ROOT / "experiments/mixed95/benchmark_baseline.json")
    remote = read(suite / "final_audit_20261013_v1/independent_final_numeric_audit.json")
    test = [r for r in baseline["cells"] if r["split"] == "test" and r["life"] is not None]
    if len(test) != 96 or len(report["summary"]) != 13 or len(report["predictions"]) != 72:
        raise ValueError("final population/scope changed")
    errors, source, targets, remaining = [], {}, {}, {}
    source_arrays = 0
    for model, entry in manifest["models"].items():
        seeds = {}
        for seed in SEEDS:
            record = entry["seeds"][str(seed)]
            if record["status"] != "complete":
                continue
            path = (suite / "audit_inputs" / f"source_predictions_seed_{seed}_test.npz" if model == "CPTransformer"
                    else suite / "runs" / model / f"seed_{seed}" / "source_test_predictions.npz")
            with np.load(path, allow_pickle=False) as a:
                p, y, window = a["predicted"].tolist(), a["truth"].tolist(), a["observation_cycles"].tolist()
            if len(p) != 9600 or window != list(range(1, 101)) * 96:
                raise ValueError("source windows/order differs")
            if any(abs(y[i] - test[i // 100]["life"]) > .001 for i in range(9600)):
                raise ValueError("truth differs from source labels")
            keep = [i for i in range(9600) if test[i // 100]["cell"] != baseline["warning_cell"]]
            hundred = [i for i, v in enumerate(window) if v == 100]
            measured = {"all_windows": calculate(p, y),
                "cycle100_only": calculate([p[i] for i in hundred], [y[i] for i in hundred]),
                "sensitivity_without_flagged_cell": calculate([p[i] for i in keep], [y[i] for i in keep])}
            errors.extend(differences(measured, remote["models"][model]["seeds"][str(seed)]))
            seeds[str(seed)] = measured
            source_arrays += 1
        if len(seeds) == 3:
            source[model] = {protocol: summary([seeds[str(s)][protocol] for s in SEEDS])
                             for protocol in seeds["2021"]}
            for protocol, values in source[model].items():
                errors.extend(differences(values, remote["models"][model][protocol]))
        rows = entry["target"].get("rows", [])
        if rows:
            if len(rows) != 6 or {(int(r["seed"]), r["cell"]) for r in rows} != {
                    (s, c) for s in SEEDS for c in ("C1", "C2")}:
                raise ValueError("target identities/count differ")
            seed_metrics, seed_remaining = [], []
            for seed in SEEDS:
                selected = [r for r in rows if int(r["seed"]) == seed]
                p = [r["predicted_life_cycle"] for r in selected]
                y = [{"C1": 379, "C2": 419}[r["cell"]] for r in selected]
                for row, truth in zip(selected, y):
                    errors.extend(differences({"absolute_error_cycles": abs(row["predicted_life_cycle"] - truth),
                        "ape_percent": 100 * abs(row["predicted_life_cycle"] - truth) / truth},
                        {k: row[k] for k in ("absolute_error_cycles", "ape_percent")}))
                seed_metrics.append(calculate(p, y))
                seed_remaining.append(calculate([x - 100 for x in p], [x - 100 for x in y]))
            targets[model] = summary(seed_metrics)
            remaining[model] = summary(seed_remaining)
            errors.extend(differences(targets[model], report["models"][model]["target_metrics"]))
    if source_arrays != 36 or len(source) != 12 or len(targets) != 12:
        raise ValueError("full final model/seed count differs")
    keys = [(r["model"], int(r["seed"]), r["cell"]) for r in report["predictions"]]
    if len(set(keys)) != 72:
        raise ValueError("duplicate official target predictions")
    return {"passed": True, "no_model_inference": True, "algorithm": "stdlib math.fsum; no production metric helpers",
        "source_prediction_arrays": source_arrays, "formal_target_rows": 72, "max_arithmetic_delta": max(errors),
        "source": source, "target_total_life": targets, "target_remaining_life_after_physical100": remaining,
        "source_sensitivity_includes_cp_acc10": True,
        "physical_invalid_predictions": [r for r in report["predictions"] if r["predicted_life_cycle"] <= 100],
        "negative_predictions": [r for r in report["predictions"] if r["predicted_life_cycle"] < 0],
        "csv": {"comparison.csv": csv_check(suite / "comparison.csv", report["summary"], ("model",)),
                "comac_predictions.csv": csv_check(suite / "comac_predictions.csv", report["predictions"], ("model", "seed", "cell"))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    suite = args.suite.resolve()
    result = audit(suite)
    path = suite / "final_audit_20261013_v1/local_independent_math_audit.json"
    with path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("passed", "source_prediction_arrays", "formal_target_rows", "max_arithmetic_delta", "csv")}))


if __name__ == "__main__":
    main()
