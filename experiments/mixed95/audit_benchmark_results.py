"""Read-only numerical audit, independent of benchmark metric/report helpers.

This never instantiates a model or predicts COMAC again. Live mode checks only
finished source trials; final mode additionally requires terminal source/target
states and checks the published report. Printed JSON is evidence, not a queue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import csv
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("CPTransformer", "CPMLP", "CPGRU", "CPBiGRU", "CPLSTM", "CPBiLSTM",
          "MLP", "CNN", "DLinear", "PatchTST", "Autoformer", "iTransformer", "MICN")
SEEDS = (2021, 42, 2024)
RNN = {"CPGRU", "CPLSTM", "CPBiGRU", "CPBiLSTM"}
TERMINAL = {"complete", "failed", "blocked_smoke"}
CP_ARRAY_SHA = {
    2021: "f606a7c03d4b694eb48621225419bea7af34897f990dc6468ed37a4a998c0d0c",
    42: "c84620aab26d27152e03e7408eef539f543b84aeaa1317bca75305cd64ef2bbb",
    2024: "9692f0212bb1e19b9777be15b8afd82ccf5528ce0f4e6299750bcfff6f87143c",
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def calculate(p, y):
    p, y = np.asarray(p, dtype="float64"), np.asarray(y, dtype="float64")
    if p.shape != y.shape or not p.size or not np.isfinite(p).all() or not (
            np.isfinite(y).all() and (y > 0).all()):
        raise ValueError("nonempty finite predictions and positive matching truth required")
    error = p - y
    relative = abs(error) / y
    return {"mae_cycles": float(abs(error).sum() / p.size),
            "rmse_cycles": float(np.sqrt(np.dot(error.ravel(), error.ravel()) / p.size)),
            "mape": float(relative.sum() / p.size),
            "acc15": float(np.count_nonzero(relative <= .15) / p.size * 100),
            "acc10": float(np.count_nonzero(relative <= .10) / p.size * 100)}


def compare(actual, saved):
    for name, value in saved.items():
        if name not in actual:
            continue
        limit = (.002 if name in {"mae_cycles", "rmse_cycles"} else .000051)
        if not np.isfinite(value) or abs(actual[name] - value) > limit:
            raise ValueError(f"independent audit mismatch {name}: {actual[name]} != {value}")


def source_arrays(path, test_cells, warning_cell):
    with np.load(path) as a:
        p, y, w = a["predicted"], a["truth"], a["observation_cycles"]
        ids = np.repeat([r["cell"] for r in test_cells], 100)
        if p.shape != (9600,) or y.shape != (9600,) or not np.array_equal(
                w, np.tile(np.arange(1, 101), 96)):
            raise ValueError("source window count/order changed")
        if "cell" in a.files and not np.array_equal(a["cell"], ids):
            raise ValueError("source cell order changed")
        expected = np.repeat([r["life"] for r in test_cells], 100)
        if not np.allclose(y, expected, atol=.001, rtol=0):
            raise ValueError("source truth differs from frozen labels")
        keep = ids != warning_cell
        if keep.sum() != 9500 or (w == 100).sum() != 96:
            raise ValueError("sensitivity or cycle100 population changed")
        return {"all_windows": calculate(p, y),
                "cycle100_only": calculate(p[w == 100], y[w == 100]),
                "sensitivity_without_flagged_cell": calculate(p[keep], y[keep])}


def summarize(values):
    if len(values) != 3:
        raise ValueError("three complete seeds, not partial aggregation, required")
    return {k: {"mean": float(np.mean([v[k] for v in values])),
                "std_population": float(np.std([v[k] for v in values], ddof=0))}
            for k in values[0]}


def failure_reasons(entry):
    """Require explicit provenance for failed/blocked tasks, never infer scores."""
    reasons = []
    for record in [*entry["seeds"].values(), entry["target"]]:
        if record["status"] not in {"failed", "blocked_smoke", "blocked_incomplete"}:
            continue
        reason = record.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("failed/blocked task is missing its explicit reason")
        reasons.append(reason)
    return "; ".join(reasons)


def csv_matches_json(path, expected, identity):
    """Read-only serialization reconciliation, including blank versus zero.

    Scientific CSV records remain intact. Header/identity matches are exact;
    finite numeric cells must agree with their underlying JSON values.
    """
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        rows = list(reader)
    expected_headers = {k for row in expected for k in row}
    if len(headers) != len(set(headers)) or set(headers) != expected_headers:
        raise ValueError("CSV columns duplicated, missing, or unexpected")
    def key(row):
        return tuple(str(row[k]) for k in identity)
    lookup = {key(row): row for row in expected}
    keys = [key(row) for row in rows]
    if len(lookup) != len(expected) or len(keys) != len(set(keys)) or set(keys) != set(lookup):
        raise ValueError("CSV records duplicated, missing, or unexpected")
    for row in rows:
        if None in row or any(v is None for v in row.values()):
            raise ValueError("CSV row length differs from header")
        reference = lookup[key(row)]
        for name, raw in row.items():
            value = reference.get(name)
            if value is None:
                equal = raw == ""
            elif isinstance(value, bool):
                equal = raw == str(value)
            elif isinstance(value, (int, float)):
                try:
                    numeric = float(raw)
                    equal = np.isfinite(numeric) and np.isclose(numeric, value, rtol=0, atol=1e-9)
                except ValueError:
                    equal = False
            else:
                equal = raw == str(value)
            if not equal:
                raise ValueError(f"CSV/JSON mismatch {key(row)} / {name}: {raw!r} != {value!r}")
    return {"passed": True, "rows": len(rows), "columns": len(headers), "sha256": digest(path)}


def audit(suite, cp_arrays, live=False):
    import joblib
    m = read(suite / "manifest.json")
    basepath = ROOT / "experiments/mixed95/benchmark_baseline.json"
    baseline = read(basepath)
    if set(m["models"]) != set(MODELS) or m["identity"]["baseline_sha256"] != digest(basepath):
        raise ValueError("model whitelist or frozen baseline changed")
    for name, sha in m["identity"]["code_sha256"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"frozen benchmark code changed: {name}")
    eligible = {s: [r for r in baseline["cells"] if r["split"] == s and r["life"] is not None]
                for s in ("train", "val", "test")}
    if {s: len(v) for s, v in eligible.items()} != {"train": 283, "val": 89, "test": 96}:
        raise ValueError("source coverage changed")
    names = [r["cell"] for rows in eligible.values() for r in rows]
    if len(names) != len(set(names)) or any(r["life"] <= 100 for rows in eligible.values() for r in rows):
        raise ValueError("split overlap or early end-of-life")
    labels = np.asarray([r["life"] for r in eligible["train"]], dtype="float64")
    if not (np.isclose(labels.mean(), baseline["scaler"]["mean"], rtol=0, atol=1e-10)
            and np.isclose(labels.std(), baseline["scaler"]["scale"], rtol=0, atol=1e-10)):
        raise ValueError("frozen scaler differs from unique training cells")
    results, target_keys = {}, set()
    terminal = all(r["status"] in TERMINAL for e in m["models"].values() for r in e["seeds"].values())
    if not live and not terminal:
        raise ValueError("final audit requires all 39 source tasks terminal")
    published = None if live else read(suite / "report.json")
    for model in MODELS:
        e, result = m["models"][model], {"seeds": {}}
        if set(e["seeds"]) != {str(s) for s in SEEDS}:
            raise ValueError("seed set changed")
        expected_failure_reason = failure_reasons(e)
        for seed in SEEDS:
            r = e["seeds"][str(seed)]
            if r["status"] != "complete":
                continue
            cp = Path(r["checkpoint"])
            config = read(cp / "args.json")
            for k, value in dict(m["identity"]["configs"][model], seed=seed).items():
                if config.get(k) != value:
                    raise ValueError(f"saved configuration changed: {model}/{seed}/{k}")
            scaler = joblib.load(cp / "label_scaler")
            if int(scaler.n_samples_seen_) != 283 or not np.allclose(
                    [scaler.mean_[0], scaler.scale_[0]], [labels.mean(), labels.std()], rtol=0, atol=1e-10):
                raise ValueError("scaler leakage or changed statistics")
            for name, sha in r.get("checkpoint_sha256", {}).items():
                if digest(cp / name) != sha:
                    raise ValueError("checkpoint artifact changed")
            array = suite / "runs" / model / f"seed_{seed}" / "source_test_predictions.npz"
            if model == "CPTransformer":
                array = cp_arrays / f"source_predictions_seed_{seed}_test.npz"
                if digest(array) != CP_ARRAY_SHA[seed]:
                    raise ValueError("historical CP prediction array changed")
            measured = source_arrays(array, eligible["test"], baseline["warning_cell"])
            for protocol, values in measured.items():
                compare(values, r["source_verified"][protocol])
            compare(measured["all_windows"], r["source_test"])
            result["seeds"][str(seed)] = measured
        full = len(result["seeds"]) == 3
        if full:
            for protocol in next(iter(result["seeds"].values())):
                result[protocol] = summarize([v[protocol] for v in result["seeds"].values()])
        target = e["target"]
        if not live and target["status"] not in {"complete", "failed", "blocked_incomplete"}:
            raise ValueError("target phase not terminal")
        if target["status"] == "complete":
            if not full or (model != "CPTransformer" and not terminal):
                raise ValueError("target evaluated before all source tasks ended")
            rows = target["rows"]
            keys = {(r["model"], int(r["seed"]), r["cell"]) for r in rows}
            if keys != {(model, s, c) for s in SEEDS for c in ("C1", "C2")} or len(rows) != 6:
                raise ValueError("incomplete/duplicate target predictions")
            target_keys.update(keys)
            for row in rows:
                true, pred = {"C1": 379, "C2": 419}[row["cell"]], row["predicted_life_cycle"]
                if row["proxy_label_life_cycle"] != true or not np.isfinite(pred):
                    raise ValueError("incorrect proxy labels or nonfinite prediction")
                if not np.isclose(abs(pred-true), row["absolute_error_cycles"], rtol=0, atol=1e-8) or not np.isclose(
                        100*abs(pred-true)/true, row["ape_percent"], rtol=0, atol=1e-8):
                    raise ValueError("per-cell target errors inconsistent")
            result["target_metrics"] = summarize([calculate(
                [r["predicted_life_cycle"] for r in rows if int(r["seed"]) == s],
                [r["proxy_label_life_cycle"] for r in rows if int(r["seed"]) == s]) for s in SEEDS])
            if model != "CPTransformer":
                for cell in ("C1", "C2"):
                    mapping = read(suite / "target_mapping" / f"{model}_{cell}.json")
                    expected_ids = list(range(2, 101)) + [0] if model in RNN else list(range(1, 101))
                    expected_mask = [1]*99 + [0] if model in RNN else [0] + [1]*99
                    if mapping["physical_cycle_by_slot"] != expected_ids or mapping["mask"] != expected_mask:
                        raise ValueError("physical cycle100 omitted or future input included")
        if published is not None:
            saved = published["models"][model]
            summary_rows = [row for row in published["summary"] if row["model"] == model]
            if len(summary_rows) != 1:
                raise ValueError("published comparison has missing/duplicate model rows")
            summary_row = summary_rows[0]
            expected_counts = dict(Counter(r["status"] for r in e["seeds"].values()))
            actual_status = summary_row["source_status"]
            if (full and actual_status != "complete") or (not full and (
                    actual_status == "complete" or json.loads(actual_status) != expected_counts)):
                raise ValueError("comparison source status disagrees with manifest")
            if summary_row["target_status"] != target["status"]:
                raise ValueError("comparison target status disagrees with manifest")
            if summary_row.get("failure_reason", "") != expected_failure_reason:
                raise ValueError("comparison failure reason disagrees with manifest")
            if not full and any(k.startswith("source_") and k not in {"source_status"}
                                for k in summary_row):
                raise ValueError("partial model received published source metrics")
            for protocol in ("all_windows", "cycle100_only", "sensitivity_without_flagged_cell", "target_metrics"):
                if protocol not in result:
                    if protocol in saved:
                        raise ValueError("partial/failed model received aggregate metrics")
                    continue
                for key, stats in saved[protocol].items():
                    for stat, value in stats.items():
                        compare({key: result[protocol][key][stat]}, {key: value})
                        prefix = "comac" if protocol == "target_metrics" else f"source_{protocol}"
                        suffix = "std" if stat == "std_population" else stat
                        compare({key: result[protocol][key][stat]},
                                {key: summary_row[f"{prefix}_{key}_{suffix}"]})
            if target["status"] == "complete":
                for cell in ("C1", "C2"):
                    rows = [r for r in target["rows"] if r["cell"] == cell]
                    values = [r["predicted_life_cycle"] for r in rows]
                    expected = {f"{cell}_predicted_mean": float(np.mean(values)),
                                f"{cell}_predicted_std": float(np.std(values, ddof=0)),
                                f"{cell}_absolute_error_mean": float(np.mean([r["absolute_error_cycles"] for r in rows])),
                                f"{cell}_ape_percent_mean": float(np.mean([r["ape_percent"] for r in rows]))}
                    if any(not np.isclose(v, summary_row[k], rtol=0, atol=1e-8) for k, v in expected.items()):
                        raise ValueError("published per-cell aggregate differs")
        results[model] = result
    if published is not None:
        if len(published["summary"]) != 13 or published["target_independent_cell_count"] != 2:
            raise ValueError("comparison scope or independent target cell count changed")
        if len(published["predictions"]) != len(target_keys):
            raise ValueError("published target row count differs from official manifest")
        actual_rows = {(r["model"], int(r["seed"]), r["cell"]): r
                       for e in m["models"].values() if e["target"]["status"] == "complete"
                       for r in e["target"]["rows"]}
        keys = [(r["model"], int(r["seed"]), r["cell"]) for r in published["predictions"]]
        if set(keys) != target_keys or len(keys) != len(set(keys)):
            raise ValueError("published target predictions missing/duplicated")
        for key, row in zip(keys, published["predictions"]):
            for name in ("predicted_life_cycle", "proxy_label_life_cycle", "absolute_error_cycles", "ape_percent"):
                if not np.isclose(row[name], actual_rows[key][name], rtol=0, atol=1e-8):
                    raise ValueError("published target prediction differs from manifest")
        csv_checks = {
            "comparison.csv": csv_matches_json(suite / "comparison.csv", published["summary"], ("model",)),
            "comac_predictions.csv": csv_matches_json(suite / "comac_predictions.csv", published["predictions"],
                                                     ("model", "seed", "cell")),
        }
    return {"passed": True, "final": not live, "source_terminal": terminal,
            "suite_id": m["suite_id"], "manifest_sha256": digest(suite / "manifest.json"),
            "target_prediction_rows": len(target_keys), "target_independent_cells": 2,
            "models": results, "csv_checks": csv_checks if published is not None else "not run in live mode",
            "no_model_inference_or_training": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--cp-arrays", type=Path, default=ROOT / "dataset/mixed95/audits/20261009_v1/remote_numeric")
    parser.add_argument("--live", action="store_true", help="never claims final completion")
    args = parser.parse_args()
    print(json.dumps(audit(args.suite.resolve(), args.cp_arrays.resolve(), args.live), indent=2))


if __name__ == "__main__":
    main()
