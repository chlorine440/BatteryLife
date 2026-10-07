"""Read-only CMU eVTOL CSV adapter for the independent 95% life experiment.

Each CSV restarts its local cycleNumber around reference performance tests.
Chronological contiguous runs, not the repeating local number, define the
observation order. RPT capacity is the C/5 stage-7 discharge, in mAh.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.comac95.data import THRESHOLD
from experiments.comac95.run import (CURVE_LENGTH, EARLY_CYCLES,
                                     _normalized_curve, _write_csv,
                                     prepare_comac, train, evaluate,
                                     DEFAULT_OUTPUT, DEFAULT_RAW_ROOT)

NOMINAL_AH = 3.0
REQUIRED_COLUMNS = ("time_s", "Ecell_V", "I_mA", "QCharge_mA_h",
                    "QDischarge_mA_h", "cycleNumber", "Ns")


def classify_runs(frame: pd.DataFrame):
    """Summarize chronological protocol runs, distinguishing RPT and mission."""
    if tuple(frame.columns) != REQUIRED_COLUMNS:
        raise ValueError(f"unexpected eVTOL CSV columns: {list(frame.columns)}")
    frame = frame.copy()
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("eVTOL CSV contains missing required fields")
    frame["run"] = frame["cycleNumber"].ne(frame["cycleNumber"].shift()).cumsum()
    grouped = frame.groupby("run", sort=False)
    summary = grouped.agg(local_cycle=("cycleNumber", "first"),
                          points=("Ns", "size"), time_start_s=("time_s", "first"),
                          time_end_s=("time_s", "last"),
                          qdis_mAh=("QDischarge_mA_h", "max"),
                          qchg_mAh=("QCharge_mA_h", "max"),
                          min_current_mA=("I_mA", "min"),
                          max_current_mA=("I_mA", "max"))
    for step in (3, 5, 8):
        summary[f"step{step}_points"] = frame["Ns"].eq(step).groupby(frame["run"]).sum()
    summary["rpt_protocol"] = (summary.step3_points.ge(1000) &
                               summary.step5_points.ge(1000))
    summary["rpt_discharge_mAh"] = (frame.loc[frame.Ns.eq(7)]
                                    .groupby("run").QDischarge_mA_h.max())
    summary["rpt_complete"] = (summary.rpt_protocol & summary.step8_points.ge(30) &
                               summary.rpt_discharge_mAh.between(1500, 3500))
    summary["mission_valid"] = (
        ~summary.rpt_protocol & summary.step8_points.eq(0) &
        summary.points.ge(100) & summary.qdis_mAh.between(500, 3000) &
        summary.qchg_mAh.gt(100) & summary.min_current_mA.lt(-1000) &
        summary.max_current_mA.gt(100))
    # The initial RPT is age zero; the next RPT follows 49 mission records
    # and represents the cycle-50 capacity check in the source protocol.
    missions_before = summary.mission_valid.cumsum().shift(fill_value=0)
    summary["observed_cycle"] = (missions_before + 1).astype(int)
    summary.loc[summary.index[0], "observed_cycle"] = 0
    return frame, summary


def extract_cell(path: Path):
    frame = pd.read_csv(path, usecols=list(REQUIRED_COLUMNS))
    frame = frame.loc[:, list(REQUIRED_COLUMNS)]
    frame, summary = classify_runs(frame)
    rpt_rows, issues = [], []
    for run_id, row in summary[summary.rpt_protocol].iterrows():
        capacity = float(row.rpt_discharge_mAh) / 1000 if row.rpt_complete else None
        rpt_rows.append({"cell": path.stem, "source_file": path.name,
                         "chronological_run": int(run_id),
                         "source_local_cycle": int(row.local_cycle),
                         "observed_cycle": int(row.observed_cycle),
                         "discharge_ah": capacity,
                         "soh_rpt_measured": capacity / NOMINAL_AH if capacity is not None else None,
                         "complete": bool(row.rpt_complete),
                         "label_source": "RPT_C_over_5_full_discharge"})
        if not row.rpt_complete:
            issues.append({"cell": path.stem, "run": int(run_id),
                           "reason": "incomplete_or_invalid_RPT"})
    completed = [row for row in rpt_rows if row["complete"]]
    crossing = next((row for row in completed if row["soh_rpt_measured"] <= THRESHOLD), None)
    life = crossing["observed_cycle"] if crossing else None
    previous = (next((row for row in reversed(completed[:completed.index(crossing)])
                      if row["soh_rpt_measured"] > THRESHOLD), None)
                if crossing else None)

    features = np.zeros((EARLY_CYCLES, 3, CURVE_LENGTH), dtype=np.float32)
    mask = np.zeros(EARLY_CYCLES, dtype=np.float32)
    early = frame[frame.run.between(2, EARLY_CYCLES + 1)]
    for run_id, samples in early.groupby("run", sort=False):
        if not summary.loc[run_id, "mission_valid"]:
            continue
        current = samples.I_mA.to_numpy(dtype=np.float64)
        voltage = samples.Ecell_V.to_numpy(dtype=np.float64)
        charge_capacity = samples.QCharge_mA_h.to_numpy(dtype=np.float64)
        discharge_capacity = samples.QDischarge_mA_h.to_numpy(dtype=np.float64)
        charge = np.column_stack((voltage[current > 30], current[current > 30] / 1000,
                                  charge_capacity[current > 30] / 1000))
        discharge = np.column_stack((voltage[current < -30], current[current < -30] / 1000,
                                     discharge_capacity[current < -30] / 1000))
        try:
            features[run_id - 2] = _normalized_curve(charge, discharge, NOMINAL_AH)
            mask[run_id - 2] = 1
        except ValueError:
            issues.append({"cell": path.stem, "run": int(run_id),
                           "reason": "invalid_early_curve"})
    audit = {"cell": path.stem, "source_file": path.name,
             "raw_chronological_runs": len(summary),
             "valid_mission_runs": int(summary.mission_valid.sum()),
             "rpt_protocol_runs": int(summary.rpt_protocol.sum()),
             "complete_rpt_runs": len(completed),
             "early_valid_curves": int(mask.sum()),
             "life_95_first_observed_rpt_cycle": life,
             "previous_above_95_rpt_cycle": previous["observed_cycle"] if previous else None,
             "label_is_interval_censored": crossing is not None}
    return life, features, mask, rpt_rows, audit, issues


def _stratified_splits(eligible, seed=2024):
    groups = {}
    for name, life in eligible.items():
        groups.setdefault(life, []).append(name)
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for life, names in sorted(groups.items()):
        rng.shuffle(names)
        if len(names) < 3:
            raise RuntimeError(f"eVTOL 95% label {life} has fewer than 3 cells")
        holdout = max(1, round(len(names) * 0.18))
        splits["val"].extend(names[:holdout])
        splits["test"].extend(names[holdout:2 * holdout])
        splits["train"].extend(names[2 * holdout:])
    for names in splits.values():
        names.sort()
    if len(set.union(*(set(names) for names in splits.values()))) != len(eligible):
        raise ValueError("eVTOL split overlap or omission")
    return splits


def prepare(raw_root: Path, output: Path, seed=2024):
    files = sorted((raw_root / "eVTOL").glob("VAH[0-9][0-9].csv"))
    if len(files) != 22:
        raise RuntimeError(f"expected 22 eVTOL source cells, found {len(files)}")
    output.mkdir(parents=True, exist_ok=True)
    all_cells, audit, issues, rpt_rows = {}, [], [], []
    for path in files:
        life, features, mask, rpt, cell_audit, cell_issues = extract_cell(path)
        all_cells[path.stem] = {"life": life, "features": features, "mask": mask,
                                "audit": cell_audit}
        audit.append(cell_audit)
        issues.extend(cell_issues)
        rpt_rows.extend(rpt)
        print(f"{path.stem}: life={life}, early={int(mask.sum())}/100, "
              f"RPT={cell_audit['complete_rpt_runs']}", flush=True)
    _write_csv(output / "evtol_rpt_95.csv", rpt_rows)
    _write_csv(output / "evtol_cell_audit_95.csv", audit)
    (output / "evtol_issues_95.json").write_text(json.dumps(issues, indent=2), encoding="utf-8")
    (output / "EVTOL_labels_95.json").write_text(json.dumps(
        {name: value["life"] for name, value in all_cells.items()}, indent=2), encoding="utf-8")
    eligible = {name: value["life"] for name, value in all_cells.items()
                if value["life"] is not None and value["life"] > EARLY_CYCLES
                and value["mask"].sum() >= 90}
    if len(eligible) < 10:
        raise RuntimeError("insufficient eVTOL 95% labels; see evtol_cell_audit_95.csv")
    splits = _stratified_splits(eligible, seed)
    report = {}
    for split, names in splits.items():
        report[split] = {"requested": len(names), "eligible": len(names),
                         "cells": names, "excluded": []}
        np.savez_compressed(output / f"evtol_{split}_95.npz",
                            features=np.asarray([all_cells[name]["features"] for name in names]),
                            masks=np.asarray([all_cells[name]["mask"] for name in names]),
                            lives=np.asarray([eligible[name] for name in names], dtype=np.float32),
                            cells=np.asarray(names))
    report["excluded_cells"] = sorted(set(all_cells) - set(eligible))
    (output / "evtol_eligibility_95.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if len(report["train"]["cells"]) < 5 or len(report["val"]["cells"]) < 2:
        raise RuntimeError("insufficient eVTOL 95% train/validation cells")
    comac = prepare_comac(raw_root, output)
    print(json.dumps({"evtol": report, "comac_proxy_life": comac}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.raw_root, args.output, args.seed)
    elif args.stage == "train":
        train(args.output, args.epochs, args.seed, args.batch_size, source="evtol")
    else:
        evaluate(args.output, source="evtol")


if __name__ == "__main__":
    main()
