"""Independent XJTU -> COMAC CPTransformer experiment at 95% SOH.

Run from the repository root. See README.md in this directory for commands and
for the distinction between COMAC proxy labels and full-RPT measurements.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import pickle
import random
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.comac95.data import (NOMINAL_AH, SOC_SPAN, THRESHOLD,
                                      first_crossing, read_capacity_history,
                                      read_early_curves)

EARLY_CYCLES = 100
CURVE_LENGTH = 300
DEFAULT_RAW_ROOT = Path("/Users/curtischan/Datasets/comac/raw")
DEFAULT_OUTPUT = ROOT / "dataset" / "comac95"


def _field(record, key):
    return record[key] if isinstance(record, dict) else getattr(record, key)


def _resample(points, size=CURVE_LENGTH // 2):
    if len(points) < 2:
        raise ValueError("too few charge/discharge points")
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError("invalid curve values")
    old = np.arange(1, len(array) + 1, dtype=np.float64)
    new = np.linspace(1, len(array) + 1, num=size)
    return np.stack([np.interp(new, old, array[:, index]) for index in range(3)])


def _normalized_curve(charge, discharge, nominal_ah, capacity_span=1.0):
    """Match BatteryLife feature order: voltage, C-rate, capacity fraction."""
    charge = np.asarray(charge, dtype=np.float64).copy()
    discharge = np.asarray(discharge, dtype=np.float64).copy()
    # Some cyclers (including COMAC) export lifetime-cumulative capacity.
    # BatteryLife expects capacity traversed within each charge/discharge leg.
    if len(charge):
        charge[:, 2] -= charge[0, 2]
    if len(discharge):
        discharge[:, 2] -= discharge[0, 2]
    charge = _resample(charge)
    discharge = _resample(discharge)
    joined = np.concatenate([charge, discharge], axis=1)
    voltage_scale = np.max(joined[0])
    if voltage_scale <= 0:
        raise ValueError("invalid voltage scale")
    joined[0] /= voltage_scale
    joined[1] /= nominal_ah
    joined[2] /= nominal_ah * capacity_span
    return joined.astype(np.float32)


def _xjtu_curve(cycle):
    current = np.asarray(_field(cycle, "current_in_A"), dtype=np.float64)
    voltage = np.asarray(_field(cycle, "voltage_in_V"), dtype=np.float64)
    charge = np.asarray(_field(cycle, "charge_capacity_in_Ah"), dtype=np.float64)
    discharge = np.asarray(_field(cycle, "discharge_capacity_in_Ah"), dtype=np.float64)
    n = min(map(len, (current, voltage, charge, discharge)))
    current, voltage, charge, discharge = (x[:n] for x in (current, voltage, charge, discharge))
    charge_mask = np.isfinite(current) & (current >= 0.02)
    discharge_mask = np.isfinite(current) & (current <= -0.02)
    if charge_mask.sum() < 2 or discharge_mask.sum() < 2:
        raise ValueError("missing XJTU charge/discharge trace")
    charge_points = np.column_stack((voltage[charge_mask], current[charge_mask], charge[charge_mask]))
    discharge_points = np.column_stack((voltage[discharge_mask], current[discharge_mask], discharge[discharge_mask]))
    return _normalized_curve(charge_points, discharge_points, 2.0)


def _load_xjtu_raw(path: Path, name: str):
    """Use BatteryLife's own XJTU organize/split functions without its CLI."""
    import pandas as pd
    from scipy.io import loadmat

    # process_scripts/__init__.py is a BatteryML registration file, not needed here.
    package = types.ModuleType("process_scripts")
    package.__path__ = [str(ROOT / "process_scripts")]
    sys.modules.setdefault("process_scripts", package)
    module = importlib.import_module("process_scripts.preprocess_XJTU")
    data = loadmat(str(path))["data"]
    frames = []
    for cycle in range(1, data.shape[1] + 1):
        frame = module.get_one_cycle(data, cycle)
        frame["cycle_number"] = cycle
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table = module.split_capacity_column(table, "cycle_number", "current_A", "capacity_Ah", 2.0)
    return module.organize_cell(table, f"XJTU_{name}", path.parent.name)


def _xjtu_raw_path(raw_root: Path, name: str):
    root = raw_root / "XJTU"
    stem = Path(name).stem.removeprefix("XJTU_")
    matches = [root / batch / f"{stem}.mat" for batch in ("Batch-1", "Batch-2")]
    present = [path for path in matches if path.is_file()]
    if len(present) != 1:
        raise FileNotFoundError(f"expected one raw MAT for {name}: {matches}")
    return present[0]


def _xjtu_life_and_features(battery):
    cycles = _field(battery, "cycle_data")
    nominal = float(_field(battery, "nominal_capacity_in_Ah"))
    if not math.isclose(nominal, 2.0):
        raise ValueError(f"unexpected XJTU nominal capacity: {nominal}")
    soh_points = []
    by_cycle = {}
    for cycle in cycles:
        number = int(_field(cycle, "cycle_number"))
        current = np.asarray(_field(cycle, "current_in_A"), dtype=np.float64)
        discharge = np.asarray(_field(cycle, "discharge_capacity_in_Ah"), dtype=np.float64)
        n = min(len(current), len(discharge))
        valid = np.isfinite(current[:n]) & np.isfinite(discharge[:n]) & (current[:n] < -0.02)
        if valid.any():
            qd = float(np.max(discharge[:n][valid]))
            if 0.5 <= qd / nominal <= 1.2:
                soh_points.append((number, qd / nominal))
        if 1 <= number <= EARLY_CYCLES:
            by_cycle[number] = cycle
    life = first_crossing(soh_points)
    features = np.zeros((EARLY_CYCLES, 3, CURVE_LENGTH), dtype=np.float32)
    mask = np.zeros(EARLY_CYCLES, dtype=np.float32)
    for number, cycle in by_cycle.items():
        try:
            features[number - 1] = _xjtu_curve(cycle)
            mask[number - 1] = 1
        except ValueError:
            pass
    return life, features, mask


def _write_csv(path: Path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def prepare(raw_root: Path, output: Path, *, xjtu_processed: Path | None = None):
    from data_provider.data_split_recorder import split_recorder

    output.mkdir(parents=True, exist_ok=True)
    split_files = {
        "train": split_recorder.XJTU_train_files,
        "val": split_recorder.XJTU_val_files,
        "test": split_recorder.XJTU_test_files,
    }
    identities = [name for names in split_files.values() for name in names]
    if len(identities) != len(set(identities)):
        raise ValueError("XJTU cell leakage across splits")
    source_labels, report = {}, {}
    for split, names in split_files.items():
        features, masks, lives, kept = [], [], [], []
        exclusions = []
        for name in names:
            processed_path = xjtu_processed / name if xjtu_processed else None
            if processed_path and processed_path.exists():
                with processed_path.open("rb") as handle:
                    battery = pickle.load(handle)
            else:
                raw_path = _xjtu_raw_path(raw_root, name)
                battery = _load_xjtu_raw(raw_path, Path(name).stem.removeprefix("XJTU_"))
            life, curve, mask = _xjtu_life_and_features(battery)
            source_labels[name] = life
            if life is None:
                exclusions.append({"cell": name, "reason": "no_observed_95_crossing"})
            elif life <= EARLY_CYCLES:
                exclusions.append({"cell": name, "reason": "crossing_within_input_window", "life": life})
            elif mask.sum() < 90:
                exclusions.append({"cell": name, "reason": "less_than_90_valid_early_cycles", "valid": int(mask.sum())})
            else:
                features.append(curve)
                masks.append(mask)
                lives.append(life)
                kept.append(name)
        report[split] = {"requested": len(names), "eligible": len(kept), "excluded": exclusions, "cells": kept}
        np.savez_compressed(output / f"xjtu_{split}_95.npz",
                            features=np.asarray(features, dtype=np.float32),
                            masks=np.asarray(masks, dtype=np.float32),
                            lives=np.asarray(lives, dtype=np.float32),
                            cells=np.asarray(kept))
    (output / "XJTU_labels_95.json").write_text(json.dumps(source_labels, indent=2), encoding="utf-8")
    (output / "xjtu_eligibility_95.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["train"]["eligible"] < 5 or report["val"]["eligible"] < 2 or report["test"]["eligible"] < 1:
        raise RuntimeError("insufficient XJTU 95% cells; see xjtu_eligibility_95.json. No training should start.")

    prepare_comac(raw_root, output)
    print(json.dumps({"xjtu": report, "comac_proxy_life": {"C1": 379, "C2": 419}}, indent=2))


def prepare_comac(raw_root: Path, output: Path):
    """Build held-out COMAC inputs without using them for source selection."""
    output.mkdir(parents=True, exist_ok=True)
    comac_labels = {}
    for cell in ("C1", "C2"):
        folder = raw_root / "商飞800VSOH-tight" / "商飞800VSOH-tight"
        history, rpt, life, issues = read_capacity_history(folder, cell)
        comac_labels[cell] = life
        _write_csv(output / f"{cell}_ordinary_proxy_95.csv", history)
        _write_csv(output / f"{cell}_rpt_measured.csv", rpt)
        (output / f"{cell}_capacity_issues.json").write_text(json.dumps(issues, indent=2), encoding="utf-8")
        if life is None:
            raise RuntimeError(f"{cell} has no observed proxy 95% crossing")
        if life != {"C1": 379, "C2": 419}[cell]:
            raise RuntimeError(f"{cell} first crossing changed to {life}; inspect source rows and issue log")
        raw_curves = read_early_curves(folder, cell)
        features = np.zeros((EARLY_CYCLES, 3, CURVE_LENGTH), dtype=np.float32)
        mask = np.zeros(EARLY_CYCLES, dtype=np.float32)
        for physical_cycle, parts in raw_curves.items():
            if physical_cycle > EARLY_CYCLES or physical_cycle == 1:
                raise ValueError("COMAC input contains RPT or future cycle")
            try:
                features[physical_cycle - 1] = _normalized_curve(
                    parts["charge"], parts["discharge"], NOMINAL_AH, SOC_SPAN)
                mask[physical_cycle - 1] = 1
            except ValueError:
                pass
        if mask.sum() < 90 or mask[0] != 0:
            raise RuntimeError(f"{cell}: invalid early curve coverage ({int(mask.sum())}/99)")
        np.savez_compressed(output / f"{cell}_early_95.npz", features=features, mask=mask,
                            physical_cycles=np.arange(1, EARLY_CYCLES + 1))
    (output / "COMAC_proxy_labels_95.json").write_text(json.dumps(comac_labels, indent=2), encoding="utf-8")
    return comac_labels


def _load_split(output: Path, split: str, source: str = "xjtu"):
    if source not in {"xjtu", "evtol"}:
        raise ValueError(f"unsupported source: {source}")
    path = output / f"{source}_{split}_95.npz"
    if not path.exists():
        raise FileNotFoundError(f"run prepare first: {path}")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _model(device):
    from argparse import Namespace
    from models.CPTransformer import Model
    config = Namespace(d_ff=128, d_model=64, charge_discharge_length=CURVE_LENGTH,
                       early_cycle_threshold=EARLY_CYCLES, dropout=0.1, e_layers=2,
                       d_layers=2, factor=3, n_heads=4, activation="gelu", output_num=1)
    return Model(config).to(device)


def _predict(model, features, mask, device, mean, std):
    import torch
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        m = torch.as_tensor(mask, dtype=torch.float32, device=device)
        raw = model(x * m[:, :, None, None], m).squeeze(-1)
        return (raw.cpu().numpy() * std + mean).tolist()


def train(output: Path, epochs: int, seed: int, batch_size: int, source: str = "xjtu"):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_data, val_data = _load_split(output, "train", source), _load_split(output, "val", source)
    eligibility = json.loads((output / f"{source}_eligibility_95.json").read_text())
    if eligibility["train"]["eligible"] < 5 or eligibility["val"]["eligible"] < 2:
        raise RuntimeError(f"{source} preflight did not pass")
    mean, std = float(train_data["lives"].mean()), float(train_data["lives"].std())
    if not std > 0:
        raise RuntimeError("training labels have no variation")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = _model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Keep one copy per cell; materializing all 91 windows would take several GB.
    x = torch.from_numpy(train_data["features"])
    mask = torch.from_numpy(train_data["masks"])
    y = torch.from_numpy(train_data["lives"])
    target = (y - mean) / std
    windows = [(cell, anchor) for cell in range(len(y))
               for anchor in range(10, EARLY_CYCLES + 1)]
    best, stale = float("inf"), 0
    checkpoint = output / f"cptransformer_{source}_95.pt"
    log = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(windows))
        losses = []
        for selection in order.split(batch_size):
            cells = torch.tensor([windows[int(idx)][0] for idx in selection], dtype=torch.long)
            anchors = torch.tensor([windows[int(idx)][1] for idx in selection], dtype=torch.long)
            mm = mask[cells] * (torch.arange(EARLY_CYCLES)[None] < anchors[:, None])
            xx = (x[cells] * mm[:, :, None, None]).to(device)
            mm = mm.to(device)
            yy = target[cells].to(device)
            optimizer.zero_grad()
            prediction = model(xx, mm).squeeze(-1)
            loss = torch.nn.functional.mse_loss(prediction, yy)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_prediction = _predict(model, val_data["features"], val_data["masks"], device, mean, std)
        val_mae = float(np.mean(np.abs(np.asarray(val_prediction) - val_data["lives"])))
        log.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "val_cell_mae_cycles": val_mae})
        print(f"epoch {epoch}: train_mse={log[-1]['train_mse']:.5f} val_mae={val_mae:.2f} cycles", flush=True)
        if val_mae < best:
            best, stale = val_mae, 0
            torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                        "seed": seed, "threshold": THRESHOLD, "val_mae": best,
                        "source": source}, checkpoint)
        else:
            stale += 1
            if stale >= 10:
                break
    (output / f"training_log_{source}_95.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"checkpoint: {checkpoint}")


def evaluate(output: Path, source: str = "xjtu"):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    checkpoint = output / f"cptransformer_{source}_95.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"train first: {checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload["threshold"] != THRESHOLD:
        raise ValueError("wrong checkpoint threshold")
    if payload.get("source", "xjtu") != source:
        raise ValueError("wrong checkpoint source")
    model = _model(device)
    model.load_state_dict(payload["state_dict"])
    mean, std = payload["mean"], payload["std"]
    rows = []
    test = _load_split(output, "test", source)
    for cell, true, predicted in zip(test["cells"], test["lives"],
                                     _predict(model, test["features"], test["masks"], device, mean, std)):
        rows.append({"dataset": source.upper(), "cell": str(cell),
                     "label_type": "RPT_full_discharge" if source == "evtol" else "full_discharge_measured",
                     "true_life_cycle": int(true), "predicted_life_cycle": predicted,
                     "absolute_error_cycles": abs(predicted - true), "ape_percent": 100 * abs(predicted - true) / true})
    for cell in ("C1", "C2"):
        with np.load(output / f"{cell}_early_95.npz") as data:
            curve = data["features"][None]
            mask = data["mask"][None]
        truth = json.loads((output / "COMAC_proxy_labels_95.json").read_text())[cell]
        predicted = _predict(model, curve, mask, device, mean, std)[0]
        rows.append({"dataset": "COMAC", "cell": cell, "label_type": "ordinary_3_to_97_proxy",
                     "true_life_cycle": truth, "predicted_life_cycle": predicted,
                     "absolute_error_cycles": abs(predicted - truth), "ape_percent": 100 * abs(predicted - truth) / truth})
        with (output / f"{cell}_ordinary_proxy_95.csv").open(encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        with (output / f"{cell}_rpt_measured.csv").open(encoding="utf-8") as handle:
            rpt = list(csv.DictReader(handle))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot([int(r["cycle"]) for r in history], [100 * float(r["soh_cycle_estimate"]) for r in history],
                linewidth=1, label="Ordinary-cycle SOH proxy")
        ax.scatter([int(r["cycle"]) for r in rpt], [100 * float(r["soh_rpt_measured"]) for r in rpt],
                   label="Full-discharge RPT measurement")
        ax.axhline(95, color="grey", linestyle="--")
        ax.set(xlabel="Physical cycle", ylabel="SOH (%)", title=f"{cell}: proxy vs RPT")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{cell}_soh_proxy_vs_rpt_{source}_95.png", dpi=150)
        plt.close(fig)
    suffix = "95" if source == "xjtu" else f"{source}_95"
    _write_csv(output / f"evaluation_{suffix}.csv", rows)
    (output / f"evaluation_{suffix}.json").write_text(json.dumps({
        "threshold": THRESHOLD, "train_source": source,
        "target_use": "C1/C2 held-out evaluation only",
        "warning": "COMAC 379/419 are ordinary-cycle proxy crossings, not full-RPT measured crossings.",
        "source_test_mae_cycles": float(np.mean([r["absolute_error_cycles"] for r in rows if r["dataset"] != "COMAC"])),
        "predictions": rows,
    }, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--xjtu-processed", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.raw_root, args.output, xjtu_processed=args.xjtu_processed)
    elif args.stage == "train":
        train(args.output, args.epochs, args.seed, args.batch_size)
    else:
        evaluate(args.output)


if __name__ == "__main__":
    main()
