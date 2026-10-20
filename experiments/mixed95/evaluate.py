"""Read-only COMAC preparation and one-time held-out evaluation of paper models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from experiments.comac95.data import read_capacity_history
from experiments.mixed95.comac_input import (
    PREPROCESSING_VERSION, TARGET_DIRECTORY, load_target_input,
    original_loader_features, read_early_cycle_records,
)
from experiments.mixed95.prepare import DEFAULT_OUTPUT, EARLY_CYCLES


DEFAULT_COMAC = Path("/Users/curtischan/Datasets/comac/raw/商飞800VSOH-tight/商飞800VSOH-tight")
EXPECTED_LIFE = {"C1": 379, "C2": 419}
CURVE_LENGTH = 300


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_target_input(raw_root: Path, cell: str) -> tuple[np.ndarray, np.ndarray]:
    """Use physical cycles 1..100, with RPT cycle 1 zeroed and masked."""
    return original_loader_features(read_early_cycle_records(raw_root, cell))


def prepare_comac(raw_root: Path = DEFAULT_COMAC, output: Path = DEFAULT_OUTPUT) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_dir = output / TARGET_DIRECTORY
    if target_dir.exists():
        raise FileExistsError(f"refusing to overwrite corrected COMAC inputs: {target_dir}")
    prepared = {}
    labels = {}
    for cell in ("C1", "C2"):
        ordinary, rpt, life, issues = read_capacity_history(raw_root, cell)
        if life != EXPECTED_LIFE[cell]:
            raise RuntimeError(f"{cell}: got proxy crossing {life}, expected {EXPECTED_LIFE[cell]}; inspect audit")
        features, mask = build_target_input(raw_root, cell)
        labels[cell] = life
        prepared[cell] = (ordinary, rpt, issues, features, mask)
    target_dir.mkdir(parents=True)
    for cell, (ordinary, rpt, issues, features, mask) in prepared.items():
        life = labels[cell]
        _write_csv(target_dir / f"{cell}_ordinary_proxy.csv", ordinary)
        _write_csv(target_dir / f"{cell}_rpt_measured.csv", rpt)
        (target_dir / f"{cell}_capacity_issues.json").write_text(
            json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.savez_compressed(target_dir / f"{cell}_input.npz", features=features, mask=mask,
                            physical_cycles=np.arange(1, EARLY_CYCLES + 1),
                            preprocessing_version=np.asarray(PREPROCESSING_VERSION))

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot([r["cycle"] for r in ordinary], [100 * r["soh_cycle_estimate"] for r in ordinary],
                linewidth=1, label="Ordinary-cycle SOH proxy")
        ax.scatter([r["cycle"] for r in rpt], [100 * r["soh_rpt_measured"] for r in rpt],
                   label="Full-discharge RPT measurement")
        ax.axhline(95, color="grey", linestyle="--")
        ax.axvline(life, color="tab:red", linestyle=":", label=f"Proxy crossing: {life}")
        ax.set(xlabel="Physical cycle", ylabel="SOH (%)", title=f"{cell}: proxy and measured RPT")
        ax.legend()
        fig.tight_layout()
        fig.savefig(target_dir / f"{cell}_proxy_vs_rpt.png", dpi=150)
        plt.close(fig)
    (target_dir / "COMAC_proxy_labels_95.json").write_text(
        json.dumps(labels, indent=2), encoding="utf-8"
    )
    (target_dir / "input_provenance.json").write_text(json.dumps({
        "preprocessing_version": PREPROCESSING_VERSION,
        "feature_method": "Dataset_original.get_charge_discharge_curves (unchanged)",
        "loader_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[2] / "data_provider/data_loader.py").read_bytes()
        ).hexdigest(),
        "physical_cycles": [1, 100], "masked_rpt_cycles": [1],
        "capacity_feature_denominator_ah": 183,
        "label_formula": "ordinary_discharge_ah / 0.94 / 183",
        "legacy_input_preserved": str(output / "comac"),
        "input_sha256": {cell: hashlib.sha256(
            (target_dir / f"{cell}_input.npz").read_bytes()).hexdigest() for cell in labels},
    }, indent=2), encoding="utf-8")
    return labels


def _load_paper_model(checkpoint: Path):
    import joblib
    import torch
    from models.CPTransformer import Model

    args = json.loads((checkpoint / "args.json").read_text(encoding="utf-8"))
    if (args.get("model"), args.get("dataset"), args.get("d_model"), args.get("d_ff"),
        args.get("e_layers"), args.get("d_layers")) != (
            "CPTransformer", "MIX_large", 256, 64, 1, 12
        ):
        raise ValueError(f"checkpoint is not paper-configured mixed95: {checkpoint}")
    model = Model(Namespace(**args)).float()
    safe_path = checkpoint / "model.safetensors"
    bin_path = checkpoint / "pytorch_model.bin"
    if safe_path.is_file():
        from safetensors.torch import load_file
        state = load_file(str(safe_path), device="cpu")
    elif bin_path.is_file():
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"no saved model weights in {checkpoint}")
    model.load_state_dict(state)
    model.eval()
    scaler = joblib.load(checkpoint / "label_scaler")
    return model, scaler


def evaluate(output: Path = DEFAULT_OUTPUT, *, run_id: str = "paper95_v1") -> dict:
    import torch

    run_root = output / "runs" / run_id
    destination = run_root / "held_out_comac_evaluation_95_loader_v2.json"
    if destination.exists():
        raise FileExistsError(f"COMAC has already been evaluated for this run: {destination}")
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("smoke_only"):
        raise ValueError("smoke-test weights may not be evaluated on C1/C2")
    records = manifest["seeds"]
    if set(records) != {"2021", "42", "2024"} or any(
        row.get("status") != "complete" for row in records.values()
    ):
        raise RuntimeError("all three source-trained seeds must complete before COMAC evaluation")
    target_dir = output / TARGET_DIRECTORY
    labels = json.loads((target_dir / "COMAC_proxy_labels_95.json").read_text(encoding="utf-8"))
    if labels != EXPECTED_LIFE:
        raise ValueError("COMAC proxy labels do not match audited 379/419 crossings")
    inputs = {cell: load_target_input(target_dir / f"{cell}_input.npz") for cell in labels}
    rows = []
    for seed in (2021, 42, 2024):
        checkpoint = Path(records[str(seed)]["checkpoint"])
        model, scaler = _load_paper_model(checkpoint)
        for cell in ("C1", "C2"):
            features, mask = inputs[cell]
            with torch.no_grad():
                x = torch.from_numpy(features[None]) * torch.from_numpy(mask[None, :, None, None])
                prediction_scaled = float(model(x, torch.from_numpy(mask[None])).item())
            prediction = float(scaler.inverse_transform([[prediction_scaled]])[0, 0])
            truth = labels[cell]
            rows.append({"seed": seed, "cell": cell, "predicted_life_cycle": prediction,
                         "proxy_label_life_cycle": truth,
                         "absolute_error_cycles": abs(prediction - truth),
                         "ape_percent": 100 * abs(prediction - truth) / truth})
    _write_csv(run_root / "held_out_comac_evaluation_95_loader_v2.csv", rows)
    report = {
        "source_test_by_seed": {seed: row["source_test"] for seed, row in records.items()},
        "comac": rows,
        "warning": "379/419 are ordinary-cycle proxy crossings, not full-RPT measured 95% lives.",
        "target_use": "C1/C2 held out; no training, normalization, fine-tuning or checkpoint selection",
        "target_preprocessing_version": PREPROCESSING_VERSION,
        "target_input_dir": str(target_dir),
        "target_input_sha256": {cell: hashlib.sha256(
            (target_dir / f"{cell}_input.npz").read_bytes()).hexdigest() for cell in labels},
        "prior_evaluation_preserved": (run_root / "held_out_comac_evaluation_95.json").exists(),
    }
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-comac", "evaluate"))
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_COMAC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default="paper95_v1")
    args = parser.parse_args()
    result = (prepare_comac(args.raw_root, args.output) if args.command == "prepare-comac"
              else evaluate(args.output, run_id=args.run_id))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
