"""Read-only COMAC preparation and one-time held-out evaluation of paper models."""

from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from experiments.comac95.data import NOMINAL_AH, SOC_SPAN, read_capacity_history, read_early_curves
from experiments.comac95.run import _normalized_curve
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
    curves = read_early_curves(raw_root, cell, max_physical_cycle=EARLY_CYCLES)
    features = np.zeros((EARLY_CYCLES, 3, CURVE_LENGTH), dtype=np.float32)
    mask = np.zeros(EARLY_CYCLES, dtype=np.float32)
    for physical_cycle, parts in curves.items():
        if not 2 <= physical_cycle <= EARLY_CYCLES:
            raise ValueError(f"future/RPT curve in {cell} input: {physical_cycle}")
        # The source loader divides both input current and capacity by nominal Ah.
        # The 0.94 SOC span belongs only to COMAC's life-label definition.
        features[physical_cycle - 1] = _normalized_curve(
            parts["charge"], parts["discharge"], NOMINAL_AH, capacity_span=1.0
        )
        mask[physical_cycle - 1] = 1
    if set(curves) != set(range(2, EARLY_CYCLES + 1)) or mask[0] != 0:
        raise ValueError(f"{cell}: missing or extra early physical cycles")
    if not np.isfinite(features).all():
        raise ValueError(f"{cell}: non-finite input features")
    return features, mask


def prepare_comac(raw_root: Path = DEFAULT_COMAC, output: Path = DEFAULT_OUTPUT) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_dir = output / "comac"
    target_dir.mkdir(parents=True, exist_ok=True)
    labels = {}
    for cell in ("C1", "C2"):
        ordinary, rpt, life, issues = read_capacity_history(raw_root, cell)
        if life != EXPECTED_LIFE[cell]:
            raise RuntimeError(f"{cell}: got proxy crossing {life}, expected {EXPECTED_LIFE[cell]}; inspect audit")
        features, mask = build_target_input(raw_root, cell)
        labels[cell] = life
        _write_csv(target_dir / f"{cell}_ordinary_proxy.csv", ordinary)
        _write_csv(target_dir / f"{cell}_rpt_measured.csv", rpt)
        (target_dir / f"{cell}_capacity_issues.json").write_text(
            json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.savez_compressed(target_dir / f"{cell}_input.npz", features=features, mask=mask,
                            physical_cycles=np.arange(1, EARLY_CYCLES + 1))

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
    destination = run_root / "held_out_comac_evaluation_95.json"
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
    target_dir = output / "comac"
    labels = json.loads((target_dir / "COMAC_proxy_labels_95.json").read_text(encoding="utf-8"))
    if labels != EXPECTED_LIFE:
        raise ValueError("COMAC proxy labels do not match audited 379/419 crossings")
    rows = []
    for seed in (2021, 42, 2024):
        checkpoint = Path(records[str(seed)]["checkpoint"])
        model, scaler = _load_paper_model(checkpoint)
        for cell in ("C1", "C2"):
            with np.load(target_dir / f"{cell}_input.npz") as data:
                features, mask, cycles = data["features"], data["mask"], data["physical_cycles"]
            if features.shape != (100, 3, 300) or mask.shape != (100,) or \
                    not np.array_equal(cycles, np.arange(1, 101)) or mask[0] != 0:
                raise ValueError(f"{cell}: invalid early input artifact")
            with torch.no_grad():
                x = torch.from_numpy(features[None]) * torch.from_numpy(mask[None, :, None, None])
                prediction_scaled = float(model(x, torch.from_numpy(mask[None])).item())
            prediction = float(scaler.inverse_transform([[prediction_scaled]])[0, 0])
            truth = labels[cell]
            rows.append({"seed": seed, "cell": cell, "predicted_life_cycle": prediction,
                         "proxy_label_life_cycle": truth,
                         "absolute_error_cycles": abs(prediction - truth),
                         "ape_percent": 100 * abs(prediction - truth) / truth})
    _write_csv(run_root / "held_out_comac_evaluation_95.csv", rows)
    report = {
        "source_test_by_seed": {seed: row["source_test"] for seed, row in records.items()},
        "comac": rows,
        "warning": "379/419 are ordinary-cycle proxy crossings, not full-RPT measured 95% lives.",
        "target_use": "C1/C2 held out; no training, normalization, fine-tuning or checkpoint selection",
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
