"""Check that every 95%-eligible source cell reaches the original curve loader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.mixed95.prepare import DEFAULT_OUTPUT


def check_curves(output: Path = DEFAULT_OUTPUT) -> dict:
    from data_provider.data_loader import Dataset_original
    from utils.augmentation import BatchAugmentation_battery_revised

    preflight = json.loads((output / "preflight_95.json").read_text(encoding="utf-8"))
    if not preflight["passed"]:
        raise RuntimeError("source label preflight failed")
    cells = json.loads((output / "source_labels_audit_95.json").read_text(encoding="utf-8"))
    loader = Dataset_original.__new__(Dataset_original)
    loader.root_path = preflight["input_root"]
    loader.early_cycle_threshold = 100
    loader.charge_discharge_len = 300
    loader.need_keys = ["current_in_A", "voltage_in_V", "charge_capacity_in_Ah",
                        "discharge_capacity_in_Ah", "time_in_s"]
    loader.ZN_coin_charge_first_file_names = []  # No Zn-ion cells in MIX_large.
    loader.aug_helper = BatchAugmentation_battery_revised()
    issues = []
    checked = 0
    for row in cells:
        if row["life"] is None:
            continue
        try:
            _, curves, life, _, _, valid_cycles = loader.read_cell_df(row["cell"])
            if life != row["life"] or valid_cycles < 100 or curves.shape != (100, 3, 300):
                raise ValueError(f"inconsistent life/cycle count/shape: {life}, {valid_cycles}, {curves.shape}")
            if not np.isfinite(curves).all():
                raise ValueError("non-finite curve feature")
        except Exception as exc:
            issues.append({"cell": row["cell"], "split": row["split"],
                           "error": f"{type(exc).__name__}: {exc}"})
        checked += 1
        if checked % 50 == 0:
            print(f"source curves: {checked} checked, {len(issues)} issues", flush=True)
    report = {"checked": checked, "issues": issues, "passed": not issues}
    (output / "source_curves_audit_95.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"source curves: {checked} checked, {len(issues)} issues", flush=True)
    if issues:
        raise RuntimeError("source curve preflight failed; inspect source_curves_audit_95.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    check_curves(args.output)


if __name__ == "__main__":
    main()
