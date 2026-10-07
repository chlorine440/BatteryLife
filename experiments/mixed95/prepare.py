"""Build auditable 95% life labels without changing the downloaded curves.

Run from the repository root with the experiment environment activated:
    python -m experiments.mixed95.prepare
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

from data_provider.data_split_recorder import split_recorder


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("/Users/curtischan/Datasets/batterylife")
DEFAULT_OUTPUT = ROOT / "dataset" / "mixed95"
THRESHOLD = 0.95
EXTRAPOLATION_LIMIT = 0.975
EARLY_CYCLES = 100
SPLITS = {
    "train": split_recorder.MIX_large_train_files,
    "val": split_recorder.MIX_large_val_files,
    "test": split_recorder.MIX_large_test_files,
}
MICH_EXP_FILES = set(
    split_recorder.MICH_EXP_train_files
    + split_recorder.MICH_EXP_val_files
    + split_recorder.MICH_EXP_test_files
)


def source_folder(file_name: str) -> str:
    if file_name in MICH_EXP_FILES:
        return "MICH_EXP"
    prefix = file_name.split("_", 1)[0]
    return {"UL-PUR": "UL_PUR", "ISU-ILCC": "ISU_ILCC"}.get(
        prefix, "Tongji" if prefix.startswith("Tongji") else prefix
    )


def label_file_and_key(file_name: str) -> tuple[str, str]:
    folder = source_folder(file_name)
    if folder in {"MICH", "MICH_EXP"}:
        return "total_MICH_labels.json", file_name
    if folder == "Tongji":
        # Dataset_original changes '--' to '-#' before its Tongji label lookup.
        return "Tongji_labels.json", file_name.replace("--", "-#")
    return f"{file_name.split('_', 1)[0]}_labels.json", file_name


def capacity_denominator(data: dict, file_name: str) -> tuple[float, float]:
    if file_name.startswith("RWTH"):
        nominal = 1.85
    elif file_name.startswith("SNL_18650_NCA_25C_20-80"):
        nominal = 3.2
    else:
        nominal = float(data["nominal_capacity_in_Ah"])
    low, high = data["SOC_interval"]
    span = abs(float(high) - float(low))
    if not math.isfinite(span) or span == 0:
        span = 1.0
    if not math.isfinite(nominal) or nominal <= 0:
        raise ValueError(f"invalid nominal capacity: {nominal}")
    return nominal, span


def cycle_soh(cycle: dict, nominal: float, span: float, file_name: str) -> float:
    discharge = np.asarray(cycle["discharge_capacity_in_Ah"], dtype=float)
    if file_name.startswith("XJTU"):
        # Retain the repository's XJTU discharge-only trace selection.
        current = np.asarray(cycle["current_in_A"], dtype=float)
        n = min(len(current), len(discharge))
        mask = np.isfinite(current[:n]) & np.isfinite(discharge[:n])
        discharge_mask = mask & (current[:n] < 0)
        values = discharge[:n][discharge_mask if discharge_mask.any() else mask]
    else:
        values = discharge[np.isfinite(discharge)]
    if not len(values):
        return math.nan
    qd = float(np.max(values))
    return qd / nominal / span if qd > 0 else math.nan


def life_from_soh(sohs: list[float], threshold: float = THRESHOLD) -> tuple[str, int | None]:
    """First observed crossing; paper's 20-cycle tail extrapolation if near it."""
    for cycle_number, soh in enumerate(sohs, start=1):
        if math.isfinite(soh) and soh <= threshold:
            if cycle_number <= EARLY_CYCLES:
                return "life_at_or_before_100", None
            return "observed", cycle_number

    if not sohs or not math.isfinite(sohs[-1]):
        return "invalid_terminal_soh", None
    terminal = sohs[-1]
    if terminal > threshold + 0.025 + 1e-12:
        return "not_reached", None
    if terminal <= threshold:
        return "invalid_crossing", None
    valid = [(i + 1, value) for i, value in enumerate(sohs) if math.isfinite(value)]
    if len(valid) < 20:
        return "insufficient_tail_for_extrapolation", None
    tail = valid[-20:]
    regressor = LinearRegression().fit(
        np.asarray([value for _, value in tail]).reshape(-1, 1),
        np.asarray([cycle for cycle, _ in tail]),
    )
    estimated = float(regressor.predict(np.asarray([[threshold]]))[0])
    if not math.isfinite(estimated) or estimated <= len(sohs):
        return "invalid_extrapolation", None
    life = int(estimated)  # Same truncation as Extract_life_labels.py.
    if life <= EARLY_CYCLES:
        return "life_at_or_before_100", None
    return "extrapolated", life


def _link(target: Path, link: Path) -> None:
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise FileExistsError(f"{link} points to another source")
    elif link.exists():
        raise FileExistsError(f"refusing to replace existing path: {link}")
    else:
        link.symlink_to(target.resolve())


def stage_input(source: Path, output: Path, grouped_labels: dict[str, dict[str, int]]) -> Path:
    input_root = output / "input"
    input_root.mkdir(parents=True, exist_ok=True)
    for folder in sorted({source_folder(name) for names in SPLITS.values() for name in names}):
        _link(source / folder, input_root / folder)
    _link(ROOT / "dataset" / "seen_unseen_labels", input_root / "seen_unseen_labels")

    # Original loader expects both Michigan subsets in total_MICH; symlinks avoid a 54 GB copy.
    combined = input_root / "total_MICH"
    combined.mkdir(exist_ok=True)
    for folder in ("MICH", "MICH_EXP"):
        for path in (source / folder).glob("*.pkl"):
            _link(path, combined / path.name)

    label_dir = input_root / "Life labels"
    label_dir.mkdir(exist_ok=True)
    for label_file, mapping in grouped_labels.items():
        (label_dir / label_file).write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return input_root


def prepare(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT) -> dict:
    identities = [name for names in SPLITS.values() for name in names]
    if len(identities) != len(set(identities)):
        raise ValueError("MIX_large splits overlap or contain duplicate cells")
    if any(name.startswith("C1") or name.startswith("C2") for name in identities):
        raise ValueError("COMAC cell in source split")
    output.mkdir(parents=True, exist_ok=True)

    audit = []
    labels = defaultdict(dict)
    counts = defaultdict(Counter)
    life_by_split = defaultdict(list)
    sample_by_split = Counter()
    for split, names in SPLITS.items():
        for index, file_name in enumerate(names, start=1):
            dataset = source_folder(file_name)
            path = source / dataset / file_name
            record = {"split": split, "dataset": dataset, "cell": file_name,
                      "source_path": str(path), "threshold": THRESHOLD}
            if not path.is_file() or path.stat().st_size == 0:
                record.update(status="missing_or_empty_file", life=None)
            else:
                try:
                    with path.open("rb") as handle:
                        data = pickle.load(handle)
                    nominal, span = capacity_denominator(data, file_name)
                    cycles = data["cycle_data"]
                    sohs = [cycle_soh(cycle, nominal, span, file_name) for cycle in cycles]
                    status, life = life_from_soh(sohs)
                    record.update(status=status, life=life, nominal_ah=nominal,
                                  soc_span=span, denominator_ah=nominal * span,
                                  n_cycles=len(cycles), n_finite_soh=sum(map(math.isfinite, sohs)),
                                  first_soh=next((v for v in sohs if math.isfinite(v)), None),
                                  last_soh=sohs[-1] if sohs and math.isfinite(sohs[-1]) else None)
                    if life is not None:
                        label_file, label_key = label_file_and_key(file_name)
                        labels[label_file][label_key] = life
                        life_by_split[split].append(life)
                        sample_by_split[split] += min(EARLY_CYCLES, len(cycles), life - 1)
                    del data
                except Exception as exc:
                    record.update(status="unreadable_or_invalid", life=None,
                                  error=f"{type(exc).__name__}: {exc}")
            audit.append(record)
            counts[split][record["status"]] += 1
            counts[dataset][record["status"]] += 1
            if index % 25 == 0 or index == len(names):
                print(f"{split}: {index}/{len(names)} inspected", flush=True)

    # A missing label file would make the original loader fail before it could skip a cell.
    for name in identities:
        label_file, _ = label_file_and_key(name)
        labels.setdefault(label_file, {})
    input_root = stage_input(source, output, labels)
    summary = {
        "threshold": THRESHOLD,
        "extrapolation_limit": EXTRAPOLATION_LIMIT,
        "source": str(source),
        "input_root": str(input_root),
        "split_counts": {split: dict(counts[split]) for split in SPLITS},
        "dataset_counts": {name: dict(counts[name]) for name in sorted({source_folder(i) for i in identities})},
        "eligible_cells": {split: len(life_by_split[split]) for split in SPLITS},
        "estimated_source_windows": dict(sample_by_split),
        "train_label_variation": len(set(life_by_split["train"])) > 1,
        "passed": (all(life_by_split[split] for split in SPLITS)
                   and len(set(life_by_split["train"])) > 1
                   and sample_by_split["train"] >= 256
                   and not any(row["status"] in {"missing_or_empty_file", "unreadable_or_invalid"} for row in audit)),
    }
    (output / "source_labels_audit_95.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output / "preflight_95.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["passed"]:
        raise RuntimeError(f"95% source preflight failed; inspect {output / 'preflight_95.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
