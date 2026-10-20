"""Map COMAC records to BatteryLife's schema, then use its unchanged loader.

Do not split by current sign here: the original loader owns the phase boundary,
rest filtering, resampling and feature normalization (including its endpoint
convention). COMAC's lifetime-cumulative Ah must first become per-cycle Ah.
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from experiments.comac95.data import NOMINAL_AH, XlsxRows, _cell_files, _number
from experiments.mixed95.prepare import EARLY_CYCLES


PREPROCESSING_VERSION = "batterylife_original_loader_v2"
TARGET_DIRECTORY = "comac_loader_v2"


def read_early_cycle_records(raw_root: Path, cell: str) -> dict:
    """Keep full chronological records for physical cycles 2..100, not RPT 1.

    Tuples are (Excel time, voltage, current, cumulative charge Ah,
    cumulative discharge Ah). Include rests and both capacity columns so the
    original loader, not an independent adapter, chooses the phase boundary.
    """
    _, first = _cell_files(raw_root, cell, first_segment_only=True)[0]
    paths = [first]
    continuation = first.with_name(f"{cell}-100-2.xlsx")
    if continuation.is_file():
        paths.append(continuation)
    records = defaultdict(list)
    fields = ("绝对时间", "电压(V)", "电流(A)",
              "累计充电容量(Ah)", "累计放电容量(Ah)")
    for path in paths:
        with XlsxRows(path) as book:
            for sheet in book.sheets:
                if not sheet.startswith("详细数据"):
                    continue
                rows = iter(book.rows(sheet))
                columns = {str(v).strip(): k for k, v in next(rows).items()}
                if any(k not in columns for k in ("循环序号", *fields)):
                    raise ValueError(f"{path}:{sheet}: missing detail columns")
                for row in rows:
                    cycle = _number(row.get(columns["循环序号"]))
                    if cycle is None or cycle <= 1 or cycle > EARLY_CYCLES:
                        continue
                    if cycle != int(cycle):
                        raise ValueError(f"{path}:{sheet}: non-integer cycle {cycle}")
                    point = tuple(_number(row.get(columns[k])) for k in fields)
                    if any(v is None for v in point) or min(point[3:]) < 0:
                        raise ValueError(f"{path}:{sheet}: invalid record in cycle {cycle}")
                    records[int(cycle)].append(point)
    # Stable timestamp sort preserves export order when timestamps are tied.
    return {cycle: sorted(dict.fromkeys(points), key=lambda p: p[0])
            for cycle, points in records.items()}


def cycle_frame(physical_cycle: int, records: list) -> pd.DataFrame:
    """Convert lifetime Ah to per-leg Ah without splitting or dropping rests."""
    a = np.asarray(records, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 5 or not np.isfinite(a).all():
        raise ValueError(f"cycle {physical_cycle}: invalid chronological records")
    if np.any(np.diff(a[:, 0]) < 0):
        raise ValueError(f"cycle {physical_cycle}: unordered records")
    positive = np.flatnonzero(a[:, 2] / NOMINAL_AH >= .01)
    negative = np.flatnonzero(a[:, 2] / NOMINAL_AH <= -.01)
    if len(positive) < 2 or len(negative) < 2 or positive[-1] >= negative[0]:
        raise ValueError(f"cycle {physical_cycle}: expected a complete charge-first cycle")
    # Raw '充电容量' resets between COMAC's multiple charge steps; subtracting
    # the lifetime counter's initial value preserves their combined capacity.
    charge = a[:, 3] - a[positive[0], 3]
    discharge = a[:, 4] - a[negative[0], 4]
    return pd.DataFrame({
        "cycle_number": physical_cycle,
        "voltage_in_V": a[:, 1], "current_in_A": a[:, 2],
        "charge_capacity_in_Ah": charge,
        "discharge_capacity_in_Ah": discharge,
        "time_in_s": (a[:, 0] - a[0, 0]) * 86400,
    })


def original_loader_features(cycles: dict) -> tuple[np.ndarray, np.ndarray]:
    """Delegate all curve feature processing to Dataset_original, unchanged."""
    from data_provider.data_loader import Dataset_original

    if set(cycles) != set(range(2, EARLY_CYCLES + 1)):
        raise ValueError("COMAC input must contain exactly physical cycles 2..100")
    df = pd.concat([cycle_frame(n, cycles[n]) for n in range(2, EARLY_CYCLES + 1)],
                   ignore_index=True)
    loader = Dataset_original.__new__(Dataset_original)
    loader.charge_discharge_len = 300
    loader.ZN_coin_charge_first_file_names = []
    features = loader.get_charge_discharge_curves(
        "COMAC_charge_first.pkl", df, EARLY_CYCLES, NOMINAL_AH
    ).astype(np.float32)
    mask = np.ones(EARLY_CYCLES, dtype=np.float32)
    mask[0] = 0
    if features.shape != (100, 3, 300) or not np.isfinite(features).all() or features[0].any():
        raise ValueError("invalid original-loader COMAC features or unmasked RPT")
    return features, mask


def load_target_input(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Reject legacy sign-split artifacts rather than silently reusing them."""
    with np.load(path, allow_pickle=False) as data:
        if "preprocessing_version" not in data or \
                str(data["preprocessing_version"].item()) != PREPROCESSING_VERSION:
            raise ValueError(f"{path}: regenerate COMAC inputs with the original-loader adapter")
        features, mask, cycles = data["features"], data["mask"], data["physical_cycles"]
    if features.shape != (100, 3, 300) or mask.shape != (100,) or \
            features.dtype != np.float32 or mask.dtype != np.float32 or \
            not np.array_equal(cycles, np.arange(1, 101)) or \
            not np.array_equal(mask, np.r_[0, np.ones(99)]) or \
            not np.isfinite(features).all() or features[0].any():
        raise ValueError(f"{path}: invalid early physical-cycle input")
    return features, mask
