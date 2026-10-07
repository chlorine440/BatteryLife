"""Read-only COMAC C1/C2 adapter for the 95% nominal-capacity experiment.

The cycle-level label is a *proxy*, not an RPT measurement:
    ordinary discharge Ah / (183 Ah * 0.94).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET


NOMINAL_AH = 183.0
SOC_SPAN = 0.94
THRESHOLD = 0.95
NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def global_cycle(cell: str, segment: int, local_cycle: int) -> int:
    if cell not in {"C1", "C2"} or segment not in {100, 200, 300, 400, 500}:
        raise ValueError(f"unsupported COMAC cell/segment: {cell}/{segment}")
    if not 1 <= local_cycle <= 100:
        raise ValueError(f"invalid local cycle: {local_cycle}")
    return segment - 100 + local_cycle


def first_crossing(cycles_and_soh, threshold: float = THRESHOLD):
    """First measured/estimated cycle at or below the threshold, else None."""
    for cycle, soh in sorted(cycles_and_soh):
        if soh is not None and math.isfinite(soh) and soh <= threshold:
            return cycle
    return None


def proxy_soh(discharge_ah: float) -> float:
    if not math.isfinite(discharge_ah) or discharge_ah <= 0:
        raise ValueError(f"invalid ordinary discharge capacity: {discharge_ah}")
    return discharge_ah / (NOMINAL_AH * SOC_SPAN)


def deduplicate_trace(points):
    """Merge an overlapping -1/-2 detail export without duplicating samples."""
    unique = list(dict.fromkeys(points))
    if all(point[0] is not None for point in unique):
        unique.sort(key=lambda point: point[0])
    return [(voltage, current, capacity) for _, voltage, current, capacity in unique]


def _column_index(ref: str) -> int:
    result = 0
    for char in ref:
        if not char.isalpha():
            break
        result = result * 26 + ord(char.upper()) - ord("A") + 1
    return result - 1


class XlsxRows:
    """Small streaming XLSX reader; never loads multi-GB detail sheets into RAM."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.archive = zipfile.ZipFile(self.path)
        workbook = ET.fromstring(self.archive.read("xl/workbook.xml"))
        self.sheets = {
            sheet.attrib["name"]: f"xl/worksheets/sheet{idx}.xml"
            for idx, sheet in enumerate(workbook.findall(".//x:sheet", NS), start=1)
        }
        self.strings = []
        if "xl/sharedStrings.xml" in self.archive.namelist():
            root = ET.fromstring(self.archive.read("xl/sharedStrings.xml"))
            self.strings = ["".join(item.itertext()) for item in root]

    def close(self):
        self.archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def rows(self, sheet_name: str):
        if sheet_name not in self.sheets:
            raise KeyError(f"{self.path.name}: missing sheet {sheet_name}")
        with self.archive.open(self.sheets[sheet_name]) as stream:
            parser = ET.iterparse(stream, events=("start", "end"))
            _, root = next(parser)
            for event, element in parser:
                if event != "end" or element.tag.rsplit("}", 1)[-1] != "row":
                    continue
                values = {}
                for cell in element:
                    ref = cell.attrib.get("r", "")
                    value = cell.find("x:v", NS)
                    if not ref or value is None or value.text is None:
                        continue
                    raw = value.text
                    if cell.attrib.get("t") == "s":
                        raw = self.strings[int(raw)]
                    values[_column_index(ref)] = raw
                yield values
                element.clear()
                root.clear()


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _cell_files(raw_root: Path, cell: str, *, first_segment_only=False):
    folder = Path(raw_root) / {"C1": "Cell01", "C2": "Cell02"}[cell]
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    files = []
    for path in folder.glob("*.xlsx"):
        match = re.fullmatch(rf"{cell}-(100|200|300|400|500)(?:-1)?\.xlsx", path.name)
        if match:
            segment = int(match.group(1))
            if not first_segment_only or segment == 100:
                files.append((segment, path))
    if len(files) != (1 if first_segment_only else 5):
        raise ValueError(f"{cell}: expected {'1' if first_segment_only else '5'} summary workbooks, found {len(files)}")
    return sorted(files)


def read_capacity_history(raw_root: Path, cell: str):
    """Return ordinary-cycle rows, independent RPT rows, and the 95% proxy life."""
    ordinary, rpt, duplicate_log = {}, [], []
    for segment, path in _cell_files(raw_root, cell):
        with XlsxRows(path) as book:
            steps = iter(book.rows("工步数据表"))
            header = next(steps)
            columns = {str(value).strip(): key for key, value in header.items()}
            required = ("循环号", "工步号", "容量(Ah)")
            if any(key not in columns for key in required):
                raise ValueError(f"{path}: unexpected step-table columns")
            rpt_local = set()
            discharge_by_local = {}
            for row in steps:
                local = _number(row.get(columns["循环号"]))
                step = _number(row.get(columns["工步号"]))
                capacity = _number(row.get(columns["容量(Ah)"]))
                if local is None or step is None or capacity is None:
                    continue
                if int(step) == 6 and capacity > 100:
                    rpt_local.add(int(local))
                if int(step) == 8 and capacity > 100:
                    discharge_by_local[int(local)] = capacity
            for local in sorted(rpt_local):
                if local not in discharge_by_local:
                    raise ValueError(f"{path}: RPT cycle {local} lacks step-8 discharge")
                rpt.append({"cell": cell, "cycle": global_cycle(cell, segment, local),
                            "discharge_ah": discharge_by_local[local],
                            "soh_rpt_measured": discharge_by_local[local] / NOMINAL_AH,
                            "source_file": path.name, "source_sheet": "工步数据表",
                            "source_local_cycle": local, "label_source": "RPT_full_discharge"})

            summary = iter(book.rows("循环数据表"))
            header = next(summary)
            columns = {str(value).strip(): key for key, value in header.items()}
            if "循环序号" not in columns or "放电容量(Ah)" not in columns:
                raise ValueError(f"{path}: unexpected cycle-table columns")
            for row in summary:
                local = _number(row.get(columns["循环序号"]))
                capacity = _number(row.get(columns["放电容量(Ah)"]))
                if local is None or int(local) in rpt_local:
                    continue
                if capacity is None or capacity <= 0 or not 0.5 <= capacity / (NOMINAL_AH * SOC_SPAN) <= 1.2:
                    duplicate_log.append({"file": path.name, "local_cycle": local, "reason": "invalid_or_outlier_capacity"})
                    continue
                cycle = global_cycle(cell, segment, int(local))
                record = {"cell": cell, "cycle": cycle, "discharge_ah": capacity,
                          "soh_cycle_estimate": proxy_soh(capacity),
                          "source_file": path.name, "source_sheet": "循环数据表",
                          "source_local_cycle": int(local), "label_source": "ordinary_3_to_97_proxy"}
                if cycle in ordinary:
                    if ordinary[cycle]["discharge_ah"] != capacity:
                        raise ValueError(f"{cell} cycle {cycle}: conflicting duplicate capacity")
                    duplicate_log.append({"file": path.name, "local_cycle": local, "reason": "duplicate_same_value"})
                else:
                    ordinary[cycle] = record
    cycles = [ordinary[key] for key in sorted(ordinary)]
    life = first_crossing((row["cycle"], row["soh_cycle_estimate"]) for row in cycles)
    return cycles, sorted(rpt, key=lambda row: row["cycle"]), life, duplicate_log


def read_early_curves(raw_root: Path, cell: str, max_physical_cycle: int = 100):
    """Return charge/discharge traces keyed by physical cycle; RPT cycle 1 is absent."""
    if max_physical_cycle > 100 or max_physical_cycle < 2:
        raise ValueError("early COMAC input must be within physical cycles 2..100")
    segment, summary_file = _cell_files(raw_root, cell, first_segment_only=True)[0]
    paths = [summary_file]
    continuation = summary_file.with_name(f"{cell}-100-2.xlsx")
    if continuation.exists():
        paths.append(continuation)
    traces = defaultdict(lambda: {"charge": [], "discharge": []})
    for path in paths:
        with XlsxRows(path) as book:
            for sheet_name in book.sheets:
                if not sheet_name.startswith("详细数据"):
                    continue
                rows = iter(book.rows(sheet_name))
                header = next(rows)
                columns = {str(value).strip(): key for key, value in header.items()}
                required = ("循环序号", "电压(V)", "电流(A)", "累计充电容量(Ah)", "累计放电容量(Ah)")
                if any(key not in columns for key in required):
                    raise ValueError(f"{path}:{sheet_name}: unexpected detail columns")
                time_col = columns.get("绝对时间")
                for row in rows:
                    local = _number(row.get(columns["循环序号"]))
                    if local is None or local <= 1 or local > max_physical_cycle:
                        continue
                    voltage = _number(row.get(columns["电压(V)"]))
                    current = _number(row.get(columns["电流(A)"]))
                    if voltage is None or current is None:
                        continue
                    kind = "charge" if current > NOMINAL_AH * 0.01 else "discharge" if current < -NOMINAL_AH * 0.01 else None
                    if kind is None:
                        continue
                    cap_name = "累计充电容量(Ah)" if kind == "charge" else "累计放电容量(Ah)"
                    capacity = _number(row.get(columns[cap_name]))
                    if capacity is None or capacity < 0:
                        continue
                    time = _number(row.get(time_col)) if time_col is not None else None
                    traces[int(local)][kind].append((time, voltage, current, capacity))
    result = {}
    for cycle, parts in traces.items():
        if not parts["charge"] or not parts["discharge"]:
            continue
        cleaned = {}
        for kind, points in parts.items():
            cleaned[kind] = deduplicate_trace(points)
        result[cycle] = cleaned
    return result


def main():
    """Allow label/Excel QA before numerical training packages are installed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/Users/curtischan/Datasets/comac/raw/商飞800VSOH-tight/商飞800VSOH-tight"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "dataset" / "comac95")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    labels = {}
    for cell in ("C1", "C2"):
        ordinary, rpt, life, issues = read_capacity_history(args.raw_root, cell)
        labels[cell] = life
        for name, rows in ((f"{cell}_ordinary_proxy_95.csv", ordinary),
                           (f"{cell}_rpt_measured.csv", rpt)):
            with (args.output / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (args.output / f"{cell}_capacity_issues.json").write_text(json.dumps(issues, indent=2), encoding="utf-8")
    (args.output / "COMAC_proxy_labels_95.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    if labels != {"C1": 379, "C2": 419}:
        raise RuntimeError(f"unexpected proxy threshold crossings: {labels}")
    print(json.dumps(labels, indent=2))


if __name__ == "__main__":
    main()
