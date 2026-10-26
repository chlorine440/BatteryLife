"""Auditable inputs for a Che III-A style C3 reconstruction (not replication).

The old Ridge exploration, raw workbooks and frozen lifetime experiments are
never changed. Only charge features are shared between training and inference.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import pickle

import numpy as np

from experiments.comac95.data import XlsxRows, _number
from experiments.mixed95.c3_evaluate import ROOT, RAW, OUTPUT, rows_named, sha
from experiments.mixed95.c3_quarantine import FILE, FILE_SHA, PART, quarantined_sheet_rows

VERSION = "che_c3_charge_v1"
OUT = ROOT / "dataset/mixed95/c3_che_20261026_v1"
SOURCE = Path("/Users/curtischan/Datasets/batterylife/MICH")
# Fixed from physical coverage, not selected by C3 prediction errors.
GRID = np.linspace(2.5, 4.25, 1001)
WINDOW = GRID[(GRID >= 3.6) & (GRID <= 4.0)]
TRAIN_CELLS = (
    "MICH_BLForm1_pouch_NMC_45C_0-100_1-1C_a.pkl",
    "MICH_MCForm22_pouch_NMC_25C_0-100_1-1C_b.pkl",
    "MICH_MCForm32_pouch_NMC_45C_0-100_1-1C_b.pkl",
)
VAL_CELL = "MICH_BLForm3_pouch_NMC_45C_0-100_1-1C_c.pkl"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def protect():
    audit = OUTPUT / "input_audit.json"
    previous = read(audit)
    expected = dict(previous["protected_input_sha256"])
    expected[str(audit)] = sha(audit)
    for path, value in expected.items():
        if sha(Path(path)) != value:
            raise ValueError(f"frozen input changed: {path}")
    return expected


def charge_increments(points, *, target=False):
    """First-passage Q(V), in recorded order; no extrapolation or gap filling.

    Columns: record ID, timestamp, V, I, step-local Q. Target IDs define acquisition
    order even when one Excel timestamp is rounded/mis-recorded by one second.
    Non-charging samples outside the selected voltage window do not invalidate
    a later, uninterrupted CC segment. Values are never invented.
    """
    a = np.asarray(points, dtype=float)
    if a.ndim != 2 or a.shape[1] != 5 or len(a) < 10 or not np.isfinite(a).all():
        raise ValueError("invalid or insufficient charging records")
    unique = {}
    for row in a:
        key = int(row[0])
        if key in unique and not np.array_equal(unique[key], row):
            raise ValueError("conflicting duplicate record ID")
        unique[key] = row
    a = np.asarray([unique[k] for k in sorted(unique)])
    dt = np.diff(a[:, 1])
    if target and dt.min(initial=0) < -2 / 86400 - 1e-9:
        raise ValueError("timestamp reversal exceeds verified two-second tolerance")
    if (np.diff(a[:, 4]) < -.002).any():
        raise ValueError("capacity counter reset in acquisition order")
    positive = a[:, 3] > 0
    if ((~positive) & (a[:, 2] >= WINDOW[0]) & (a[:, 2] <= WINDOW[-1])).any():
        raise ValueError("interrupted charge inside feature window")
    valid = a[positive]
    if len(valid) < 10:
        raise ValueError("insufficient positive-current records")
    # Keep first passage through voltage levels, not a voltage-sorted trace.
    indices = [0]
    for j in range(1, len(valid)):
        if valid[j, 2] > valid[indices[-1], 2]:
            indices.append(j)
    trace = valid[indices]
    if trace[0, 2] > WINDOW[0] or trace[-1, 2] < WINDOW[-1]:
        raise ValueError("feature voltage window not covered")
    lo = max(0, int(np.searchsorted(trace[:, 2], WINDOW[0])) - 1)
    hi = min(len(trace) - 1, int(np.searchsorted(trace[:, 2], WINDOW[-1])))
    window_trace = trace[lo:hi + 1]
    if np.diff(window_trace[:, 2]).max(initial=0) > .02:
        raise ValueError("voltage sampling gap exceeds 20mV")
    dq = np.diff(np.interp(WINDOW, trace[:, 2], trace[:, 4]))
    if (dq < -1e-9).any() or dq.sum() <= 0:
        raise ValueError("nonphysical partial charging increments")
    stat = dict(records=len(a), duplicates_removed=len(points)-len(a),
                timestamp_reversals=int((dt < -1e-9).sum()),
                min_dt_seconds=float(dt.min(initial=0) * (86400 if target else 1)),
                noncharging_records_outside_window=int((~positive).sum()),
                charge_current_median_A=float(np.median(window_trace[:, 3])),
                voltage_min=float(trace[0, 2]), voltage_max=float(trace[-1, 2]),
                maximum_voltage_gap_V=float(np.diff(window_trace[:, 2]).max(initial=0)),
                partial_capacity_Ah=float(dq.sum()),
                step_capacity_min_Ah=float(a[:, 4].min()),
                step_capacity_max_Ah=float(a[:, 4].max()))
    return dq, stat


def target_features(out=OUT, raw=RAW):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "target_features_audit.json").exists():
        raise FileExistsError("target feature artifacts already complete; choose another output directory")
    protected = protect()
    audit = read(OUTPUT / "input_audit.json")
    mapping = {(r["source_file"], r["local_cycle"]): r["physical_cycle"]
               for r in audit["mapping"]}
    paths = sorted((raw / "Cell03").glob("*.xlsx"))
    hashes = {str(p): sha(p) for p in paths}
    prior = ROOT / "dataset/mixed95/c3_reconstruction_20261020_v2/raw_integrity.json"
    integrity = read(prior)
    if {r["file"]: r["sha256"] for r in integrity["files"]} != hashes:
        raise ValueError("original integrity scan no longer matches raw files")
    bad = [r for r in integrity["files"] if not r["all_parts_pass_crc"]]
    if len(bad) != 1 or Path(bad[0]["file"]).name != FILE or bad[0]["sha256"] != FILE_SHA:
        raise ValueError("unexpected raw integrity failure")
    if [r["part"] for r in bad[0]["bad_parts"]] != [PART]:
        raise ValueError("unexpected corrupt worksheet")
    cache = out / "target_charge_traces.npz"
    cache_audit = out / "target_trace_audit.json"
    if cache.exists():
        meta = read(cache_audit)
        if meta["version"] != VERSION or meta["raw_sha256"] != hashes or meta["cache_sha256"] != sha(cache):
            raise ValueError("stale or changed raw trace cache")
        quarantine = meta["quarantine"]
        steps = {int(k):v for k,v in meta["step_summary"].items()}
        with np.load(cache, allow_pickle=False) as data:
            points = {n: data[f"cycle_{n}"] for n in range(1, 405)}
    else:
        points = defaultdict(list)
        steps, quarantine, parsed = {}, [], []
        for path in paths:
            base = path.name.replace("-2.xlsx", "-1.xlsx") if path.name.startswith(("C3-100-", "C3-200-", "C3-300-")) else path.name
            with XlsxRows(path) as book:
                for row in (rows_named(book, "工步数据表") if "工步数据表" in book.sheets else ()):
                    if _number(row.get("工步号")) == 10:
                        n = mapping.get((base, int(float(row["循环号"]))))
                        if n is not None:
                            value = float(row["容量(Ah)"])
                            if n in steps and steps[n] != value:
                                raise ValueError("conflicting charge summary")
                            steps[n] = value
                kept = 0
                for sheet in book.sheets:
                    if not sheet.startswith("详细数据"):
                        continue
                    rows = iter(quarantined_sheet_rows(book, sheet, quarantine)
                                if path.name == FILE and book.sheets[sheet] == PART else book.rows(sheet))
                    columns = {str(v).strip():k for k,v in next(rows).items()}
                    for row in rows:
                        if _number(row.get(columns["工步号"])) != 10:
                            continue
                        n = mapping.get((base, int(float(row[columns["循环序号"]]))))
                        if n is not None:
                            points[n].append(tuple(float(row[columns[k]]) for k in
                                ("记录序号", "绝对时间", "电压(V)", "电流(A)", "容量(Ah)")))
                            kept += 1
                parsed.append(dict(file=path.name, charge_records=kept))
                print(json.dumps(parsed[-1]), flush=True)
        if set(points) != set(range(1, 405)) or set(steps) != set(points):
            raise ValueError("incomplete physical-cycle map")
        np.savez_compressed(cache, **{f"cycle_{n}": np.asarray(p) for n,p in points.items()})
        save(cache_audit, dict(version=VERSION, raw_sha256=hashes, parsed=parsed,
            step_summary=steps, quarantine=quarantine, cache_sha256=sha(cache)))
    features, quality, missing = [], {}, []
    for n in range(1, 405):
        try:
            dq, stat = charge_increments(points[n], target=True)
            if abs(stat["step_capacity_max_Ah"] - steps[n]) > .002 or stat["step_capacity_min_Ah"] > .05:
                raise ValueError("trace disagrees with full step summary")
            features.append(dq / 183.)
            quality[str(n)] = stat
        except ValueError as exc:
            missing.append(dict(cycle=n, reason=str(exc)))
            features.append(np.full(len(WINDOW)-1, np.nan))
    if hashes != {p:sha(Path(p)) for p in hashes} or protected != protect():
        raise ValueError("raw or frozen files changed while extracting")
    np.savez_compressed(out / "target_features.npz", cycles=np.arange(1,405),
                        delta_q=np.asarray(features), voltage_knots=WINDOW)
    save(out / "target_features_audit.json", dict(version=VERSION, raw_sha256=hashes,
        protected_sha256=protected, quality=quality, missing=missing, rpt=audit["rpt"],
        quarantine=quarantine, nominal_Ah=183., capacity_feature_divisor_Ah=183.,
        feature_step=10, voltage_window_V=[float(WINDOW[0]),float(WINDOW[-1])],
        ordering="record_id; timestamps checked but not used to swap acquired rows",
        no_daily_capacity_label=True, raw_files_modified=False,
        feature_sha256=sha(out / "target_features.npz")))
    print(json.dumps(dict(target_cycles=404, usable=404-len(missing), missing=missing)), flush=True)


def source_features(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "source_features_audit.json").exists():
        raise FileExistsError("source feature artifacts already complete; choose another output directory")
    records, arrays = [], {}
    for name in (*TRAIN_CELLS, VAL_CELL):
        path = SOURCE / name
        with path.open("rb") as handle:
            data = pickle.load(handle)
        nominal = float(data["nominal_capacity_in_Ah"])
        if data["SOC_interval"] != [0,1] or nominal != 2.36:
            raise ValueError("unexpected source capacity or SOC range")
        q, y, cycles, skipped = [], [], [], []
        for cycle in data["cycle_data"]:
            n = int(cycle["cycle_number"])
            i, v, qc, qd, time = (np.asarray(cycle[k], dtype=float) for k in
                ("current_in_A", "voltage_in_V", "charge_capacity_in_Ah", "discharge_capacity_in_Ah", "time_in_s"))
            try:
                if len({len(i),len(v),len(qc),len(qd),len(time)}) != 1:
                    raise ValueError("unequal trace lengths")
                dis = np.isfinite(qd) & np.isfinite(i) & (i < -.01*nominal)
                label = float(qd[dis].max()) / nominal if dis.any() else float("nan")
                if not np.isfinite(label) or not .5 <= label <= 1.2:
                    raise ValueError("invalid full-discharge capacity (outside 0.5..1.2 nominal)")
                positive = np.isfinite(i) & (i > .01*nominal)
                indices = np.flatnonzero(positive)
                if len(indices) < 10:
                    raise ValueError("insufficient charging trace")
                # Counter reset separates independent charging events; never join them.
                breaks = np.flatnonzero((np.diff(indices)>1) | (np.diff(qc[indices])<-.002))+1
                candidates=[]
                for group in np.split(indices, breaks):
                    if len(group)<10:continue
                    try:
                        values, stat=charge_increments(np.c_[group,time[group],v[group],i[group],qc[group]])
                    except ValueError:
                        continue
                    candidates.append((values,stat))
                if len(candidates) != 1:
                    raise ValueError(f"expected one covering charging event, found {len(candidates)}")
                q.append(candidates[0][0]/nominal);y.append(label);cycles.append(n)
            except ValueError as exc:
                skipped.append(dict(cycle=n, reason=str(exc)))
        key = path.stem
        arrays[key+"_q"] = np.asarray(q)
        arrays[key+"_y"] = np.asarray(y)
        arrays[key+"_cycles"] = np.asarray(cycles)
        records.append(dict(cell=name, split="val" if name==VAL_CELL else "train",
                            source_path=str(path), sha256=sha(path), nominal_Ah=nominal,
                            n_total=len(data["cycle_data"]), n_valid=len(y), skipped=skipped,
                            soh_range=[float(min(y)),float(max(y))] if y else None))
        print(json.dumps({k:v for k,v in records[-1].items() if k!="skipped"}),flush=True)
        if len(y)<150 or np.ptp(y)<.02:
            raise ValueError("source cell has insufficient valid capacity coverage")
    np.savez_compressed(out / "source_features.npz", **arrays)
    save(out / "source_features_audit.json", dict(version=VERSION, cells=records,
        nominal_normalization=True, voltage_knots=WINDOW.tolist(),
        selection="three fixed existing MICH NMC111 cells, one separate source validation cell; no C3 score selection",
        feature_sha256=sha(out / "source_features.npz")))


def fit_indicators(q):
    q=np.asarray(q,dtype=float)
    if not np.isfinite(q).all() or (q<0).any():raise ValueError("invalid training Q increments")
    center=q.mean(axis=0)
    _,_,vt=np.linalg.svd(q-center,full_matrices=False)
    loading=vt[0]
    if loading[np.argmax(np.abs(loading))]<0:loading=-loading
    state=dict(q_mean=center.tolist(),pca_loading=loading.tolist())
    x=transform_indicators(q,state,standardize=False)
    scale=x.std(axis=0);scale[scale<1e-12]=1
    state.update(hi_mean=x.mean(axis=0).tolist(),hi_scale=scale.tolist())
    return state


def transform_indicators(q,state,standardize=True):
    q=np.asarray(q,dtype=float)
    p=q/q.sum(axis=1,keepdims=True)
    entropy=-(p*np.log(np.maximum(p,np.finfo(float).tiny))).sum(axis=1)
    x=np.c_[q.std(axis=1),entropy,(q-state['q_mean'])@np.asarray(state['pca_loading'])]
    return (x-state['hi_mean'])/state['hi_scale'] if standardize else x


def sequences(x,cycles,length):
    """Causal history; repeat first feature at start, never bridge missing cycles."""
    x=np.asarray(x,dtype=float);cycles=np.asarray(cycles,dtype=int)
    if not np.isfinite(x).all() or len(set(cycles))!=len(cycles) or (np.diff(cycles)<=0).any():
        raise ValueError("invalid feature sequence")
    result=[];start=0
    for j in range(len(x)):
        if j and cycles[j]!=cycles[j-1]+1:start=j
        rows=x[max(start,j-length+1):j+1]
        result.append(np.r_[np.repeat(rows[:1],length-len(rows),axis=0),rows])
    return np.asarray(result,dtype=np.float32)
