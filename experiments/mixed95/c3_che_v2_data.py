"""Fixed, auditable Q-V candidate search for Che III-A C3 reconstruction v2.

No target capacities enter extraction or PCA. Selection accepts only the
explicitly supplied calibration labels, never an outer fold's hidden labels.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from experiments.mixed95.c3_che_data import (
    OUT as V1, ROOT, SOURCE, TRAIN_CELLS, VAL_CELL, GRID, read, save, protect,
)
from experiments.mixed95.c3_evaluate import sha

DEFAULT_SUITE = "c3_che_20261026_v2"
BASE = ROOT / "dataset/mixed95"
SEEDS = (2021, 42, 2024)
GROUPS = ((1,), (101,), (201,), (301,), (401, 402))
KINDS = ("total", "mean", "std", "q25", "q50", "q75", "iqr", "entropy", "pc1")
PROTOCOL = dict(version=2, seeds=list(SEEDS), window_start_V=3.2,
    window_end_V=4., endpoint_step_V=.1, widths_V=[.2, .3, .4],
    grid_full_range_V=[2.5, 4.25], grid_segments=1000,
    candidate_kinds=list(KINDS), minimum_source_cycles=150,
    minimum_abs_pearson=.8, redundancy_above=.98, maximum_features=3,
    grouped_RPT_cycles=[list(g) for g in GROUPS],
    source_cells=list(TRAIN_CELLS), validation_cell=VAL_CELL,
    hidden_sizes=[8, 16, 32], history_lengths=[1, 5, 10, 20],
    optimizer="Adam", learning_rate=.001, batch_size=64, max_epochs=3000,
    patience=100, min_delta=1e-8, gradient_clip_norm=5.,
    source_precision="float32", calibration_precision="float64",
    dense_solver="minimum source-weight update, unregularized SVD least squares",
    svd_rcond=1e-12, max_normal_equation_residual=1e-8,
    max_reload_soh_delta=1e-8, device="cpu", threads=2,
    target_nominal_Ah=183., target_label_div094=False,
    target_input_columns="Q-V segment capacities only; no cycle/current/RPT input",
    feature_screening_is_task_adaptation=True, strict_paper_replication=False,
    retrospective=True, interpolation_correction=False, monotonic_projection=False)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def windows():
    return [(lo / 10, (lo + width) / 10)
            for lo in range(32, 39) for width in (2, 3, 4) if lo + width <= 40]


def prepare_trace(points, target=False):
    a = np.asarray(points, dtype=float)
    if a.ndim != 2 or a.shape[1] != 5 or len(a) < 10 or not np.isfinite(a).all():
        raise ValueError("invalid charging records")
    unique = {}
    for row in a:
        key = int(row[0])
        if key in unique and not np.array_equal(unique[key], row):
            raise ValueError("conflicting duplicate record ID")
        unique[key] = row
    a = np.asarray([unique[k] for k in sorted(unique)])
    if target and np.diff(a[:, 1]).min(initial=0) < -2 / 86400 - 1e-9:
        raise ValueError("unverified timestamp reversal")
    if (np.diff(a[:, 4]) < -.002).any():
        raise ValueError("capacity counter reset")
    positive = a[a[:, 3] > 0]
    if len(positive) < 10:
        raise ValueError("insufficient positive current samples")
    # Strictly increasing FIRST passage, never voltage-sort the acquisition.
    keep = np.r_[True, positive[1:, 2] > np.maximum.accumulate(positive[:, 2])[:-1]]
    return a, positive[keep]


def window_q(prepared, lo, hi):
    a, trace = prepared
    knots = GRID[(GRID >= lo - 1e-12) & (GRID <= hi + 1e-12)]
    if len(knots) < 3 or trace[0, 2] > knots[0] or trace[-1, 2] < knots[-1]:
        raise ValueError("window not covered")
    if ((a[:, 3] <= 0) & (a[:, 2] >= knots[0]) & (a[:, 2] <= knots[-1])).any():
        raise ValueError("charge interruption inside window")
    start = max(0, int(np.searchsorted(trace[:, 2], knots[0])) - 1)
    end = min(len(trace) - 1, int(np.searchsorted(trace[:, 2], knots[-1])))
    if np.diff(trace[start:end+1, 2]).max(initial=0) > .02:
        raise ValueError("voltage gap above 20 mV")
    q = np.diff(np.interp(knots, trace[:, 2], trace[:, 4]))
    if (q < -1e-9).any() or q.sum() <= 0:
        raise ValueError("nonphysical capacity increments")
    return q


def scalar_features(q, center, loading):
    q = np.asarray(q, dtype=float)
    p = q / q.sum(axis=1, keepdims=True)
    quantiles = np.quantile(q, [.25, .5, .75], axis=1).T
    return np.c_[q.sum(axis=1), q.mean(axis=1), q.std(axis=1), quantiles,
        quantiles[:, 2] - quantiles[:, 0],
        -(p * np.log(np.maximum(p, np.finfo(float).tiny))).sum(axis=1),
        (q - center) @ loading]


def source_events(data):
    nominal = float(data["nominal_capacity_in_Ah"])
    if data["SOC_interval"] != [0, 1] or nominal != 2.36:
        raise ValueError("source nominal capacity/SOC differs from audited experiment")
    rows = []
    for cycle in data["cycle_data"]:
        n = int(cycle["cycle_number"])
        i, v, qc, qd, time = [np.asarray(cycle[k], dtype=float) for k in
            ("current_in_A", "voltage_in_V", "charge_capacity_in_Ah",
             "discharge_capacity_in_Ah", "time_in_s")]
        try:
            if len({len(i), len(v), len(qc), len(qd), len(time)}) != 1:
                raise ValueError("unequal trace lengths")
            dis = np.isfinite(qd) & np.isfinite(i) & (i < -.01 * nominal)
            y = float(qd[dis].max()) / nominal if dis.any() else float("nan")
            if not np.isfinite(y) or not .5 <= y <= 1.2:
                raise ValueError("invalid full-discharge capacity")
            idx = np.flatnonzero(np.isfinite(i) & (i > .01 * nominal))
            if len(idx) < 10:
                raise ValueError("insufficient charging records")
            breaks = np.flatnonzero((np.diff(idx) > 1) | (np.diff(qc[idx]) < -.002)) + 1
            events = []
            for group in np.split(idx, breaks):
                try:
                    events.append(prepare_trace(np.c_[group, time[group], v[group], i[group], qc[group]]))
                except ValueError:
                    pass
            rows.append(dict(cycle=n, soh=y, events=events, error=None))
        except ValueError as exc:
            rows.append(dict(cycle=n, soh=None, events=[], error=str(exc)))
    if len({r['cycle'] for r in rows}) != len(rows):
        raise ValueError("duplicate source cycle")
    return sorted(rows, key=lambda r:r['cycle'])


def extract_candidates(out):
    out = Path(out)
    if (out / "candidates.json").exists():
        return load_candidates(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        if read(out / "protocol.json") != PROTOCOL:
            raise ValueError("protocol changed; choose a new suite-id")
    else:
        save(out / "protocol.json", PROTOCOL)
    protected = protect()
    old_manifest = read(V1 / "run_manifest.json")
    immutable = {str(ROOT/p): h for group in ('code_sha256', 'artifact_sha256')
                 for p,h in old_manifest[group].items()}
    immutable[str(V1 / "run_manifest.json")] = sha(V1 / "run_manifest.json")
    trace_audit = read(V1 / "target_trace_audit.json")
    target_audit = read(V1 / "target_features_audit.json")
    if sha(V1 / 'target_charge_traces.npz') != trace_audit['cache_sha256']:
        raise ValueError("target raw trace cache changed")
    immutable.update(trace_audit['raw_sha256'])
    immutable.update(protected)
    raw_target = np.load(V1 / 'target_charge_traces.npz', allow_pickle=False)
    target = [prepare_trace(raw_target[f'cycle_{n}'], target=True) for n in range(1,405)]
    raw_target.close()
    for n,(a,_) in enumerate(target,1):
        if abs(float(a[:,4].max())-trace_audit['step_summary'][str(n)]) > .002 or a[:,4].min() > .05:
            raise ValueError("target detail/summary capacity mismatch")
    source = []
    for name in (*TRAIN_CELLS, VAL_CELL):
        path = SOURCE / name
        immutable[str(path)] = sha(path)
        with path.open('rb') as stream:
            source.append(source_events(pickle.load(stream)))
        print(json.dumps(dict(stage='extract_source', cell=name, cycles=len(source[-1]))), flush=True)
    arrays = {'target_cycles': np.arange(1,405)}
    for c,rows in enumerate(source):
        arrays[f'c{c}_cycles'] = np.array([r['cycle'] for r in rows])
        arrays[f'c{c}_soh'] = np.array([r['soh'] if r['soh'] is not None else np.nan for r in rows])
    matrices = [[] for _ in source]; target_matrix = []; descriptors = []; audit = []; transforms = {}
    for lo,hi in windows():
        name = f'{lo:.1f}-{hi:.1f}V'
        item = dict(window=name, requested_V=[lo,hi], eligible=False)
        knots = GRID[(GRID >= lo-1e-12) & (GRID <= hi+1e-12)]
        item['actual_knots_V'] = knots.tolist()
        tq = []; target_rejected=[]
        for n,trace in enumerate(target,1):
            try:
                tq.append(window_q(trace,lo,hi)/183.)
            except ValueError as exc:
                target_rejected.append(dict(cycle=n, reason=str(exc)))
        item['target_rejected'] = target_rejected
        if target_rejected:
            item['reason'] = 'not all 404 target cycles have an uninterrupted covered window'
            audit.append(item); continue
        sq=[]; cell_audit=[]
        for c,rows in enumerate(source):
            values=np.full((len(rows),len(knots)-1),np.nan); rejected=[]
            for j,row in enumerate(rows):
                candidates=[]
                for event in row['events']:
                    try:candidates.append(window_q(event,lo,hi)/2.36)
                    except ValueError:pass
                if len(candidates)==1 and row['soh'] is not None:
                    values[j]=candidates[0]
                else:
                    rejected.append(dict(cycle=row['cycle'],reason=row['error'] or
                                         f'covering charge events={len(candidates)}'))
            valid=np.isfinite(values).all(axis=1)
            cell_audit.append(dict(cell=(*TRAIN_CELLS,VAL_CELL)[c],valid=int(valid.sum()),rejected=rejected))
            sq.append(values)
        item['source_cells']=cell_audit
        if any(r['valid'] < PROTOCOL['minimum_source_cycles'] for r in cell_audit):
            item['reason']='source/validation cell has fewer than 150 valid cycles'
            audit.append(item);continue
        # Each window's PCA is trained exclusively on the three source training cells.
        training=np.concatenate([q[np.isfinite(q).all(axis=1)] for q in sq[:3]])
        center=training.mean(axis=0)
        _,_,vt=np.linalg.svd(training-center,full_matrices=False)
        loading=vt[0]
        if loading[np.argmax(np.abs(loading))]<0:loading=-loading
        transforms[name]=dict(center=center.tolist(),loading=loading.tolist(),
            fitted_cells=list(TRAIN_CELLS),n_training_cycles=len(training))
        for c,q in enumerate(sq):
            valid=np.isfinite(q).all(axis=1);x=np.full((len(q),len(KINDS)),np.nan)
            x[valid]=scalar_features(q[valid],center,loading);matrices[c].append(x)
        target_matrix.append(scalar_features(np.asarray(tq),center,loading))
        descriptors.extend(dict(id=f'{name}:{kind}',window=name,kind=kind) for kind in KINDS)
        item.update(eligible=True,reason='covered target and adequate source cycles')
        audit.append(item)
        print(json.dumps(dict(stage='window',window=name,valid_counts=[r['valid'] for r in cell_audit])),flush=True)
    for c,parts in enumerate(matrices):
        arrays[f'c{c}_x']=np.concatenate(parts,axis=1) if parts else np.empty((len(source[c]),0))
    arrays['target_x']=np.concatenate(target_matrix,axis=1) if target_matrix else np.empty((404,0))
    arrays['target_current_A']=np.array([target_audit['quality'][str(n)]['charge_current_median_A'] for n in range(1,405)])
    for path,expected in immutable.items():
        if sha(Path(path)) != expected:raise ValueError(f'immutable file changed: {path}')
    np.savez_compressed(out/'candidates.npz',**arrays)
    metadata=dict(protocol=PROTOCOL,descriptors=descriptors,windows=audit,
        pca=transforms,rpt=target_audit['rpt'],immutable_sha256=immutable,
        candidates_sha256=sha(out/'candidates.npz'),
        current_is_diagnostic_not_model_input=True,target_labels_used_for_extraction=False)
    save(out/'candidates.json',metadata)
    return arrays,metadata


def load_candidates(out):
    meta=read(Path(out)/'candidates.json')
    if meta['protocol']!=PROTOCOL or sha(Path(out)/'candidates.npz')!=meta['candidates_sha256']:
        raise ValueError('candidate protocol/hash mismatch')
    with np.load(Path(out)/'candidates.npz',allow_pickle=False) as d:
        arrays={k:d[k].copy() for k in d.files}
    return arrays,meta


def pearson(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    valid=np.isfinite(a)&np.isfinite(b)
    if valid.sum()<3 or np.ptp(a[valid])<1e-14 or np.ptp(b[valid])<1e-14:return None
    return float(np.corrcoef(a[valid],b[valid])[0,1])


def grouped_values(target_x,calibration):
    """Only supplied labels are readable; 401/402 have one equal-weight group."""
    labels={int(r['cycle']):float(r['soh']) for r in calibration}
    xs=[];ys=[]
    for group in GROUPS:
        present=[n for n in group if n in labels]
        if present and len(present)!=len(group):raise ValueError('401/402 must be kept or hidden together')
        if present:
            xs.append(np.mean(target_x[np.array(present)-1],axis=0))
            ys.append(np.mean([labels[n] for n in present]))
    if len(xs)<3:raise ValueError('insufficient calibration groups for feature screening')
    return np.asarray(xs),np.asarray(ys)


def select_features(arrays,meta,calibration):
    gx,gy=grouped_values(arrays['target_x'],calibration)
    audit=[];eligible=[]
    for j,desc in enumerate(meta['descriptors']):
        source_r=[pearson(arrays[f'c{c}_x'][:,j],arrays[f'c{c}_soh']) for c in range(3)]
        target_r=pearson(gx[:,j],gy);rs=source_r+[target_r]
        reason=None
        if any(r is None for r in rs):reason='undefined correlation'
        elif min(abs(r) for r in rs)<PROTOCOL['minimum_abs_pearson']:reason='absolute correlation below 0.8'
        elif len({int(np.sign(r)) for r in rs})!=1:reason='source/target correlation direction differs'
        score=min(abs(r) for r in rs) if all(r is not None for r in rs) else None
        item=dict(**desc,index=j,source_pearson=source_r,target_grouped_pearson=target_r,
                  score=score,selected=False,reason=reason)
        # Current is recorded for confounding diagnostics only, never for ranking.
        item['target_current_pearson_diagnostic']=pearson(arrays['target_x'][:,j],arrays['target_current_A'])
        audit.append(item)
        if reason is None:eligible.append(item)
    selected=[]
    for item in sorted(eligible,key=lambda r:(-r['score'],r['id'])):
        j=item['index'];redundant=[]
        for other in selected:
            k=other['index']
            # Correlation within each source cell and target calibration groups;
            # never pool cell offsets, and never use target held-out curves here.
            rs=[pearson(arrays[f'c{c}_x'][:,j],arrays[f'c{c}_x'][:,k]) for c in range(3)]
            rs.append(pearson(gx[:,j],gx[:,k]))
            if any(r is not None and abs(r)>PROTOCOL['redundancy_above'] for r in rs):
                redundant.append(other['id'])
        if redundant:item['reason']='redundant above 0.98 with '+', '.join(redundant)
        elif len(selected)>=PROTOCOL['maximum_features']:item['reason']='maximum three features already selected'
        else:item.update(selected=True,reason='selected');selected.append(item)
    ids=[r['index'] for r in selected]
    return dict(status='selected' if ids else 'stopped_no_qualified_features',
        selected_indices=ids,selected_ids=[r['id'] for r in selected],
        calibration_cycles=sorted(int(r['cycle']) for r in calibration),
        calibration_label_sha256=fingerprint(calibration),candidate_audit=audit,
        source_validation_labels_used=False,hidden_labels_used=False,
        selection_sha256=fingerprint(dict(indices=ids,labels=calibration)))


def selected_arrays(arrays,selection):
    indices=selection['selected_indices']
    if not indices:raise ValueError('no qualified features; training prohibited')
    cells=[]
    for c in range(4):
        x=arrays[f'c{c}_x'][:,indices];y=arrays[f'c{c}_soh'];cycles=arrays[f'c{c}_cycles']
        valid=np.isfinite(x).all(axis=1)&np.isfinite(y)
        if valid.sum()<150:raise ValueError('intersection of selected windows leaves fewer than 150 source cycles')
        cells.append((x[valid],y[valid],cycles[valid]))
    train=np.concatenate([row[0] for row in cells[:3]])
    mean=train.mean(axis=0);scale=train.std(axis=0)
    if (scale<1e-12).any():raise ValueError('constant selected feature')
    state=dict(mean=mean.tolist(),scale=scale.tolist(),fitted_cells=list(TRAIN_CELLS),
               n_training_cycles=len(train),selected_ids=selection['selected_ids'])
    return [((x-mean)/scale,y,cycles) for x,y,cycles in cells],state
