"""Exploratory, retrospective C3 capacity reconstruction inspired by Che (2023).

This is NOT the paper's pretrained LSTM/SSSL reproduction and NOT a new test
truth for the frozen BatteryLife benchmark. Raw Excel and old results stay read-only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy.interpolate import PchipInterpolator

from experiments.comac95.data import XlsxRows, _number
from experiments.mixed95.c3_evaluate import ROOT, RAW, OUTPUT, rows_named, sha

OUT = ROOT / "dataset/mixed95/c3_reconstruction_20261020_v2"
# A priori common CC-charge window, avoiding turn-on and current transitions.
# This is a C3 adaptation, NOT Che's LFP 2.85--3.25 V window.
GRID = np.linspace(2.5, 4.25, 1001)
WINDOW = GRID[(GRID >= 3.4) & (GRID <= 4.0)]
RIDGE_ALPHA = 1.0  # fixed, no selection using held-out RPTs


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def workbook_integrity(path):
    """Read every ZIP member to EOF so CRC failures cannot be silently ignored."""
    bad = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            try:
                with archive.open(name) as stream:
                    while stream.read(1024*1024):
                        pass
            except (zipfile.BadZipFile, RuntimeError, EOFError) as exc:
                bad.append(dict(part=name, error=str(exc)))
    return dict(file=str(path),sha256=sha(path),bad_parts=bad,all_parts_pass_crc=not bad)


def check_raw(raw, out):
    paths=sorted((raw/'Cell03').glob('*.xlsx'))
    records=[]
    for path in paths:
        record=workbook_integrity(path);records.append(record)
        print(json.dumps(record,ensure_ascii=False),flush=True)
    save(out/'raw_integrity.json',dict(files=records,raw_files_modified=False,
         all_pass=all(r['all_parts_pass_crc'] for r in records)))
    return records


def partial_q(points, grid=WINDOW):
    """Q increments on first-passage voltage knots; no endpoint extrapolation.

    Points: record_id, absolute_time, voltage, current, step-local Ah.
    Repeated exports are deduplicated; conflicting record IDs are rejected.
    Only original step 10 CC-charge rows are passed here by the extractor.
    """
    unique = {}
    for p in points:
        key = int(p[0])
        if key in unique and tuple(p) != unique[key]:
            raise ValueError("conflicting duplicate charge record")
        unique[key] = tuple(p)
    ordered = sorted(unique.values(), key=lambda p: (p[1], p[0]))
    if len(ordered) < 10:
        raise ValueError("insufficient step10 charging records")
    a = np.asarray(ordered, dtype=np.float64)
    if not np.isfinite(a).all() or (a[:, 3] <= 0).any():
        raise ValueError("nonfinite or non-charging sample")
    if (np.diff(a[:, 4]) < -0.002).any():
        raise ValueError("step10 capacity counter reset or out-of-order")
    v, q = [], []
    for p in a:
        if not v or p[2] > v[-1]:
            v.append(p[2]); q.append(p[4])
    if v[0] > grid[0] or v[-1] < grid[-1]:
        raise ValueError("charging window not covered; cannot extrapolate")
    delta = np.diff(np.interp(grid, v, q))
    if (delta < -1e-9).any() or delta.sum() <= 0:
        raise ValueError("nonphysical partial capacity increments")
    return delta, dict(records=len(a), duplicates_removed=len(points)-len(a),
                       voltage_min=float(a[:, 2].min()), voltage_max=float(a[:, 2].max()),
                       current_median_A=float(np.median(a[:, 3])),
                       capacity_span_Ah=float(delta.sum()))


def indicators(q, fit_indices):
    """STD, Shannon entropy of positive normalized delta-Q, and first PCA score.

    The PCA mean/loading is fitted ONLY on fit_indices, including within CV.
    Entropy convention and population STD are explicit task choices because the
    paper does not fully specify their numerical implementation.
    """
    q = np.asarray(q, dtype=float)
    fit = np.asarray(fit_indices, dtype=int)
    if len(fit) < 2 or not np.isfinite(q).all() or (q < 0).any():
        raise ValueError("invalid PCA/HI input")
    mean = q[fit].mean(axis=0)
    _, singular, vectors = np.linalg.svd(q[fit]-mean, full_matrices=False)
    loading = vectors[0]
    if loading[np.argmax(np.abs(loading))] < 0:
        loading = -loading
    probs = q/q.sum(axis=1, keepdims=True)
    entropy = -(probs*np.log(np.maximum(probs, np.finfo(float).tiny))).sum(axis=1)
    x = np.c_[q.std(axis=1), entropy, (q-mean)@loading]
    return x, dict(mean=mean.tolist(), loading=loading.tolist(), singular_values=singular.tolist())


def estimate(q, anchor_indices, target_soh):
    """Fixed small-data ridge baseline; no cycle-number input, no target scaling."""
    x, pca = indicators(q, anchor_indices)
    idx = np.asarray(anchor_indices, dtype=int)
    y = np.asarray(target_soh, dtype=float)
    mu, sd = x[idx].mean(axis=0), x[idx].std(axis=0)
    sd[sd < 1e-12] = 1
    z = (x-mu)/sd
    center = y.mean()
    weights = np.linalg.solve(z[idx].T@z[idx]+RIDGE_ALPHA*np.eye(3),
                              z[idx].T@(y-center))
    prediction = center+z@weights
    return prediction, dict(pca=pca, hi_mean=mu.tolist(), hi_scale=sd.tolist(),
                           coefficients=weights.tolist(), intercept=float(center),
                           ridge_alpha=RIDGE_ALPHA, hi=x.tolist())


def calibrate(cycles, raw_prediction, anchor_cycles, anchor_soh):
    """Retrospective anchor residual correction; outside anchors remains missing."""
    cycles = np.asarray(cycles)
    anchor_cycles = np.asarray(anchor_cycles)
    anchor_rows = np.searchsorted(cycles, anchor_cycles)
    residual = np.asarray(anchor_soh)-raw_prediction[anchor_rows]
    corrected = raw_prediction+np.interp(cycles, anchor_cycles, residual)
    corrected = corrected.astype(float)
    corrected[(cycles < anchor_cycles[0]) | (cycles > anchor_cycles[-1])] = np.nan
    return corrected


def first_estimated_crossing(cycles, values, threshold=.95):
    valid = np.isfinite(values) & (np.asarray(values) <= threshold)
    indices = np.flatnonzero(valid)
    return int(np.asarray(cycles)[indices[0]]) if len(indices) else None


def extract(raw, out, *, quarantine_known_row=False):
    from experiments.mixed95.c3_quarantine import FILE,FILE_SHA,PART,quarantined_sheet_rows
    audit_path = OUTPUT/"input_audit.json"
    audit = json.loads(audit_path.read_text())
    mapping = {(r['source_file'], r['local_cycle']):r['physical_cycle'] for r in audit['mapping']}
    folder = raw/"Cell03"
    paths = sorted(folder.glob("*.xlsx"))
    hashes = {str(p):sha(p) for p in paths}
    integrity_path=out/'raw_integrity.json'
    checks=json.loads(integrity_path.read_text())['files'] if integrity_path.exists() else check_raw(raw,out)
    if {r['file']:r['sha256'] for r in checks} != hashes:
        raise ValueError('raw integrity scan is stale; choose a new output directory')
    failed=[r for r in checks if not r['all_parts_pass_crc']]
    if failed:
        approved=(quarantine_known_row and len(failed)==1 and
                  Path(failed[0]['file']).name==FILE and failed[0]['sha256']==FILE_SHA and
                  [p['part'] for p in failed[0]['bad_parts']]==[PART])
        if not approved:
            raise ValueError('raw CRC failure outside explicit --quarantine-known-row exception')
    protected = dict(audit['protected_input_sha256'])
    protected[str(audit_path)] = sha(audit_path)
    for p, expected in protected.items():
        if sha(Path(p)) != expected:
            raise ValueError(f"previous C3 input or benchmark changed: {p}")
    points = defaultdict(list)
    steps, parse_log, quarantine_log = {}, [], []
    for path in paths:
        # 100/200/300 -2 are detail continuations, NOT new physical cycles.
        base = path.name.replace('-2.xlsx', '-1.xlsx') if path.name.startswith(('C3-100-', 'C3-200-', 'C3-300-')) else path.name
        with XlsxRows(path) as book:
            # Continuation exports may contain only detail sheets.
            for row in (rows_named(book, '工步数据表') if '工步数据表' in book.sheets else ()):
                if _number(row.get('工步号')) != 10:
                    continue
                local = int(float(row['循环号']))
                n = mapping.get((base, local))
                if n is not None:
                    info = {k:float(row[k]) for k in ('容量(Ah)', '起始电流(A)', '结束电流(A)', '起始电压(V)', '终止电压(V)')}
                    if n in steps and steps[n] != info:
                        raise ValueError('conflicting step10 summary')
                    steps[n] = info
            seen = 0
            for sheet in book.sheets:
                if not sheet.startswith('详细数据'):
                    continue
                selected = (quarantined_sheet_rows(book,sheet,quarantine_log)
                            if failed and path.name==FILE and book.sheets[sheet]==PART else book.rows(sheet))
                rows = iter(selected); columns = {str(v).strip(): k for k,v in next(rows).items()}
                for row in rows:
                    if _number(row.get(columns['工步号'])) != 10:
                        continue
                    local = int(float(row[columns['循环序号']]))
                    n = mapping.get((base, local))
                    if n is None:
                        continue
                    point = tuple(float(row[columns[k]]) for k in
                                  ('记录序号','绝对时间','电压(V)','电流(A)','容量(Ah)'))
                    points[n].append(point); seen += 1
            parse_log.append(dict(file=path.name, step10_records=seen))
            print(json.dumps(parse_log[-1]), flush=True)
    cycles, increments, quality, missing = [], [], {}, []
    for n in range(1,405):
        try:
            q, stats = partial_q(points[n])
        except ValueError as exc:
            missing.append(dict(cycle=n, reason=str(exc)))
            continue
        cycles.append(n); increments.append(q); quality[str(n)] = stats
        # Independently reconcile the entire retained charge step against the
        # CRC-valid step summary, not just the partial interpolation window.
        observed=np.array(points[n],dtype=float)
        if abs(observed[:,4].max()-steps[n]['容量(Ah)'])>.002 or observed[:,4].min()>.05:
            raise ValueError(f'cycle {n}: step10 trace incomplete or inconsistent with step summary')
    if missing:
        save(out/'extraction_failures.json', missing)
        raise ValueError('some physical cycles lack usable charge window; no silent filling')
    assert cycles == list(range(1,405))
    assert hashes == {p:sha(Path(p)) for p in hashes}
    assert protected == {p:sha(Path(p)) for p in protected}
    if failed:
        assert len(quarantine_log)==1 and quarantine_log[0]['strict_remaining_xml_pass']
    np.savez_compressed(out/'charge_features.npz', cycles=np.array(cycles), delta_q=np.array(increments), voltage_knots=WINDOW)
    save(out/'extraction_audit.json', dict(raw_sha256=hashes, protected_sha256=protected,
        cycle_quality=quality, step10_summary=steps, parse_log=parse_log,
        source_input_audit_sha256=sha(audit_path), mapped_cycles=404, feature_step=10,
        quarantine=quarantine_log,raw_crc_failures_retained=failed,
        full_charge_step_reconciled_to_summary=True,
        voltage_window_V=[float(WINDOW[0]),float(WINDOW[-1])], voltage_full_grid=[2.5,4.25,1000],
        retrospective=True, future_RPT_allowed_for_reconstruction=True, benchmark_labels_modified=False))


def reconstruct(out):
    data = np.load(out/'charge_features.npz', allow_pickle=False)
    cycles, q = data['cycles'], data['delta_q']
    audit = json.loads((OUTPUT/'input_audit.json').read_text())
    rpt = audit['rpt']
    ac = np.array([r['cycle'] for r in rpt]); ay = np.array([r['soh'] for r in rpt])
    ai = np.searchsorted(cycles, ac)
    raw, model = estimate(q, ai, ay)
    # Independent reload path uses only the serialized numerical parameters.
    pca = model['pca']
    probabilities = q/q.sum(axis=1, keepdims=True)
    entropy_reload = np.array([-math.fsum(float(p)*math.log(float(p)) for p in row if p>0)
                               for row in probabilities])
    reloaded_hi = np.c_[q.std(axis=1),entropy_reload,
                       (q-np.asarray(pca['mean']))@np.asarray(pca['loading'])]
    reloaded = model['intercept']+((reloaded_hi-model['hi_mean'])/model['hi_scale'])@np.array(model['coefficients'])
    reload_max_delta = float(np.max(np.abs(reloaded-raw)))
    np.testing.assert_allclose(reloaded,raw,atol=1e-10,rtol=0)
    anchored = calibrate(cycles, raw, ac, ay)
    pchip = PchipInterpolator(ac, ay, extrapolate=False)(cycles)
    linear = np.interp(cycles, ac, ay); linear[cycles>ac[-1]] = np.nan
    # Treat adjacent final RPT 401/402 as one group: holding just one out would
    # make an almost-adjacent known target inflate apparent validation accuracy.
    groups = [[1],[101],[201],[301],[401,402]]
    cv = []
    for group in groups:
        held = np.isin(ac, group); train = ~held
        fit, _ = estimate(q, ai[train], ay[train])
        corrected = calibrate(cycles, fit, ac[train], ay[train])
        base = PchipInterpolator(ac[train], ay[train], extrapolate=False)(ac[held])
        for j, n in enumerate(ac[held]):
            at = int(np.searchsorted(cycles,n))
            true = float(ay[held][j])
            cv.append(dict(cycle=int(n), held_group=group, training_rpt_cycles=ac[train].tolist(),
                measured_soh=true, hi_prediction=float(fit[at]), hi_error_pp=float(100*(fit[at]-true)),
                corrected_hi_prediction=float(corrected[at]) if np.isfinite(corrected[at]) else None,
                pchip_prediction=float(base[j]) if np.isfinite(base[j]) else None,
                interior=bool(ac[train].min()<n<ac[train].max())))
    def metrics(rows, field):
        valid = [r for r in rows if r.get(field) is not None]
        error = [100*(r[field]-r['measured_soh']) for r in valid]
        return dict(n=len(valid), mae_pp=math.fsum(abs(v) for v in error)/len(valid),
                    rmse_pp=math.sqrt(math.fsum(v*v for v in error)/len(valid)))
    cv_metrics = {name:metrics(cv,field) for name,field in
                  [('hi_all_checkpoints','hi_prediction'),('anchored_hi_interior','corrected_hi_prediction'),
                   ('pchip_interior','pchip_prediction')]}
    settings = []
    for i, r in enumerate(rpt):
        stop = rpt[i+1]['cycle'] if i+1 < len(rpt) else 405
        ordinary = [s for s in audit['summary_rows'] if r['cycle'] < s['cycle'] < stop]
        settings.append(dict(last_rpt_cycle=r['cycle'], rpt_capacity_Ah=r['discharge_ah'],
            expected_97pct_charge_Ah=.97*r['discharge_ah'],
            observed_daily_charge_setpoints_Ah=sorted({s['charge_ah'] for s in ordinary}),
            warning='Daily charge total is programmed from RPT; it is not an independent SOH measurement.'))
    rows = []
    rpt_by_cycle = {r['cycle']:r['soh'] for r in rpt}
    for i,n in enumerate(cycles):
        rows.append(dict(physical_cycle=int(n), measured_rpt_soh=rpt_by_cycle.get(int(n)),
            hi_raw_soh=float(raw[i]),
            reconstructed_pseudo_soh=float(anchored[i]) if np.isfinite(anchored[i]) else None,
            rpt_pchip_reference_soh=float(pchip[i]) if np.isfinite(pchip[i]) else None,
            rpt_linear_reference_soh=float(linear[i]) if np.isfinite(linear[i]) else None,
            source='RPT_measured' if int(n) in rpt_by_cycle else ('pseudo_reconstruction' if n<=402 else 'outside_RPT_support_no_accepted_estimate')))
    # Exact measured labels and original benchmark stay intact.
    np.testing.assert_allclose(anchored[ai], ay, atol=1e-12, rtol=0)
    assert np.isnan(anchored[-2:]).all()
    envelope_violations=[]
    for left,right,upper,lower in zip(ac[:-1],ac[1:],ay[:-1],ay[1:]):
        idx=np.flatnonzero((cycles>left)&(cycles<right))
        envelope_violations.extend(int(cycles[i]) for i in idx
                                   if anchored[i]>upper+1e-12 or anchored[i]<lower-1e-12)
    estimated95=first_estimated_crossing(cycles,anchored)
    quality=dict(rpt_endpoint_envelope_violation_cycles=envelope_violations,
        estimated95_within_original_monotone_bracket=bool(estimated95 is not None and 202<=estimated95<=301),
        sparse_interior_mae_better_than_pchip=bool(cv_metrics['anchored_hi_interior']['mae_pp']<cv_metrics['pchip_interior']['mae_pp']),
        same_cell_sparse_checks_not_independent_validation=True,
        not_approved_as_exact95_label=True)
    report = dict(method='Che-inspired HI ridge plus retrospective RPT residual calibration',
        strict_paper_reproduction=False, source_pretraining=False, lstm_self_training=False,
        retrospective_not_early_prediction=True, suitable_as_measured_benchmark_truth=False,
        observed_cycle_range=[1,404], accepted_reconstruction_range=[1,402],
        exact95_life_cycle=None, original_rpt95_bracket=[202,301],
        estimated95_crossing={k:first_estimated_crossing(cycles,v) for k,v in
            [('hi_raw',raw),('anchored_hi',anchored),('rpt_pchip_reference',pchip),('rpt_linear_reference',linear)]},
        validation=cv_metrics, held_out=cv, rpt=rpt, rows=rows, model=model,quality_diagnostics=quality,
        protocol_charge_settings=settings,
        warning='Only 6 RPTs / 5 time groups. Charging current/Ah settings depend on latest RPT. CV is same-cell retrospective, not independent-cell or forward validation. Calibration-point fit is not validation. No calibrated CI.',
        feature_file_sha256=sha(out/'charge_features.npz'),
        parameter_reload_independent_entropy_max_delta_soh=reload_max_delta,
        upward_changes_count=int(np.count_nonzero(np.diff(anchored[np.isfinite(anchored)])>0)),
        no_monotonic_projection_applied=True,
        paper_sha256={p.name:sha(p) for p in ROOT.glob('Che*.pdf')},
        implementation_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                              [Path(__file__),ROOT/'experiments/mixed95/test_c3_reconstruct.py',
                               ROOT/'experiments/mixed95/c3_quarantine.py',
                               ROOT/'experiments/mixed95/test_c3_quarantine.py']},
        extraction_audit_sha256=sha(out/'extraction_audit.json'),
        isolated_nonfeature_row=json.loads((out/'extraction_audit.json').read_text())['quarantine'])
    save(out/'reconstruction.json', report)
    plot(out, report)
    return report


def plot(out, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial Unicode MS','PingFang SC','DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':11})
    rows = report['rows']; x = [r['physical_cycle'] for r in rows]
    vals = lambda key: np.array([r[key] if r[key] is not None else np.nan for r in rows])*100
    fig, axes = plt.subplots(2,1,figsize=(11,8),layout='constrained')
    ax=axes[0]
    ax.plot(x,vals('hi_raw_soh'),color='#aab2bd',lw=1,label='HI 模型原始输出（诊断）')
    ax.plot(x,vals('reconstructed_pseudo_soh'),color='#2171b5',lw=1.3,label='RPT 校准后的逐圈估计（非实测）')
    ax.plot(x,vals('rpt_pchip_reference_soh'),color='#e6550d',ls='--',label='仅 RPT 插值（参考）')
    ax.scatter([r['cycle'] for r in report['rpt']],[100*r['soh'] for r in report['rpt']],color='black',zorder=5,label='完整 RPT 实测')
    ax.axhline(95,color='#c62828',ls=':',label='95% 阈值')
    ax.axvspan(202,301,color='#fdd835',alpha=.12,label='RPT 限定的阈值范围 202–301')
    ax.set(title='C3 容量衰减：事后重建，不能当作前 100 圈预测真值',ylabel='SOH（容量 / 183 Ah，%）',xlabel='物理循环号')
    ax.legend(fontsize=9,ncol=2);ax.grid(alpha=.2)
    ax=axes[1];cv=report['held_out']
    ax.bar([str(r['cycle']) for r in cv],[r['hi_error_pp'] for r in cv],color='#2171b5',label='留出 RPT：HI 模型估计误差')
    inside=[r for r in cv if r['interior']]
    ax.scatter([str(r['cycle']) for r in inside],[100*(r['corrected_hi_prediction']-r['measured_soh']) for r in inside],
               marker='D',color='#e6550d',zorder=5,label='HI + 剩余 RPT 校准后的误差')
    ax.axhline(0,color='black',lw=.8);ax.grid(axis='y',alpha=.2)
    ax.set(title='把检查点隐藏后再估计；401/402 同时留出',ylabel='SOH 估计 − RPT 实测（百分点）',xlabel='被留出的 RPT 圈号')
    ax.legend(fontsize=9)
    fig.savefig(out/'C3_reconstruction.png',dpi=170)
    plt.close(fig)


def rpt_reference(out):
    """A transparent interim interpolation, explicitly NOT HI/SSSL reconstruction."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if not font.exists():font=Path('/Library/Fonts/Arial Unicode.ttf')
    if font.exists():font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Arial Unicode MS','axes.unicode_minus':False,'font.size':11})
    audit_path=OUTPUT/'input_audit.json'
    audit=json.loads(audit_path.read_text());rpt=audit['rpt']
    assert all(sha(Path(p))==h for p,h in audit['protected_input_sha256'].items())
    x=np.array([r['cycle'] for r in rpt]);y=np.array([r['soh'] for r in rpt])
    cycles=np.arange(1,405)
    pchip=PchipInterpolator(x,y,extrapolate=False)(cycles)
    linear=np.interp(cycles,x,y);linear[cycles>x[-1]]=np.nan
    # Independent hand calculation for the pair bracketing 95%.
    continuous=201+(176.544-.95*183)/(176.544-172.871)*100
    lin_cross=first_estimated_crossing(cycles,linear)
    assert lin_cross==math.ceil(continuous)==275
    rows=[]
    for i,n in enumerate(cycles):
        rows.append(dict(physical_cycle=int(n),measured_rpt_soh=next((r['soh'] for r in rpt if r['cycle']==n),None),
            pchip_interpolated_soh=float(pchip[i]) if np.isfinite(pchip[i]) else None,
            linear_interpolated_soh=float(linear[i]) if np.isfinite(linear[i]) else None,
            interpolated_is_measured=False))
    result=dict(status='HI_RECONSTRUCTION_BLOCKED_RAW_CRC_FAILURE',method='RPT-only interpolation reference',
        paper_method_implemented=False,hi_model_trained=False,benchmark_labels_modified=False,
        retrospective_not_early_prediction=True,measured95_exact_cycle=None,
        original95_possible_integer_range_assuming_monotone=[202,301],
        interpolation_estimated95_cycles=dict(linear=lin_cross,pchip=first_estimated_crossing(cycles,pchip)),
        linear95_continuous_cycle=float(continuous),rpt=rpt,rows=rows,
        no_extrapolation_cycles=[403,404],
        source_audit_sha256=sha(audit_path),paper_sha256={p.name:sha(p) for p in ROOT.glob('Che*.pdf')},
        raw_integrity=json.loads((out/'raw_integrity.json').read_text()),
        limitation='Interpolation is not Che SSSL, not a validated pseudo-capacity model, not exact SOH or lifetime truth. It must not be used as independent benchmark evaluation truth.')
    save(out/'rpt_reference.json',result)
    fig,ax=plt.subplots(figsize=(10.5,5.8),layout='constrained')
    ax.plot(cycles,pchip*100,color='#2171b5',lw=1.8,label='PCHIP 插值参考（非实测）')
    ax.plot(cycles,linear*100,color='#777777',ls='--',lw=1,label='线性插值参考（非实测）')
    ax.scatter(x,y*100,color='black',s=38,zorder=4,label='完整 RPT 实测点')
    ax.axhline(95,color='#c62828',ls=':',label='95% SOH')
    ax.axvspan(202,301,color='#fdd835',alpha=.16,label='单调假设下真实阈值范围：202–301 圈')
    ax.scatter([275],[float(pchip[274]*100)],color='#c62828',marker='x',s=60,zorder=5)
    ax.annotate('约第 275 圈：插值估计，不是实测寿命',xy=(275,pchip[274]*100),xytext=(95,92),
                arrowprops=dict(arrowstyle='->',color='#555'),fontsize=11)
    ax.set(title='C3 容量衰减：先给出 RPT 插值参考\n逐圈健康指标重建因原始文件损坏尚未完成',
           xlabel='物理循环号',ylabel='SOH = 完整 RPT 放电容量 / 183 Ah（%）',xlim=(-3,415),ylim=(90.8,101.7))
    ax.grid(alpha=.2);ax.legend(fontsize=9,loc='upper right')
    fig.savefig(out/'C3_RPT_reference.png',dpi=180);plt.close(fig)
    print(json.dumps({k:result[k] for k in ['status','interpolation_estimated95_cycles','original95_possible_integer_range_assuming_monotone']},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['check-raw','extract','reconstruct','rpt-reference'])
    parser.add_argument('--raw',type=Path,default=RAW)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--quarantine-known-row',action='store_true',
                        help='explicitly isolate the fingerprinted step15 row; never other corruption')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    if args.stage=='check-raw':
        check_raw(args.raw,args.out)
    elif args.stage=='rpt-reference':
        rpt_reference(args.out)
    elif args.stage=='extract':
        if (args.out/'charge_features.npz').exists():raise FileExistsError('choose a fresh output directory')
        extract(args.raw,args.out,quarantine_known_row=args.quarantine_known_row)
    else:
        report=reconstruct(args.out)
        print(json.dumps({k:report[k] for k in ['estimated95_crossing','validation','warning']},ensure_ascii=False),flush=True)
    print('elapsed_seconds',time.monotonic()-started,flush=True)


if __name__=='__main__':
    main()
