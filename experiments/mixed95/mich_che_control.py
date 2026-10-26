"""One-cell, same-recorded-condition MICH control for Che III-A reconstruction.

Same v2 Q-V/HI -> source ReLU-LSTM+Dense -> Dense-only SVD route.
MCForm31/32/33 train, 34 validation, 35 sealed dense-test cell. Six
pseudo-checkpoints are visible; all other capacity values are read ONLY in
the evaluation stage. No C3 result, model or scaler is reused for training.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import pickle
import platform
import re
import subprocess
import sys
import time

import numpy as np
from scipy.interpolate import PchipInterpolator
import torch
from torch.nn import functional as F

from experiments.mixed95.c3_che_data import ROOT, OUT as C3_V1, read, save, sequences
from experiments.mixed95.c3_che_v2_data import (
    SOURCE, BASE, PROTOCOL as C3_PROTOCOL, KINDS, windows, window_q,
    source_events, scalar_features, pearson, fingerprint,
)
from experiments.mixed95.c3_che_v2_model import CapacityModel, svd_dense
from experiments.mixed95.c3_che_model import encoder_hash
from experiments.mixed95.c3_evaluate import sha

DEFAULT_SUITE='mich_che_control_20261026_v1'
CELLS=tuple(f'MICH_MCForm{n}_pouch_NMC_45C_0-100_1-1C_{s}.pkl'
            for n,s in ((31,'a'),(32,'b'),(33,'c'),(34,'d'),(35,'e')))
SEEDS=(2021,42,2024)
CONDITION_KEYS=('form_factor','anode_material','cathode_material',
    'nominal_capacity_in_Ah','SOC_interval','charge_protocol','discharge_protocol',
    'min_voltage_limit_in_V','max_voltage_limit_in_V')
PROTOCOL=copy.deepcopy(C3_PROTOCOL)
PROTOCOL.update(experiment='same recorded-condition MICH one-cell control',
    version=1,source_cells=list(CELLS[:3]),validation_cell=CELLS[3],test_cell=CELLS[4],
    recorded_temperature_C=45,formation_family='MCForm',target_nominal_Ah=2.36,
    target_eligibility='complete 3.0 V discharge and uninterrupted 3.6..4.0 V reference charge window',
    checkpoint_selection='valid-cycle positions 0,25%,50%,75%,penultimate,last; selected without capacity values',
    checkpoint_count=6,source_model_selection='three-seed mean source-validation MSE',
    evaluation='all eligible test cycles EXCEPT six visible checkpoints',
    cycle_axis='BatteryLife preprocessed cycle_number, not recovered raw physical cycles')
PROTOCOL.pop('grouped_RPT_cycles',None)
OUT=BASE/DEFAULT_SUITE


def execution_hashes():
    names=('mich_che_control.py','test_mich_che_control.py','c3_che_v2_data.py',
           'c3_che_v2_model.py','c3_che_data.py','c3_che_model.py')
    return {str(ROOT/'experiments/mixed95'/name):sha(ROOT/'experiments/mixed95'/name) for name in names}


def write_once(path,value):
    if path.exists():
        if read(path)!=value:raise ValueError(f'existing experiment record changed: {path}')
    else:save(path,value)


def verify_condition(metadata):
    if len(metadata)!=5:raise ValueError('exactly five distinct batteries required')
    if len({r['cell_id'] for r in metadata})!=5:raise ValueError('battery split overlaps')
    first={k:metadata[0][k] for k in CONDITION_KEYS}
    for cell in metadata:
        if {k:cell[k] for k in CONDITION_KEYS}!=first:
            raise ValueError('recorded chemistry/format/protocol differs across splits')
        if '_45C_0-100_1-1C_' not in cell['cell_id'] or not cell['cell_id'].startswith('MICH_MCForm'):
            raise ValueError('temperature/formation-family identifiers differ')
    if first['SOC_interval']!=[0,1] or first['nominal_capacity_in_Ah']!=2.36:
        raise ValueError('unexpected MICH protocol')
    return first


def checkpoint_indices(n):
    if n<150:raise ValueError('fewer than 150 complete input/label cycles')
    result=np.array([0,(n-1)//4,(n-1)//2,3*(n-1)//4,n-2,n-1])
    if len(set(result.tolist()))!=6:raise ValueError('checkpoint collision')
    return result


def protected_files():
    """Protect ALL frozen C3 v2 files, original v1 inputs and raw source pkls."""
    c3=BASE/'c3_che_20261026_v2';m=read(c3/'run_manifest.json')
    hashes={str(c3/p):h for p,h in m['artifact_sha256'].items()}
    hashes.update(read(c3/'candidates.json')['immutable_sha256'])
    for name in ('run_manifest.json','verification.json','regression_tests.txt'):
        hashes[str(c3/name)]=sha(c3/name)
    hashes[str(Path(m['report_path']))]=m['report_sha256']
    for path in (ROOT/'experiments/mixed95').glob('*c3_che*.py'):
        hashes[str(path)]=sha(path)
    return hashes


def check_hashes(hashes):
    for path,expected in hashes.items():
        if sha(Path(path))!=expected:raise ValueError(f'protected input/output changed: {path}')


def extract(out):
    out.mkdir(parents=True,exist_ok=True)
    write_once(out/'protocol.json',PROTOCOL)
    if (out/'features.json').exists():return
    hashes=protected_files();source=[];metadata=[];temperatures=[]
    target_discharge_complete={}
    for name in CELLS:
        path=SOURCE/name;hashes[str(path)]=sha(path)
        with path.open('rb') as stream:data=pickle.load(stream)
        metadata.append({k:v for k,v in data.items() if k!='cycle_data'})
        sample_temp=[]
        for row in data['cycle_data']:
            t=np.asarray(row.get('temperature_in_C',[]),float)
            if np.isfinite(t).any():sample_temp.append(float(np.nanmedian(t)))
            if name==CELLS[4]:
                current=np.asarray(row['current_in_A'],float);v=np.asarray(row['voltage_in_V'],float)
                discharge=np.flatnonzero(np.isfinite(v)&(current<-.0236))
                target_discharge_complete[int(row['cycle_number'])]=bool(len(discharge) and abs(v[discharge[-1]]-3.)<=.005)
        temperatures.append(dict(cell=name,median_recorded_temperature_C=float(np.median(sample_temp)) if sample_temp else None))
        source.append(source_events(data))
    condition=verify_condition(metadata)
    # Eligibility is frozen before seeing prediction errors; every exclusion is
    # saved. The reference window is fixed in PROTOCOL, not optimized for score.
    eligible=[];excluded=[]
    for j,row in enumerate(source[4]):
        reason=row['error'];q=[]
        for event in row['events']:
            try:q.append(window_q(event,3.6,4.))
            except ValueError:pass
        if not target_discharge_complete[row['cycle']]:reason='missing complete discharge ending at 3.0 V'
        elif len(q)!=1:reason=f'covering 3.6..4.0 V charging events={len(q)}'
        if reason is None:eligible.append(j)
        else:excluded.append(dict(cycle=row['cycle'],reason=reason))
    target=[source[4][j] for j in eligible]
    anchor_rows=checkpoint_indices(len(target))
    cycles=np.array([row['cycle'] for row in target]);truth=np.array([row['soh'] for row in target])
    calibration=[dict(row=int(j),cycle=int(cycles[j]),soh=float(truth[j])) for j in anchor_rows]
    arrays={'target_cycles':cycles};parts=[[] for _ in range(5)];descriptors=[];audit=[];pca={}
    for c,rows in enumerate(source[:4]):
        arrays[f'c{c}_cycles']=np.array([row['cycle'] for row in rows])
        arrays[f'c{c}_soh']=np.array([row['soh'] if row['soh'] is not None else np.nan for row in rows])
    for lo,hi in windows():
        window=f'{lo:.1f}-{hi:.1f}V';q_all=[];counts=[];rejections=[]
        knots=np.linspace(2.5,4.25,1001);knots=knots[(knots>=lo-1e-12)&(knots<=hi+1e-12)]
        for c,rows in enumerate([*source[:4],target]):
            values=np.full((len(rows),len(knots)-1),np.nan);bad=[]
            for j,row in enumerate(rows):
                found=[]
                for event in row['events']:
                    try:found.append(window_q(event,lo,hi)/2.36)
                    except ValueError:pass
                if len(found)==1 and row['error'] is None:values[j]=found[0]
                else:bad.append(int(row['cycle']))
            q_all.append(values);counts.append(int(np.isfinite(values).all(axis=1).sum()));rejections.append(bad)
        accepted=counts[4]==len(target) and min(counts[:4])>=150
        audit.append(dict(window=window,valid_cycles=counts,eligible=accepted,rejected_cycles=rejections))
        if not accepted:continue
        train=np.concatenate([q[np.isfinite(q).all(axis=1)] for q in q_all[:3]])
        center=train.mean(axis=0);_,_,vt=np.linalg.svd(train-center,full_matrices=False);loading=vt[0]
        if loading[np.argmax(np.abs(loading))]<0:loading=-loading
        pca[window]=dict(center=center.tolist(),loading=loading.tolist(),fitted_cells=list(CELLS[:3]))
        for c,q in enumerate(q_all):
            valid=np.isfinite(q).all(axis=1);features=np.full((len(q),len(KINDS)),np.nan)
            features[valid]=scalar_features(q[valid],center,loading);parts[c].append(features)
        descriptors.extend(dict(id=f'{window}:{kind}',window=window,kind=kind) for kind in KINDS)
    if not descriptors:raise ValueError('no common eligible Q-V windows; stop without changing protocol')
    for c in range(4):arrays[f'c{c}_x']=np.concatenate(parts[c],axis=1)
    arrays['target_x']=np.concatenate(parts[4],axis=1)
    np.savez_compressed(out/'features.npz',**arrays)
    # Separate truth file is deliberately not opened by screen/train/calibrate.
    np.savez_compressed(out/'test_truth.npz',cycles=cycles,soh=truth,
        original_cycles=np.array([row['cycle'] for row in source[4]]),
        original_soh=np.array([row['soh'] if row['soh'] is not None else np.nan for row in source[4]]))
    save(out/'calibration_points.json',dict(points=calibration,group_rows=[[int(j)] for j in anchor_rows[:4]]+
        [[int(j) for j in anchor_rows[4:]]],selection_uses_cycle_positions_not_capacity_values=True))
    check_hashes(hashes)
    save(out/'features.json',dict(protocol=PROTOCOL,condition=condition,cells_metadata=metadata,
        recorded_temperatures=temperatures,descriptors=descriptors,windows=audit,pca=pca,
        target_eligible_cycles=cycles.tolist(),target_excluded_cycles=excluded,
        target_total_preprocessed_cycles=len(source[4]),
        visible_checkpoints=6,hidden_capacity_count=len(target)-6,
        features_sha256=sha(out/'features.npz'),truth_sha256=sha(out/'test_truth.npz'),
        calibration_sha256=sha(out/'calibration_points.json'),protected_sha256=hashes))
    print(json.dumps(dict(stage='extracted',test_cell=CELLS[4],eligible=len(target),
        hidden=len(target)-6,anchors=[r['cycle'] for r in calibration],candidates=len(descriptors))),flush=True)


def features(out):
    meta=read(out/'features.json')
    if meta['protocol']!=PROTOCOL:raise ValueError('protocol changed; choose a new suite')
    for name,key in (('features.npz','features_sha256'),('calibration_points.json','calibration_sha256')):
        if sha(out/name)!=meta[key]:raise ValueError(f'input changed: {name}')
    with np.load(out/'features.npz',allow_pickle=False) as data:arrays={key:data[key].copy() for key in data.files}
    return arrays,meta,read(out/'calibration_points.json')


def screen(arrays,meta,calibration):
    rows=calibration['group_rows'];y_by_row={r['row']:r['soh'] for r in calibration['points']}
    gx=np.array([arrays['target_x'][group].mean(axis=0) for group in rows])
    gy=np.array([np.mean([y_by_row[j] for j in group]) for group in rows])
    candidates=[]
    for j,d in enumerate(meta['descriptors']):
        rs=[pearson(arrays[f'c{c}_x'][:,j],arrays[f'c{c}_soh']) for c in range(3)]
        rt=pearson(gx[:,j],gy);values=rs+[rt];reason=None
        if any(v is None for v in values):reason='undefined correlation'
        elif min(abs(v) for v in values)<.8:reason='absolute correlation below 0.8'
        elif len({int(np.sign(v)) for v in values})!=1:reason='source/calibration direction mismatch'
        candidates.append(dict(**d,index=j,source_pearson=rs,calibration_group_pearson=rt,
            score=min(abs(v) for v in values) if all(v is not None for v in values) else None,
            reason=reason,selected=False))
    selected=[]
    for row in sorted([r for r in candidates if r['reason'] is None],key=lambda r:(-r['score'],r['id'])):
        j=row['index'];redundant=False
        for other in selected:
            k=other['index'];corr=[pearson(arrays[f'c{c}_x'][:,j],arrays[f'c{c}_x'][:,k]) for c in range(3)]
            corr.append(pearson(gx[:,j],gx[:,k]))
            if any(v is not None and abs(v)>.98 for v in corr):redundant=True
        if redundant:row['reason']='redundant above 0.98'
        elif len(selected)>=3:row['reason']='three features already selected'
        else:row.update(selected=True,reason='selected');selected.append(row)
    return dict(status='selected' if selected else 'stopped_no_qualified_features',
        selected_indices=[r['index'] for r in selected],selected_ids=[r['id'] for r in selected],
        candidate_audit=candidates,calibration_sha256=fingerprint(calibration),
        dense_test_labels_used=False,selection_sha256=fingerprint([r['id'] for r in selected]))


def model_inputs(arrays,selection):
    cols=selection['selected_indices']
    if not cols:raise ValueError('no qualified features; training prohibited')
    cells=[]
    for c in range(4):
        x=arrays[f'c{c}_x'][:,cols];y=arrays[f'c{c}_soh'];n=arrays[f'c{c}_cycles']
        valid=np.isfinite(x).all(axis=1)&np.isfinite(y)
        if valid.sum()<150:raise ValueError('insufficient valid source/validation cycles')
        cells.append((x[valid],y[valid],n[valid]))
    train=np.concatenate([r[0] for r in cells[:3]]);mean=train.mean(axis=0);scale=train.std(axis=0)
    if (scale<1e-12).any():raise ValueError('constant HI')
    state=dict(mean=mean.tolist(),scale=scale.tolist(),fitted_cells=list(CELLS[:3]),
        source_counts=[len(r[0]) for r in cells],selected_ids=selection['selected_ids'])
    return [((x-mean)/scale,y,n) for x,y,n in cells],state


def source_run(cells,state,hidden,history,seed,folder):
    folder.mkdir(parents=True,exist_ok=True)
    config=dict(protocol=PROTOCOL,hidden_size=hidden,history=history,seed=seed,
        input_dim=len(state['mean']),scaler=state)
    if (folder/'training.json').exists():
        old=read(folder/'training.json')
        if old['config']!=config or sha(folder/'source.pt')!=old['checkpoint_sha256']:
            raise ValueError('completed source run changed')
        return old
    if (folder/'source.pt').exists():raise FileExistsError('incomplete source checkpoint')
    torch.manual_seed(seed);np.random.seed(seed)
    x=torch.from_numpy(np.concatenate([sequences(a,n,history) for a,_,n in cells[:3]]))
    y=torch.tensor(np.concatenate([b for _,b,_ in cells[:3]]),dtype=torch.float32)
    vx,vy,vn=cells[3];vx=torch.from_numpy(sequences(vx,vn,history));vy=torch.tensor(vy,dtype=torch.float32)
    model=CapacityModel(hidden,len(state['mean']));optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    best=float('inf');best_epoch=0;best_state=None;log=[];started=time.monotonic()
    for epoch in range(1,3001):
        model.train();total=0.
        for idx in torch.randperm(len(x)).split(64):
            optimizer.zero_grad(set_to_none=True);loss=F.mse_loss(model(x[idx]),y[idx])
            if not torch.isfinite(loss):raise ValueError('nonfinite source training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            total+=float(loss.detach())*len(idx)
        model.eval()
        with torch.no_grad():value=float(F.mse_loss(model(vx),vy))
        if not np.isfinite(value):raise ValueError('nonfinite source validation loss')
        log.append(dict(epoch=epoch,train_mse=total/len(x),validation_mse=value))
        if value<best-1e-8:best=value;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
        if epoch-best_epoch>=100:break
    model.load_state_dict(best_state);model.eval()
    torch.save(dict(config=config,state_dict=best_state),folder/'source.pt')
    with torch.no_grad():pred=model(vx).numpy()
    reload=load_source(folder/'source.pt')
    with torch.no_grad():replay=reload(vx).numpy()
    if not np.array_equal(pred,replay):raise ValueError('source checkpoint replay failed')
    np.savez_compressed(folder/'validation.npz',truth=vy.numpy(),prediction=pred,cycles=vn)
    result=dict(config=config,converged=bool(epoch-best_epoch>=100),completed_epochs=epoch,
        best_epoch=best_epoch,validation_mse=best,validation_mae_pp=float(np.abs(pred-vy.numpy()).mean()*100),
        seconds=time.monotonic()-started,history=log,checkpoint_sha256=sha(folder/'source.pt'))
    save(folder/'training.json',result)
    print(json.dumps(dict(stage='source_run',hidden=hidden,history=history,seed=seed,
        epochs=epoch,converged=result['converged'],seconds=result['seconds'])),flush=True)
    return result


def load_source(path):
    d=torch.load(path,map_location='cpu',weights_only=True);c=d['config']
    model=CapacityModel(c['hidden_size'],c['input_dim']);model.load_state_dict(d['state_dict']);model.eval()
    return model


def train(out):
    write_once(out/'execution.json',dict(protocol=PROTOCOL,code_sha256=execution_hashes(),
        python=sys.version,torch=str(torch.__version__),numpy=str(np.__version__),platform=platform.platform()))
    arrays,meta,calibration=features(out);selection=screen(arrays,meta,calibration)
    write_once(out/'selection.json',selection)
    if selection['status']!='selected':raise ValueError('no qualified features; no relaxed threshold')
    cells,state=model_inputs(arrays,selection);write_once(out/'source_scaler.json',state)
    if (out/'selected_model.json').exists():return
    grid=[]
    for hidden in (8,16,32):
        for history in (1,5,10,20):
            runs=[]
            for seed in SEEDS:
                folder=out/'source'/f'h{hidden}_t{history}'/f'seed_{seed}'
                r=source_run(cells,state,hidden,history,seed,folder)
                runs.append(dict(seed=seed,checkpoint=str(folder/'source.pt'),
                    checkpoint_sha256=r['checkpoint_sha256'],converged=r['converged'],validation_mse=r['validation_mse']))
            grid.append(dict(hidden_size=hidden,history=history,runs=runs,
                mean_validation_mse=float(np.mean([r['validation_mse'] for r in runs])),
                valid=all(r['converged'] for r in runs)))
    valid=[r for r in grid if r['valid']]
    if not valid:raise ValueError('no converged source configuration')
    chosen=min(valid,key=lambda r:(r['mean_validation_mse'],r['hidden_size'],r['history']))
    save(out/'selected_model.json',dict(**chosen,grid=grid,test_scores_used=False))


def calibrate(out):
    if (out/'calibration.json').exists():return
    arrays,meta,calibration=features(out);selection=read(out/'selection.json')
    chosen=read(out/'selected_model.json');state=read(out/'source_scaler.json')
    x=(arrays['target_x'][:,selection['selected_indices']]-state['mean'])/state['scale']
    x=torch.from_numpy(sequences(x,arrays['target_cycles'],chosen['history'])).double()
    anchor=[r['row'] for r in calibration['points']];y=[r['soh'] for r in calibration['points']]
    predictions=[];uncalibrated=[];checks=[]
    for run in chosen['runs']:
        seed=run['seed'];base=load_source(run['checkpoint']).double();before=encoder_hash(base)
        with torch.no_grad():z=base.encode(x).numpy();prior=base(x).numpy()
        head,pred,check=svd_dense(base,z,anchor,y)
        model=copy.deepcopy(base);model.dense=head
        if encoder_hash(model)!=before:raise ValueError('LSTM changed during target calibration')
        path=out/f'target_seed_{seed}.pt'
        if path.exists():raise FileExistsError('incomplete calibrated output; refusing overwrite')
        torch.save(dict(config=read(Path(run['checkpoint']).parent/'training.json')['config'],
            state_dict=model.state_dict(),dtype='float64',calibration=check),path)
        replay=load_source(path).double()
        # load_source casts state to FP32 by default; reload FP64 explicitly to
        # preserve SVD head precision and enable an exact inference check.
        stored=torch.load(path,map_location='cpu',weights_only=True)
        replay.load_state_dict(stored['state_dict']);replay.eval()
        with torch.no_grad():disk=replay(x).numpy()
        delta=float(np.max(np.abs(disk-pred)))
        if delta>1e-8:raise ValueError('FP64 target checkpoint replay differs')
        predictions.append(pred);uncalibrated.append(prior)
        checks.append(dict(seed=seed,svd=check,encoder_before=before,encoder_after=encoder_hash(model),
            replay_max_soh_delta=delta,checkpoint_sha256=sha(path)))
    np.savez_compressed(out/'predictions.npz',cycles=arrays['target_cycles'],seeds=np.array(SEEDS),
        predictions=np.array(predictions),source_only=np.array(uncalibrated))
    save(out/'calibration.json',dict(points=calibration['points'],checks=checks,
        prediction_sha256=sha(out/'predictions.npz'),dense_test_labels_read=False))


def score(y,p):
    e=np.asarray(p,float)-np.asarray(y,float)
    if not len(e) or not np.isfinite(e).all():raise ValueError('invalid scores')
    return dict(n=len(e),mae_pp=float(np.abs(e).mean()*100),rmse_pp=float(np.sqrt(np.mean(e*e))*100),
        bias_pp=float(e.mean()*100),mae_Ah=float(np.abs(e).mean()*2.36),max_abs_error_pp=float(np.max(np.abs(e))*100))


def evaluate(out):
    if (out/'results.json').exists():return
    arrays,meta,calibration=features(out)
    if sha(out/'test_truth.npz')!=meta['truth_sha256']:raise ValueError('sealed truth file changed')
    with np.load(out/'test_truth.npz') as truth:cycles=truth['cycles'].copy();y=truth['soh'].copy()
    with np.load(out/'predictions.npz') as pred:
        if not np.array_equal(pred['cycles'],cycles):raise ValueError('prediction/truth cycle mismatch')
        p=pred['predictions'].copy();prior=pred['source_only'].copy()
    anchors=np.array([q['row'] for q in calibration['points']]);test=np.ones(len(y),bool);test[anchors]=False
    linear=np.interp(cycles,cycles[anchors],y[anchors])
    pchip=PchipInterpolator(cycles[anchors],y[anchors],extrapolate=False)(cycles)
    if not np.array_equal(cycles,arrays['target_cycles']):raise ValueError('target feature cycle mismatch')
    phase=np.minimum(2,((cycles-cycles[0])*3//(cycles[-1]-cycles[0]+1)).astype(int))
    per_seed=[]
    for k,seed in enumerate(SEEDS):
        per_seed.append(dict(seed=seed,test=score(y[test],p[k,test]),
            source_only_test=score(y[test],prior[k,test]),calibration=score(y[anchors],p[k,anchors]),
            phases={name:score(y[test&(phase==j)],p[k,test&(phase==j)]) for j,name in enumerate(('early','middle','late'))}))
    mean=p.mean(axis=0);std=p.std(axis=0)
    result=dict(status='completed',test_cell=CELLS[4],training_cells=list(CELLS[:3]),validation_cell=CELLS[3],
        nominal_Ah=2.36,visible_capacity_points=6,hidden_capacity_points=int(test.sum()),
        target_total_preprocessed_cycles=meta['target_total_preprocessed_cycles'],eligible_target_cycles=len(cycles),
        evaluated_cycle_range=[int(cycles[0]),int(cycles[-1])],same_recorded_condition=True,
        generalization_to_one_heldout_battery_not_all_MICH=True,
        per_seed=per_seed,linear=score(y[test],linear[test]),pchip=score(y[test],pchip[test]),
        ensemble_test=score(y[test],mean[test]),
        seed_mean_MAE_pp=float(np.mean([r['test']['mae_pp'] for r in per_seed])),
        seed_std_MAE_pp=float(np.std([r['test']['mae_pp'] for r in per_seed])),
        source_only_seed_mean_MAE_pp=float(np.mean([r['source_only_test']['mae_pp'] for r in per_seed])),
        checkpoint_cycles=cycles[anchors].tolist(),selection=read(out/'selection.json')['selected_ids'],
        selected_model={key:read(out/'selected_model.json')[key] for key in ('hidden_size','history','mean_validation_mse')},
        capacity_labels='full measured discharge / 2.36 Ah, no interpolated training labels',
        dense_truth_read_only_after_model_and_Dense_calibration_frozen=True,
        paired_C3_domain_shift_ablation=False,
        caveat='This is a same-recorded-condition one-cell sanity check. Source batteries, selected HIs and source-selected configuration differ from the C3 experiment, so it cannot isolate a domain-shift causal effect.')
    # JSON rows are a plain scientific data table; no raw observations replaced.
    result['rows']=[dict(cycle=int(n),measured_soh=float(y[j]),calibration_point=bool(not test[j]),
        evaluated=bool(test[j]),linear_soh=float(linear[j]),pchip_soh=float(pchip[j]),
        reconstructed_soh_mean=float(mean[j]),seed_std_soh=float(std[j]),
        **{f'seed_{seed}_soh':float(p[k,j]) for k,seed in enumerate(SEEDS)}) for j,n in enumerate(cycles)]
    save(out/'results.json',result)


def audit(out):
    arrays,meta,calibration=features(out);check_hashes(meta['protected_sha256'])
    check_hashes(read(out/'execution.json')['code_sha256'])
    if sha(out/'test_truth.npz')!=meta['truth_sha256']:raise ValueError('truth changed')
    selection=screen(arrays,meta,calibration)
    if selection!=read(out/'selection.json'):raise ValueError('selection replay failed')
    state=read(out/'source_scaler.json');_,replay=model_inputs(arrays,selection)
    if replay!=state or state['fitted_cells']!=list(CELLS[:3]):raise ValueError('scaler used a wrong battery')
    r=read(out/'results.json');cal=read(out/'calibration.json');chosen=read(out/'selected_model.json')
    if not all(q['converged'] for q in chosen['runs']):raise ValueError('unconverged source model accepted')
    if sha(out/'predictions.npz')!=cal['prediction_sha256']:raise ValueError('predictions changed')
    with np.load(out/'predictions.npz') as d:curves=d['predictions'].copy()
    x=(arrays['target_x'][:,selection['selected_indices']]-state['mean'])/state['scale']
    x=sequences(x,arrays['target_cycles'],chosen['history']).astype(np.float64)
    for k,q in enumerate(cal['checks']):
        seed=q['seed'];path=out/f'target_seed_{seed}.pt'
        if sha(path)!=q['checkpoint_sha256'] or q['encoder_before']!=q['encoder_after'] or not q['svd']['accepted']:
            raise ValueError('target weight/encoder/optimality audit failed')
        stored=torch.load(path,map_location='cpu',weights_only=True);w={key:value.numpy() for key,value in stored['state_dict'].items()}
        h=np.zeros((len(x),chosen['hidden_size']));c=h.copy()
        for step in range(x.shape[1]):
            gates=x[:,step]@w['kernel'].T+h@w['recurrent_kernel'].T+w['bias']
            ig,fg,candidate,og=np.split(gates,4,axis=1);sigmoid=lambda a:1/(1+np.exp(-a))
            c=sigmoid(fg)*c+sigmoid(ig)*np.maximum(candidate,0);h=sigmoid(og)*np.maximum(c,0)
        prediction=(h@w['dense.weight'].T+w['dense.bias']).ravel()
        if not np.allclose(prediction,curves[k],atol=1e-8,rtol=0):raise ValueError('independent NumPy inference failed')
        rows=[v for v in r['rows'] if v['evaluated']];errors=[v[f'seed_{seed}_soh']-v['measured_soh'] for v in rows]
        independent_mae=math.fsum(abs(e) for e in errors)/len(errors)*100
        independent_rmse=math.sqrt(math.fsum(e*e for e in errors)/len(errors))*100
        actual=next(v['test'] for v in r['per_seed'] if v['seed']==seed)
        if abs(independent_mae-actual['mae_pp'])>1e-10 or abs(independent_rmse-actual['rmse_pp'])>1e-10:
            raise ValueError('independent metric replay failed')
    if sum(row['calibration_point'] for row in r['rows'])!=6:raise ValueError('checkpoint count differs')
    if any(row['calibration_point']==row['evaluated'] for row in r['rows']):raise ValueError('calibration/test overlap')
    return dict(status='passed',split_disjoint=True,same_recorded_protocol=True,
        PCA_and_scaler_source_train_only=True,full_hidden_truth_not_in_training_API=True,
        six_calibration_points_excluded_from_test=True,Dense_only_optimality=True,
        independent_numpy_inference=True,independent_MAE_RMSE=True,
        protected_files_unchanged=len(meta['protected_sha256']))


def report(out):
    r=read(out/'results.json');checks=audit(out)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'mplcache'));os.environ.setdefault('XDG_CACHE_HOME',str(out/'cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts={f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({'font.family':next((n for n in ('Arial Unicode MS','PingFang SC') if n in fonts),'DejaVu Sans'),
                         'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    rows=r['rows'];n=np.array([q['cycle'] for q in rows]);y=np.array([q['measured_soh'] for q in rows]);anchor=np.array([q['calibration_point'] for q in rows])
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[2,1]})
    for seed in SEEDS:
        p=np.array([q[f'seed_{seed}_soh'] for q in rows]);axes[0].plot(n,p*100,lw=.7,alpha=.45,label=f'seed={seed}')
        axes[1].plot(n[~anchor],(p-y)[~anchor]*100,lw=.8,alpha=.65,label=f'seed={seed}')
    axes[0].plot(n,y*100,c='#222',lw=1.5,label='逐圈实测容量真值')
    axes[0].plot(n,[q['reconstructed_soh_mean']*100 for q in rows],c='#176b9b',lw=1.5,label='重建三种子均值')
    axes[0].plot(n,[q['linear_soh']*100 for q in rows],c='#c47629',ls='--',lw=1.4,label='同六点线性插值')
    axes[0].scatter(n[anchor],y[anchor]*100,c='#b52236',s=40,zorder=6,label='仅开放的六个容量点')
    axes[0].set(ylabel='SOH (%)',title='MICH MCForm35：同标注工况、跨电芯的容量重建对照')
    axes[1].axhline(0,c='#555',lw=1)
    axes[1].plot(n[~anchor],np.array([q['linear_soh']-q['measured_soh'] for q in rows])[~anchor]*100,
                 c='#c47629',ls='--',lw=1,label='线性插值误差')
    axes[1].set(xlabel='BatteryLife 预处理循环号（有缺失输入的循环未估计）',ylabel='隐藏点误差（百分点）')
    for ax in axes:ax.grid(alpha=.15);ax.legend(fontsize=8,ncol=2)
    fig.savefig(out/'reconstruction.png',dpi=180);plt.close(fig)
    model=r['seed_mean_MAE_pp'];linear=r['linear']['mae_pp']
    conclusion=('在这枚同标注工况电池上，神经重建优于线性插值。' if model<linear else
                '在这枚同标注工况电池上，神经重建仍未优于线性插值。')
    lines=['# MICH 同工况容量重建对照','',f'**结论：{conclusion}**',
        f"只开放六个容量点，评价其余 **{r['hidden_capacity_points']} 个逐圈实测点**。神经网络三种子平均 MAE **{model:.3f} 个 SOH 百分点**，线性插值 **{linear:.3f}**，PCHIP **{r['pchip']['mae_pp']:.3f}**。",'',
        f"![重建及隐藏点误差](../{out.relative_to(ROOT)}/reconstruction.png)",'',
        '## 固定实验设置','',
        '- 工况：同一 MCForm 系列，45℃、1C 充电/1C 放电、0–100% SOC、NMC111 软包、额定容量2.36Ah。这是相同记录工况，不代表电芯个体差异消失。',
        f"- 训练：{', '.join(CELLS[:3])}。验证：{CELLS[3]}。测试：{CELLS[4]}。电芯在评分前固定，不按结果换电池。",
        f"- 测试原文件有 {r['target_total_preprocessed_cycles']} 个预处理循环，输入/容量质量检查后保留 {r['eligible_target_cycles']} 个，范围 {r['evaluated_cycle_range']}。缺失或不完整输入逐项记录，不造数据补齐。",
        f"- 六个容量点：{r['checkpoint_cycles']}。按有效循环位置预先选择首圈、约25%/50%/75%、倒数第二、末圈；最后两点在特征相关性中合为一组。",
        '- 其余测试容量保存在独立文件；筛选、源域训练和Dense校准接口不读取这些值。最终模型和校准先冻结，再打开完整容量做评分。',
        '- 复用C3 v2的方法规则，不复用其源模型或scaler：部分Q-V特征、0.8相关性和方向一致、0.98去冗余、最多3特征；一层ReLU-LSTM＋Dense、仅Dense做FP64 SVD最优校准。',
        '- 特征搜索仍用3.2–4.0V端点间隔0.1V、宽0.2/0.3/0.4V。可评价循环先固定为具有3.6–4.0V完整充电片段、完整放电容量的循环；其他窗口必须覆盖这些循环。此覆盖限制和排除清单公开，不按误差删点。',
        '- 宽度8/16/32、历史1/5/10/20，共12配置×3种子；只按源域验证MSE选型。Adam/MSE、LR=0.001、batch64、最多3000epoch、patience100，未收敛配置不接受。',
        f"- 入选特征：{r['selection']}。最终宽度 {r['selected_model']['hidden_size']}、历史长度 {r['selected_model']['history']}。容量标签为实测完整放电Ah÷2.36，不是插值伪标签。",'',
        '## 结果（不含六个校准点）','','| 种子 | MAE（SOH百分点） | RMSE（SOH百分点） | 未校准源模型MAE |','|---|---:|---:|---:|']
    for q in r['per_seed']:lines.append(f"| {q['seed']} | {q['test']['mae_pp']:.3f} | {q['test']['rmse_pp']:.3f} | {q['source_only_test']['mae_pp']:.3f} |")
    lines+=['',f"三种子MAE：{model:.3f} ± {r['seed_std_MAE_pp']:.3f}（总体标准差，不是置信区间）。只是一枚测试电池，不把数百循环或三个种子当作数百枚独立电池。",'',
        '## 早中晚误差','','| 阶段 | 三种子平均MAE（百分点） |','|---|---:|']
    for phase in ('early','middle','late'):lines.append(f"| {phase} | {np.mean([q['phases'][phase]['mae_pp'] for q in r['per_seed']]):.3f} |")
    lines+=['','## 能得出什么结论','',
        '- 这次可以用密集实测容量检验两次校准点之间的误差，比C3仅有几个隐藏RPT点的检查更有信息。',
        '- 本实验重新训练同工况源电池，也按冻结规则重新选特征和源域超参数；不是只改变“跨域/同域”一个变量的成对消融，不能据此把C3误差完全归因于工况或源数据。',
        '- 相比Che原实验，这仍是本地方法重实现：源电芯、窗口、补充参数和SVD求解器不同；只检查历史容量重建，不做未来寿命预测。',
        '- 本地预处理文件没有可用的温度时间序列；45℃来自文件工况标注，因此本实验核验的是同标注工况，不能独立核实实际温度波动。',
        '- 此次结果不反向用于修改C3特征、模型、标签或既有benchmark。','',
        f"产物目录：`{out.relative_to(ROOT)}`。逐圈真值、预测、插值对照见results.json；封存真值在test_truth.npz，权重和参数在同目录。",'',
        '```bash',f'python -m experiments.mixed95.mich_che_control all --suite-id {out.name}','```','']
    path=ROOT/'reports'/f'MICH_同工况重建_{out.name}.md';path.write_text('\n'.join(lines))
    write_once(out/'audit.json',checks)
    print(json.dumps(dict(report=str(path),MAE_pp=model,linear_MAE_pp=linear,hidden_points=r['hidden_capacity_points']),ensure_ascii=False),flush=True)


def finalize(out):
    if (out/'run_manifest.json').exists():return
    checks=audit(out)
    env=dict(os.environ);env.update(MPLCONFIGDIR=str(out/'verify_mplcache'),
        XDG_CACHE_HOME=str(out/'verify_cache'),TRANSFORMERS_CACHE=str(out/'verify_cache/huggingface'))
    command=[sys.executable,'-m','unittest','discover','-s','experiments/mixed95','-p','test_*.py']
    run=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
    log=run.stdout+run.stderr
    with (out/'test_results.txt').open('x') as stream:stream.write(log)
    if run.returncode:raise ValueError('regression tests failed; inspect test_results.txt')
    count=re.search(r'Ran (\d+) tests?',log);skips=re.search(r'OK \(skipped=(\d+)\)',log)
    artifacts={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and 'cache' not in str(p.relative_to(out))}
    report_path=ROOT/'reports'/f'MICH_同工况重建_{out.name}.md'
    save(out/'run_manifest.json',dict(execution=read(out/'execution.json'),checks=checks,
        tests_run=int(count.group(1)),tests_skipped=int(skips.group(1)) if skips else 0,
        test_exit_code=run.returncode,test_command=command,artifact_sha256=artifacts,
        report=str(report_path),report_sha256=sha(report_path)))
    print(json.dumps(dict(stage='finalized',tests=int(count.group(1)),checks=checks)),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=('extract','train','calibrate','evaluate','report','audit','finalize','all'))
    p.add_argument('--suite-id',default=DEFAULT_SUITE);args=p.parse_args()
    if not re.fullmatch(r'mich_che_control_[A-Za-z0-9_-]+',args.suite_id):raise ValueError('invalid suite-id')
    out=BASE/args.suite_id
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if args.stage in ('all','extract'):extract(out)
    if args.stage in ('all','train'):train(out)
    if args.stage in ('all','calibrate'):calibrate(out)
    if args.stage in ('all','evaluate'):evaluate(out)
    if args.stage in ('all','report'):report(out)
    if args.stage in ('all','finalize'):finalize(out)
    if args.stage=='audit':print(json.dumps(audit(out)),flush=True)


if __name__=='__main__':main()
