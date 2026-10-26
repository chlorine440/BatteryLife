"""Source-only LSTM selection and optimally solved Dense-only adaptation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import numpy as np
from scipy import linalg
import torch
from torch import nn
from torch.nn import functional as F

from experiments.mixed95.c3_che_data import sequences, read, save
from experiments.mixed95.c3_che_model import ReLULSTMCapacity, encoder_hash
from experiments.mixed95.c3_che_v2_data import PROTOCOL, SEEDS, fingerprint, selected_arrays
from experiments.mixed95.c3_evaluate import sha


class CapacityModel(ReLULSTMCapacity):
    """Same ReLU-LSTM equations as v1; only the HI input dimension is variable."""
    def __init__(self, hidden_size, input_dim):
        super().__init__(hidden_size)
        self.input_dim = input_dim
        self.kernel = nn.Parameter(torch.empty(4 * hidden_size, input_dim))
        nn.init.xavier_uniform_(self.kernel)


def fit_source_run(cells, state, hidden, history, seed, folder, context):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    config=dict(hidden_size=hidden,history=history,seed=seed,input_dim=len(state['mean']),
        context=context,source_transform=state,protocol=PROTOCOL)
    signature=fingerprint(config)
    meta_path=folder/'training.json';weight_path=folder/'source.pt'
    if meta_path.exists():
        meta=read(meta_path)
        if meta['signature']!=signature:raise ValueError('source-run signature changed')
        if meta['checkpoint_sha256']!=sha(weight_path):raise ValueError('source checkpoint changed')
        return meta
    if weight_path.exists():raise FileExistsError('incomplete source checkpoint: use a new suite, do not overwrite')
    torch.manual_seed(seed);np.random.seed(seed)
    train_x=np.concatenate([sequences(x,n,history) for x,_,n in cells[:3]])
    train_y=np.concatenate([y for _,y,_ in cells[:3]]).astype(np.float32)
    vx,vy,vc=cells[3]
    vx=sequences(vx,vc,history)
    x=torch.from_numpy(train_x);y=torch.from_numpy(train_y)
    vx=torch.from_numpy(vx);vy=torch.tensor(vy,dtype=torch.float32)
    model=CapacityModel(hidden,config['input_dim'])
    optimizer=torch.optim.Adam(model.parameters(),lr=PROTOCOL['learning_rate'])
    best=float('inf');best_epoch=0;best_state=None;log=[];start=time.monotonic()
    for epoch in range(1,PROTOCOL['max_epochs']+1):
        model.train();total=0.;permutation=torch.randperm(len(x))
        for batch in permutation.split(PROTOCOL['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss=F.mse_loss(model(x[batch]),y[batch])
            if not torch.isfinite(loss):raise ValueError('nonfinite source loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),PROTOCOL['gradient_clip_norm'])
            optimizer.step();total+=float(loss.detach())*len(batch)
        model.eval()
        with torch.no_grad():val=float(F.mse_loss(model(vx),vy))
        log.append(dict(epoch=epoch,train_mse=total/len(x),validation_mse=val))
        if val<best-PROTOCOL['min_delta']:
            best=val;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
        if epoch%250==0:
            print(json.dumps(dict(stage='source_progress',context=context,hidden=hidden,
                history=history,seed=seed,epoch=epoch,best_epoch=best_epoch,
                validation_rmse_pp=100*np.sqrt(best))),flush=True)
        if epoch-best_epoch>=PROTOCOL['patience']:break
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():pred=model(vx).numpy()
    torch.save(dict(config=config,state_dict=best_state),weight_path)
    replay=load_source(weight_path)
    with torch.no_grad():disk_pred=replay(vx).numpy()
    if not np.array_equal(pred,disk_pred):raise ValueError('source reload predictions differ')
    np.savez_compressed(folder/'validation.npz',truth=vy.numpy(),prediction=pred,cycles=vc)
    meta=dict(signature=signature,config=config,history=log,best_epoch=best_epoch,
        completed_epochs=epoch,converged=bool(epoch-best_epoch>=PROTOCOL['patience']),
        validation_mse=best,validation_mae_pp=float(np.abs(pred-vy.numpy()).mean()*100),
        elapsed_seconds=time.monotonic()-start,n_training_cycles=len(x),n_validation_cycles=len(vx),
        checkpoint_sha256=sha(weight_path),encoder_sha256=encoder_hash(model),
        source_validation_only=True,target_scores_used=False)
    save(meta_path,meta)
    print(json.dumps(dict(stage='source_finished',context=context,hidden=hidden,history=history,
        seed=seed,epochs=epoch,converged=meta['converged'],seconds=meta['elapsed_seconds'])),flush=True)
    return meta


def load_source(path):
    stored=torch.load(path,map_location='cpu',weights_only=True)
    c=stored['config'];model=CapacityModel(c['hidden_size'],c['input_dim'])
    model.load_state_dict(stored['state_dict']);model.eval()
    return model


def train_grid(arrays,selection,folder,context):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    cells,state=selected_arrays(arrays,selection)
    signature=fingerprint(dict(selection=selection['selection_sha256'],state=state,protocol=PROTOCOL))
    if (folder/'selected_model.json').exists():
        existing=read(folder/'selected_model.json')
        if existing['signature']!=signature:raise ValueError('selected source model signature changed')
        for run in existing['runs']:
            if sha(Path(run['checkpoint']))!=run['checkpoint_sha256']:raise ValueError('source weight hash mismatch')
        return existing
    state_path=folder/'source_transform.json'
    if state_path.exists():
        if read(state_path)!=state:raise ValueError('source transform changed')
    else:save(state_path,state)
    table=[]
    for hidden in PROTOCOL['hidden_sizes']:
        for history in PROTOCOL['history_lengths']:
            runs=[]
            for seed in SEEDS:
                path=folder/f'h{hidden}_t{history}'/f'seed_{seed}'
                meta=fit_source_run(cells,state,hidden,history,seed,path,context)
                runs.append(dict(seed=seed,checkpoint=str(path/'source.pt'),
                    checkpoint_sha256=meta['checkpoint_sha256'],converged=meta['converged'],
                    validation_mse=meta['validation_mse'],elapsed_seconds=meta['elapsed_seconds']))
            table.append(dict(hidden_size=hidden,history=history,runs=runs,
                valid=all(r['converged'] for r in runs),
                mean_validation_mse=float(np.mean([r['validation_mse'] for r in runs]))))
    eligible=[row for row in table if row['valid']]
    if not eligible:
        save(folder/'failed_grid.json',dict(signature=signature,table=table))
        raise ValueError('all source configurations failed convergence; target reconstruction prohibited')
    chosen=min(eligible,key=lambda r:(r['mean_validation_mse'],r['hidden_size'],r['history']))
    result=dict(signature=signature,source_transform=state,grid=table,**chosen,
        selection_rule='minimum three-seed mean source-validation MSE; no C3 score',
        excluded_nonconverged_configs=sum(not r['valid'] for r in table))
    save(folder/'selected_model.json',result)
    return result


def svd_dense(base,embeddings,indices,labels):
    """Unregularized minimum-norm UPDATE to the pretrained Dense parameters.

    FP64 for both encoder inference and head, so saved-model replay does not
    silently switch numerical precision after an ill-conditioned calibration.
    Independent QR least squares checks achievable training predictions.
    """
    z=np.asarray(embeddings,dtype=np.float64)
    idx=np.asarray(indices,dtype=int);y=np.asarray(labels,dtype=np.float64)
    if len(idx)<2 or len(idx)!=len(y) or len(set(idx.tolist()))!=len(idx):
        raise ValueError('invalid calibration points')
    if not np.isfinite(z).all() or not np.isfinite(y).all():raise ValueError('nonfinite calibration data')
    design=np.c_[z,np.ones(len(z))];a=design[idx]
    theta=np.r_[base.dense.weight.detach().cpu().numpy().ravel(),float(base.dense.bias.detach()[0])]
    residual=y-a@theta
    u,s,vt=np.linalg.svd(a,full_matrices=False)
    keep=s>PROTOCOL['svd_rcond']*s[0]
    update=vt[keep].T@((u[:,keep].T@residual)/s[keep])
    calibrated=theta+update
    prediction=design@calibrated;error=a@calibrated-y
    gradient=a.T@error
    # GELSY uses rank-revealing QR, not our explicit SVD implementation.
    qr,_,qr_rank,_=linalg.lstsq(a,residual,cond=PROTOCOL['svd_rcond'],lapack_driver='gelsy')
    independent_error=a@(theta+qr)-y
    normal_max=float(np.max(np.abs(gradient)))
    agreement=float(np.max(np.abs((a@calibrated)-(a@(theta+qr)))))
    optimal=normal_max<=PROTOCOL['max_normal_equation_residual'] and agreement<=1e-8
    if not np.isfinite(prediction).all() or not optimal:
        raise ValueError(f'Dense least-squares acceptance failed: gradient={normal_max}, QR delta={agreement}')
    head=nn.Linear(z.shape[1],1,dtype=torch.float64)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(calibrated[:-1][None]))
        head.bias.copy_(torch.tensor(calibrated[-1:],dtype=torch.float64))
    condition=float(s[0]/s[-1]) if s[-1]>0 else None
    audit=dict(solver='FP64 SVD minimum source-weight update',regularization=0.,
        train_indices=idx.tolist(),train_soh=y.tolist(),rank=int(keep.sum()),
        singular_values=s.tolist(),condition_number=condition,
        ill_conditioned_warning=bool(condition is None or condition>1e8),
        before_rmse_pp=float(np.sqrt(np.mean(residual**2))*100),
        after_rmse_pp=float(np.sqrt(np.mean(error**2))*100),
        after_mae_pp=float(np.abs(error).mean()*100),
        normal_equation_residual_max=normal_max,independent_QR_rank=int(qr_rank),
        independent_QR_training_delta=agreement,
        independent_QR_rmse_pp=float(np.sqrt(np.mean(independent_error**2))*100),
        weight_update_norm=float(np.linalg.norm(update)),accepted=optimal)
    return head,prediction,audit


def calibrate_checkpoint(source_path,target_x,cycles,indices,labels,save_path):
    base=load_source(source_path).double();before=encoder_hash(base)
    source_config=torch.load(source_path,map_location='cpu',weights_only=True)['config']
    x=torch.from_numpy(sequences(target_x,cycles,source_config['history'])).double()
    with torch.no_grad():z=base.encode(x).numpy()
    head,pred,check=svd_dense(base,z,indices,labels)
    updated=copy.deepcopy(base);updated.dense=head
    after=encoder_hash(updated)
    if before!=after:raise ValueError('LSTM changed during Dense-only calibration')
    if Path(save_path).exists():raise FileExistsError('calibrated weights already exist')
    torch.save(dict(config=source_config,state_dict=updated.state_dict(),dtype='float64',
        calibration=check,source_checkpoint_sha256=sha(Path(source_path)),
        encoder_sha256_before=before,encoder_sha256_after=after),save_path)
    stored=torch.load(save_path,map_location='cpu',weights_only=True)
    replay=CapacityModel(source_config['hidden_size'],source_config['input_dim']).double()
    replay.load_state_dict(stored['state_dict']);replay.eval()
    with torch.no_grad():disk=replay(x).numpy()
    delta=float(np.max(np.abs(disk-pred)))
    if delta>PROTOCOL['max_reload_soh_delta']:raise ValueError('calibrated checkpoint replay failed')
    check.update(replay_max_soh_delta=delta,encoder_sha256_before=before,
                 encoder_sha256_after=after,checkpoint_sha256=sha(Path(save_path)))
    return pred,check
