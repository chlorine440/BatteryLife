"""One ReLU-LSTM layer + linear Dense: Che III-A method reimplementation.

PyTorch CPU is used for the same gated equations rather than installing another
framework. The ReLU convention follows paper prose (its generic equations show
tanh). Unpublished dimensions, history and optimizer settings are explicit.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.mixed95.c3_che_data import (
    OUT, TRAIN_CELLS, VAL_CELL, fit_indicators, transform_indicators,
    sequences, save, read,
)
from experiments.mixed95.c3_evaluate import sha

CONFIG = dict(hidden_size=16, history_cycles=10, activation="relu",
    recurrent_activation="sigmoid", optimizer="Adam", learning_rate=.001,
    loss="MSE on capacity/nominal", batch_size=64, max_epochs=300,
    patience=30, min_delta=1e-7, dropout=0., precision="float32", device="cpu",
    fine_tune_epochs=1000, fine_tune_lr=.001, fine_tune_weight_decay=0.,
    gradient_clip_norm=5., seeds=[2021,42,2024],
    checkpoint_selection="source validation cell MSE only",
    output_capacity_unit="capacity / nominal Ah; multiply by 183 for C3 Ah",
    framework="PyTorch; ReLU-gated LSTM equations equivalent to Keras activation=relu",
    not_author_hyperparameters=True, not_strict_replication=True)


class ReLULSTMCapacity(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.hidden_size=hidden_size
        self.kernel=nn.Parameter(torch.empty(4*hidden_size,3))
        self.recurrent_kernel=nn.Parameter(torch.empty(4*hidden_size,hidden_size))
        self.bias=nn.Parameter(torch.zeros(4*hidden_size))
        self.dense=nn.Linear(hidden_size,1)
        nn.init.xavier_uniform_(self.kernel)
        nn.init.orthogonal_(self.recurrent_kernel)
        with torch.no_grad():self.bias[hidden_size:2*hidden_size].fill_(1.)

    def encode(self,x):
        h=x.new_zeros((len(x),self.hidden_size));c=torch.zeros_like(h)
        for current in x.unbind(1):
            ig,fg,candidate,og=(F.linear(current,self.kernel,self.bias)+
                               F.linear(h,self.recurrent_kernel)).chunk(4,dim=-1)
            c=torch.sigmoid(fg)*c+torch.sigmoid(ig)*F.relu(candidate)
            h=torch.sigmoid(og)*F.relu(c)
        return h

    def forward(self,x):
        return self.dense(self.encode(x)).squeeze(-1)


def encoder_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        if not name.startswith("dense."):
            digest.update(name.encode());digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def source_arrays(out=OUT):
    data=np.load(out/"source_features.npz",allow_pickle=False)
    q=np.concatenate([data[name[:-4]+"_q"] for name in TRAIN_CELLS])
    state=fit_indicators(q)
    arrays={}
    for split,names in (("train",TRAIN_CELLS),("val",(VAL_CELL,))):
        xs=[];ys=[]
        for name in names:
            key=name[:-4];hi=transform_indicators(data[key+"_q"],state)
            xs.append(sequences(hi,data[key+"_cycles"],CONFIG["history_cycles"]))
            ys.append(data[key+"_y"].astype(np.float32))
        arrays[split]=(torch.from_numpy(np.concatenate(xs)),torch.from_numpy(np.concatenate(ys)))
    data.close()
    return arrays,state


def train_sources(out=OUT):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config_path=out/"config.json"
    config=dict(CONFIG,training_cells=list(TRAIN_CELLS),validation_cell=VAL_CELL,
        source_features_sha256=sha(out/"source_features.npz"))
    if config_path.exists():
        if read(config_path)!=config:raise ValueError("frozen config mismatch")
    else:save(config_path,config)
    arrays,state=source_arrays(out)
    state_path=out/"source_transform.json"
    if state_path.exists():
        if read(state_path)!=state:raise ValueError("source transform mismatch")
    else:save(state_path,state)
    x,y=arrays["train"];vx,vy=arrays["val"]
    for seed in CONFIG["seeds"]:
        checkpoint=out/f"source_seed_{seed}.pt"
        if checkpoint.exists():
            log=read(out/f"source_seed_{seed}.json")
            if log["checkpoint_sha256"]!=sha(checkpoint):raise ValueError("source checkpoint changed")
            continue
        torch.manual_seed(seed);np.random.seed(seed)
        model=ReLULSTMCapacity(CONFIG["hidden_size"])
        optimizer=torch.optim.Adam(model.parameters(),lr=CONFIG["learning_rate"])
        best=float("inf");best_epoch=0;best_state=None;log=[]
        started=time.monotonic()
        for epoch in range(1,CONFIG["max_epochs"]+1):
            model.train();permutation=torch.randperm(len(x));total=0.
            for batch in permutation.split(CONFIG["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                loss=F.mse_loss(model(x[batch]),y[batch])
                if not torch.isfinite(loss):raise ValueError("nonfinite source training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),CONFIG["gradient_clip_norm"])
                optimizer.step();total+=loss.item()*len(batch)
            model.eval()
            with torch.no_grad():val=float(F.mse_loss(model(vx),vy))
            log.append(dict(epoch=epoch,train_mse=total/len(x),val_mse=val))
            if val<best-CONFIG["min_delta"]:
                best=val;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
            if epoch%25==0:
                print(json.dumps(dict(seed=seed,epoch=epoch,val_rmse_pp=100*val**.5,best_epoch=best_epoch)),flush=True)
            if epoch-best_epoch>=CONFIG["patience"]:break
        model.load_state_dict(best_state);model.eval()
        with torch.no_grad():pred=model(vx).numpy()
        torch.save(dict(state_dict=best_state,seed=seed,config=config),checkpoint)
        reloaded=load_source(out,seed)
        with torch.no_grad():replay=reloaded(vx).numpy()
        if not np.array_equal(pred,replay):raise ValueError("source checkpoint replay mismatch")
        np.savez_compressed(out/f"source_validation_seed_{seed}.npz",prediction=pred,truth=vy.numpy())
        save(out/f"source_seed_{seed}.json",dict(seed=seed,history=log,best_epoch=best_epoch,
            best_validation_mse=best,val_mae_pp=float(np.abs(pred-vy.numpy()).mean()*100),
            val_rmse_pp=float(np.sqrt(np.mean((pred-vy.numpy())**2))*100),
            seconds=time.monotonic()-started,n_train=len(x),n_val=len(vx),
            checkpoint_sha256=sha(checkpoint),encoder_sha256=encoder_hash(model),
            source_transform_sha256=sha(state_path),source_features_sha256=config["source_features_sha256"],
            replay_max_abs=0.,target_data_used=False))
        print(json.dumps(dict(seed=seed,finished=True,best_epoch=best_epoch,seconds=time.monotonic()-started)),flush=True)


def load_source(out,seed):
    stored=torch.load(out/f"source_seed_{seed}.pt",map_location="cpu",weights_only=True)
    model=ReLULSTMCapacity(stored["config"]["hidden_size"])
    model.load_state_dict(stored["state_dict"]);model.eval()
    return model


def fit_dense(base,embeddings,anchor_indices,anchor_soh):
    """Fit exactly the last Dense, from the source weights, using only anchors.

    No target feature transform fit, interpolation correction, label scaler,
    model selection or synthetic capacity points are involved.
    """
    indices=np.asarray(anchor_indices,dtype=int)
    labels=np.asarray(anchor_soh,dtype=np.float32)
    if len(indices)!=len(labels) or len(set(indices))!=len(indices) or len(indices)<2:
        raise ValueError("invalid calibration checkpoints")
    if not np.isfinite(labels).all():raise ValueError("nonfinite calibration labels")
    head=copy.deepcopy(base.dense)
    z=embeddings[torch.as_tensor(indices)]
    y=torch.from_numpy(labels)
    optimizer=torch.optim.Adam(head.parameters(),lr=CONFIG["fine_tune_lr"],
                               weight_decay=CONFIG["fine_tune_weight_decay"])
    for _ in range(CONFIG["fine_tune_epochs"]):
        optimizer.zero_grad(set_to_none=True)
        loss=F.mse_loss(head(z).squeeze(-1),y)
        if not torch.isfinite(loss):raise ValueError("nonfinite target calibration loss")
        loss.backward();optimizer.step()
    with torch.no_grad():prediction=head(embeddings).squeeze(-1).numpy()
    if not np.isfinite(prediction).all():raise ValueError("nonfinite reconstructed capacities")
    return head,prediction,dict(training_indices=indices.tolist(),training_soh=labels.astype(float).tolist(),
        final_training_rmse_pp=float(np.sqrt(np.mean((prediction[indices]-labels)**2))*100),
        epochs=CONFIG["fine_tune_epochs"],only_dense_trainable=True)
