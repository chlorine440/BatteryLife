"""Run/report Che III-A style capacity reconstruction on C3, locally on CPU.

python -m experiments.mixed95.c3_che {extract,source,train,reconstruct,report,all}
All new artifacts live under dataset/mixed95/c3_che_20261026_v1.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator
import torch

from experiments.mixed95.c3_che_data import (
    OUT, ROOT, WINDOW, TRAIN_CELLS, VAL_CELL, target_features, source_features,
    read, save, protect, transform_indicators, sequences,
)
from experiments.mixed95.c3_che_model import (
    CONFIG, train_sources, load_source, fit_dense, encoder_hash, ReLULSTMCapacity,
)
from experiments.mixed95.c3_evaluate import sha

GROUPS = ((1,), (101,), (201,), (301,), (401,402))


def metrics(truth,prediction):
    e=np.asarray(prediction,dtype=float)-np.asarray(truth,dtype=float)
    if not len(e) or not np.isfinite(e).all():raise ValueError("invalid metric inputs")
    # Independently checked with math.fsum in audit_artifacts below.
    return dict(n=len(e),mae_pp=float(np.abs(e).mean()*100),
                rmse_pp=float(np.sqrt(np.mean(e**2))*100),
                mae_Ah=float(np.abs(e).mean()*183),
                rmse_Ah=float(np.sqrt(np.mean(e**2))*183),
                bias_pp=float(e.mean()*100))


def crossing(cycles,prediction):
    below=np.flatnonzero(np.asarray(prediction)<=.95)
    return int(cycles[below[0]]) if len(below) else None


def reconstruct(out=OUT):
    protected_outputs=[out/"reconstruction.json",out/"C3_reconstructed_soh.csv",
        out/"heldout_rpt.csv",out/"holdout_dense_weights.pt",
        *[out/f"C3_seed_{seed}.pt" for seed in CONFIG["seeds"]]]
    if any(path.exists() for path in protected_outputs):
        raise FileExistsError("neural reconstruction outputs already exist; choose a new output directory")
    torch.set_num_threads(2)
    protected=protect()
    target_audit=read(out/"target_features_audit.json")
    if target_audit["missing"]:
        raise ValueError("target contains missing charge features; no interpolation may hide it")
    for stem in ("target","source"):
        if read(out/f"{stem}_features_audit.json")["feature_sha256"]!=sha(out/f"{stem}_features.npz"):
            raise ValueError("input feature fingerprint mismatch")
    source_state=read(out/"source_transform.json")
    with np.load(out/"target_features.npz",allow_pickle=False) as data:
        cycles=data["cycles"].copy();q=data["delta_q"].copy()
    if not np.array_equal(cycles,np.arange(1,405)) or not np.isfinite(q).all():
        raise ValueError("not all physical cycles are present")
    hi=transform_indicators(q,source_state)
    inputs=sequences(hi,cycles,CONFIG["history_cycles"])
    x=torch.from_numpy(inputs)
    rpt=target_audit["rpt"]
    anchor_cycles=np.array([r["cycle"] for r in rpt],dtype=int)
    anchor_soh=np.array([r["soh"] for r in rpt])
    assert anchor_cycles.tolist()==[1,101,201,301,401,402]
    anchor_indices=np.searchsorted(cycles,anchor_cycles)
    linear=np.interp(cycles,anchor_cycles,anchor_soh)
    pchip=PchipInterpolator(anchor_cycles,anchor_soh,extrapolate=False)(cycles)
    linear[(cycles<anchor_cycles[0])|(cycles>anchor_cycles[-1])]=np.nan
    checkpoint_hashes={str(out/f"source_seed_{seed}.pt"):sha(out/f"source_seed_{seed}.pt") for seed in CONFIG["seeds"]}
    per_seed=[];cv=[];target_weights={};curves=[]
    for seed in CONFIG["seeds"]:
        base=load_source(out,seed);before=encoder_hash(base)
        with torch.no_grad():
            z=base.encode(x);uncalibrated=base.dense(z).squeeze(-1).numpy()
        head,prediction,calibration=fit_dense(base,z,anchor_indices,anchor_soh)
        updated=copy.deepcopy(base);updated.dense.load_state_dict(head.state_dict())
        with torch.no_grad():reloaded=updated(x).numpy()
        if not np.allclose(prediction,reloaded,atol=1e-6,rtol=0) or encoder_hash(updated)!=before:
            raise ValueError("Dense-only adaptation/reload check failed")
        weight_path=out/f"C3_seed_{seed}.pt"
        torch.save(dict(state_dict=updated.state_dict(),source_seed=seed,
                        config=dict(CONFIG),source_transform=source_state,
                        calibration_cycles=anchor_cycles.tolist()),weight_path)
        loaded=torch.load(weight_path,map_location="cpu",weights_only=True)
        replay=ReLULSTMCapacity(CONFIG["hidden_size"]);replay.load_state_dict(loaded["state_dict"]);replay.eval()
        with torch.no_grad():disk_prediction=replay(x).numpy()
        delta=float(np.max(np.abs(prediction-disk_prediction)))
        if delta>1e-6:raise ValueError("saved C3 checkpoint replay failed")
        curves.append(prediction)
        fold_rows=[]
        for group in GROUPS:
            held=np.isin(anchor_cycles,group);train=~held
            fold_head,pred,fit=fit_dense(base,z,anchor_indices[train],anchor_soh[train])
            if encoder_hash(base)!=before:raise ValueError("source encoder changed in CV")
            target_weights[f"seed_{seed}_holdout_{group[0]}"]=fold_head.state_dict()
            kept_cycles=anchor_cycles[train];kept_soh=anchor_soh[train]
            for cycle,index,true in zip(anchor_cycles[held],anchor_indices[held],anchor_soh[held]):
                interior=kept_cycles[0]<=cycle<=kept_cycles[-1]
                row=dict(seed=seed,group=list(group),cycle=int(cycle),measured_soh=float(true),
                    predicted_soh=float(pred[index]),source_only_soh=float(uncalibrated[index]),
                    training_rpt_cycles=kept_cycles.tolist(),
                    linear_soh=float(np.interp(cycle,kept_cycles,kept_soh)) if interior else None,
                    pchip_soh=float(PchipInterpolator(kept_cycles,kept_soh)(cycle)) if interior else None,
                    inside_retained_anchor_range=bool(interior),
                    fit_rmse_pp=fit["final_training_rmse_pp"])
                fold_rows.append(row);cv.append(row)
        scores=metrics([r["measured_soh"] for r in fold_rows],[r["predicted_soh"] for r in fold_rows])
        comparable=[r for r in fold_rows if r["inside_retained_anchor_range"]]
        common={field:metrics([r["measured_soh"] for r in comparable],[r[field] for r in comparable])
                for field in ("predicted_soh","source_only_soh","linear_soh","pchip_soh")}
        per_seed.append(dict(seed=seed,calibration=calibration,heldout_rpt=scores,
            common_interior_holdout=common,first_model_estimated_95_cycle=crossing(cycles,prediction),
            fitted_anchor_metrics=metrics(anchor_soh,prediction[anchor_indices]),
            checkpoint_sha256=sha(weight_path),encoder_sha256_before=before,
            encoder_sha256_after=encoder_hash(updated),checkpoint_replay_max_soh_delta=delta,
            reconstructed_soh_range=[float(prediction.min()),float(prediction.max())],
            weights_outside_nominal_plausibility_range=bool(((prediction<.5)|(prediction>1.2)).any())))
        print(json.dumps(dict(seed=seed,holdout=scores,estimated_crossing=crossing(cycles,prediction))),flush=True)
    torch.save(target_weights,out/"holdout_dense_weights.pt")
    a=np.asarray(curves);mean=a.mean(axis=0);std=a.std(axis=0)
    rows=[]
    by_cycle={r["cycle"]:r for r in rpt}
    for j,n in enumerate(cycles):
        rows.append(dict(physical_cycle=int(n),measured_rpt_soh=by_cycle[int(n)]["soh"] if int(n) in by_cycle else None,
            measured_rpt_capacity_Ah=by_cycle[int(n)]["discharge_ah"] if int(n) in by_cycle else None,
            estimated_soh_mean=float(mean[j]),estimated_soh_seed_std=float(std[j]),
            estimated_capacity_mean_Ah=float(mean[j]*183),
            **{f"estimated_soh_seed_{seed}":float(a[k,j]) for k,seed in enumerate(CONFIG["seeds"])},
            linear_rpt_reference_soh=float(linear[j]) if np.isfinite(linear[j]) else None,
            pchip_rpt_reference_soh=float(pchip[j]) if np.isfinite(pchip[j]) else None,
            cc_charge_current_A=target_audit["quality"][str(n)]["charge_current_median_A"],
            beyond_last_calibration_checkpoint=bool(n>anchor_cycles[-1]),
            label_kind="neural_estimate_not_measured_truth"))
    currents=np.array([r["cc_charge_current_A"] for r in rows])
    current_anchor=currents[anchor_indices]
    current_capacity_ratio=2*current_anchor/183
    report=dict(status="reconstructed_exploratory_not_independently_validated",method="Che 2023 section III-A reimplementation",
        strict_paper_replication=False,source_pretraining=True,lstm_model=True,dense_only_finetuning=True,
        historical_retrospective=True,future_lifetime_model=False,raw_files_modified=False,
        benchmark_truth_replaced=False,suitable_as_measured_benchmark_truth=False,
        nominal_Ah=183.,target_label_div094=False,config=CONFIG,
        changes_from_paper=["MICH NMC111 source cells instead of author's unidentified LFP source cells",
            "3 training cells plus 1 disjoint source validation cell",
            "3.6..4.0V common observed voltage window on 2.5..4.25V/1000 grid",
            "capacity increments and output divided by each cell's nominal Ah",
            "explicit entropy/PCA conventions; PCA/scaler fitted only on source training cells",
            "unpublished history=10, width=16 and training hyperparameters fixed before C3 scores",
            "PyTorch implementation of ReLU-LSTM rather than Keras; no probabilistic future model"],
        source_training_cells=list(TRAIN_CELLS),source_validation_cell=VAL_CELL,
        rpt=rpt,holdout_groups=[list(g) for g in GROUPS],heldout=cv,per_seed=per_seed,rows=rows,
        n_physical_cycles=404,n_rpt=6,n_independent_target_cells=1,n_holdout_groups=5,
        physical_cycle_feature_gaps=[],original_rpt95_bracket_assuming_monotone=[202,301],
        exact_measured_95_cycle=None,ensemble_first_estimated_95_cycle=crossing(cycles,mean),
        feature_ood_max_abs_source_z=np.abs(hi).max(axis=0).tolist(),
        confounding=dict(rpt_cycle_current_A=current_anchor.tolist(),
            current_only_soh_if_I_equals_half_capacity=current_capacity_ratio.tolist(),
            rpt_soh_minus_current_only_soh=(anchor_soh-current_capacity_ratio).tolist(),
            current_RPT_pearson=float(np.corrcoef(current_anchor,anchor_soh)[0,1]),
            warning="Experimental charge settings depend on preceding/new RPT capacities. Withheld RPT labels can remain indirectly encoded in process data. This is NOT an independent validation of between-RPT accuracy."),
        protected_sha256=protected,source_checkpoint_sha256=checkpoint_hashes,
        source_transform_sha256=sha(out/"source_transform.json"),
        source_features_sha256=sha(out/"source_features.npz"),target_features_sha256=sha(out/"target_features.npz"),
        no_target_score_hyperparameter_selection=True,seed_spread_is_not_confidence_interval=True,
        no_interpolation_residual_correction=True,no_monotonic_projection=True)
    if checkpoint_hashes!={p:sha(Path(p)) for p in checkpoint_hashes} or protected!=protect():
        raise ValueError("protected files changed")
    save(out/"reconstruction.json",report)
    with (out/"C3_reconstructed_soh.csv").open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (out/"heldout_rpt.csv").open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(cv[0]));writer.writeheader();writer.writerows(cv)
    audit_artifacts(out)


def audit_artifacts(out=OUT):
    r=read(out/"reconstruction.json")
    for item in r["per_seed"]:
        rows=[q for q in r["heldout"] if q["seed"]==item["seed"]]
        errors=[q["predicted_soh"]-q["measured_soh"] for q in rows]
        mae=100*math.fsum(abs(e) for e in errors)/len(errors)
        rmse=100*math.sqrt(math.fsum(e*e for e in errors)/len(errors))
        if abs(mae-item["heldout_rpt"]["mae_pp"])>1e-10 or abs(rmse-item["heldout_rpt"]["rmse_pp"])>1e-10:
            raise ValueError("independent metric cross-check failed")
        for q in rows:
            if set(q["group"]) & set(q["training_rpt_cycles"]):raise ValueError("holdout labels leaked")
            if q["cycle"] in [401,402] and set(q["training_rpt_cycles"]) & {401,402}:
                raise ValueError("neighboring endpoint leaked")
        if item["encoder_sha256_before"]!=item["encoder_sha256_after"]:raise ValueError("encoder not frozen")
    if [row["physical_cycle"] for row in r["rows"]]!=list(range(1,405)):
        raise ValueError("cycle coverage mismatch")
    if sum(row["measured_rpt_soh"] is not None for row in r["rows"])!=6:
        raise ValueError("fabricated measured capacity")
    for path,value in {**r["protected_sha256"],**r["source_checkpoint_sha256"]}.items():
        if sha(Path(path))!=value:raise ValueError("protected artifact changed")
    for path,value in read(out/"target_features_audit.json")["raw_sha256"].items():
        if sha(Path(path))!=value:raise ValueError("raw C3 workbook changed")
    for cell in read(out/"source_features_audit.json")["cells"]:
        if sha(Path(cell["source_path"]))!=cell["sha256"]:raise ValueError("source pkl changed")
    checks=dict(independent_metric_recompute=True,group_holdout_isolation=True,
        encoder_frozen=True,all_404_cycles=True,exactly_6_measured_RPTs=True,
        benchmark_and_original_inputs_unchanged=True,
        all_nine_target_workbooks_and_four_source_pkls_unchanged=True,
        caveat="These checks establish implementation consistency, not unknown between-RPT truth.")
    if not (out/"cross_check.json").exists():save(out/"cross_check.json",checks)
    return checks


def report(out=OUT):
    r=read(out/"reconstruction.json");audit_artifacts(out)
    os.environ.setdefault("MPLCONFIGDIR",str(out/"mplcache"))
    os.environ.setdefault("XDG_CACHE_HOME",str(out/"cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts={f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.family":next((name for name in ("Arial Unicode MS","PingFang SC","Heiti TC") if name in fonts),"DejaVu Sans"),
        "axes.unicode_minus":False,"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    rows=r["rows"];c=np.array([q["physical_cycle"] for q in rows])
    mean=np.array([q["estimated_soh_mean"] for q in rows])*100
    std=np.array([q["estimated_soh_seed_std"] for q in rows])*100
    ac=np.array([q["cycle"] for q in r["rpt"]]);ay=np.array([q["soh"] for q in r["rpt"]])*100
    fig,(ax,current_ax)=plt.subplots(2,1,figsize=(11,7),sharex=True,gridspec_kw={"height_ratios":[3,1]},layout="constrained")
    for seed in CONFIG["seeds"]:
        ax.plot(c,[q[f"estimated_soh_seed_{seed}"]*100 for q in rows],lw=.8,alpha=.35)
    ax.plot(c,mean,color="#176b9b",lw=2,label="LSTM 重建均值（非逐圈实测）")
    ax.fill_between(c,mean-std,mean+std,color="#176b9b",alpha=.15,label="三种子 ±1 标准差，非置信区间")
    ax.plot(c,[q["pchip_rpt_reference_soh"]*100 if q["pchip_rpt_reference_soh"] is not None else np.nan for q in rows],
            color="#777777",ls="--",lw=1.5,label="RPT 插值参考（非真值）")
    ax.scatter(ac,ay,s=42,c="#b52236",zorder=5,label="6 个 RPT 实测点")
    ax.axhline(95,color="#333333",ls=":",lw=1)
    ax.axvspan(202,301,color="#d6b24c",alpha=.09,label="RPT 支持的 95% 区间（单调假设）")
    ax.set(ylabel="SOH (%)",title="C3 历史 SOH 重建：论文方法的任务适配，不是原实验复现")
    ax.grid(alpha=.15);ax.legend(fontsize=9,loc="best")
    current_ax.plot(c,[q["cc_charge_current_A"] for q in rows],color="#b26b25",lw=1.5)
    current_ax.set(xlabel="物理循环号",ylabel="充电电流 (A)",title="程序设定的充电电流随 RPT 更新，是留点校验的混杂因素",xlim=(-3,409))
    current_ax.grid(alpha=.15)
    fig.savefig(out/"C3_neural_reconstruction.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout="constrained")
    positions=np.arange(len(ac))
    for seed in CONFIG["seeds"]:
        held=[q for q in r["heldout"] if q["seed"]==seed]
        axes[0].plot(positions,[100*q["predicted_soh"] for q in held],"o--",alpha=.7,label=f"seed={seed}")
        axes[1].plot(positions,[100*(q["predicted_soh"]-q["measured_soh"]) for q in held],"o--",alpha=.7,label=f"seed={seed}")
    axes[0].scatter(positions,ay,c="#b52236",marker="x",s=65,zorder=5,label="被留出的 RPT 真值")
    axes[0].set(title="每次留出整组 RPT，重新校准末层",xlabel="被留出的 RPT 圈号（401/402 同组）",ylabel="SOH (%)")
    axes[1].axhline(0,c="#555555",lw=1)
    axes[1].set(title="留组预测减去实测值",xlabel="被留出的 RPT 圈号（401/402 同组）",ylabel="有符号误差（SOH 百分点）")
    for ax in axes:
        ax.set_xticks(positions,ac);ax.grid(alpha=.15);ax.legend(fontsize=9)
    fig.savefig(out/"C3_heldout_RPT.png",dpi=180);plt.close(fig)
    common_neural=float(np.mean([s['common_interior_holdout']['predicted_soh']['mae_pp'] for s in r['per_seed']]))
    common_linear=float(np.mean([s['common_interior_holdout']['linear_soh']['mae_pp'] for s in r['per_seed']]))
    common_pchip=float(np.mean([s['common_interior_holdout']['pchip_soh']['mae_pp'] for s in r['per_seed']]))
    interpretation=("本次神经重建未优于简单线性插值，不支持将其作为更准确的逐圈标签。"
                    if common_neural>=common_linear else
                    "本次在少数中间 RPT 点上优于线性插值，但仍不足以证明逐圈真值精度。")
    lines=["# C3 神经网络容量重建", "",
        "已完成论文第 III-A 节路线的任务适配：三指标、源域 LSTM 预训练、C3 RPT 微调 Dense。不是原实验复现，也不是简单插值。", "",
        f"**{interpretation}** 在同样留出的 101/201/301 圈上，神经网络三种子平均 MAE 为 **{common_neural:.3f} 个 SOH 百分点**，线性插值为 **{common_linear:.3f}**，PCHIP 为 **{common_pchip:.3f}**。", "",
        "**这是一条已生成的模型估计曲线，尚不能称为已验证可靠的逐圈真值。** C3 只有六个 RPT；测试程序又按 RPT 更新充电设定，因此留点结果仍可能受间接容量信息影响。", "",
        "![C3 重建曲线](../"+str((out/"C3_neural_reconstruction.png").relative_to(ROOT))+")", "",
        "## 1. 实际完成内容", "",
        "- 输入覆盖物理第 1–404 圈。RPT 在 1、101、201、301、401、402 圈，SOH=完整放电 Ah÷183，不使用 0.94。",
        "- 只取工步 10 的 3.6–4.0 V 充电片段。源域和 C3 的分段容量均先除各自额定容量。",
        "- MICH 三枚训练电池、一枚独立源域验证电池。PCA、标准化和基础 checkpoint 只由源域确定，不由 C3 成绩选择。",
        "- 1 层 16 单元 ReLU-LSTM，过去及当前 10 圈 HI 输入，Dense 输出 SOH。PyTorch CPU，三个种子。具体参数见 config.json。",
        "- 在 C3 上仅训练 Dense，用全部六个 RPT 得到最终历史曲线，不强制经过实测点，不做残差插值校正、平滑或单调投影。",
        "- 第 403–404 圈使用已观测充电特征估计，但位于最后校准点之后，单独标记；没有预测未来未观测循环。", "",
        "## 2. 留组校验结果", "",
        "每次拿掉一组 RPT，再从源域权重重新校准 Dense。401/402 必须一起拿掉。以下误差针对未用于该次校准的六个 RPT，不是拟合用六点的误差。", "",
        "| 种子 | 源域验证 MAE（百分点） | C3 留组 MAE（百分点） | C3 留组 RMSE（百分点） | 模型曲线首次≤95%圈 |", "|---|---:|---:|---:|---:|"]
    for item in r["per_seed"]:
        src=read(out/f"source_seed_{item['seed']}.json");m=item["heldout_rpt"]
        lines.append(f"| {item['seed']} | {src['val_mae_pp']:.3f} | {m['mae_pp']:.3f} | {m['rmse_pp']:.3f} | {item['first_model_estimated_95_cycle']} |")
    m=np.array([s["heldout_rpt"]["mae_pp"] for s in r["per_seed"]])
    lines += ["",f"三种子留组 MAE：**{m.mean():.3f} ± {m.std():.3f} 个 SOH 百分点**。一个百分点对应 1.83 Ah。三个种子不是三枚独立电池。", "",
        "![留组 RPT 校验](../"+str((out/"C3_heldout_RPT.png").relative_to(ROOT))+")", "",
        "### 与插值公平比较", "",
        "只在留出的 101/201/301 圈比较，三种方法均不用被留出的容量，且不对端点外推。", "",
        "| 种子 | LSTM MAE（百分点） | 线性插值 MAE | PCHIP MAE |", "|---|---:|---:|---:|"]
    for item in r["per_seed"]:
        m=item["common_interior_holdout"]
        lines.append(f"| {item['seed']} | {m['predicted_soh']['mae_pp']:.3f} | {m['linear_soh']['mae_pp']:.3f} | {m['pchip_soh']['mae_pp']:.3f} |")
    lines += ["", "## 3. 不能忽略的限制", "",
        "1. **程序设定与容量相关。** 日常充电电量和电流按 RPT 更新。移除 RPT 容量标签，并不保证其信息从过程曲线中消失。上述留组成绩不能证明两个 RPT 之间每圈都估得准。",
        f"   在六个 RPT 圈，充电电流与容量的相关系数为 {r['confounding']['current_RPT_pearson']:.6f}；仅用充电电流(A)×2 h÷183 Ah 即与 RPT SOH 相差约 0.017–0.031 个百分点。这是测试程序关联，不是独立学出的 SOH。",
        "2. 三种子波动只反映初始化敏感性，不是完整的不确定度或 95% 置信区间。",
        "3. 原论文没有公布准确源域三枚编号和若干网络参数。本次使用已固定的 MICH NMC111 源数据、更换共同电压区间，并明确补齐宽度/输入长度/优化设置。",
        "4. 模型给出的首次 95% 是估计，不是实测寿命。原 RPT 支持的 202–301 圈条件区间及旧 benchmark 不被替换。",
        "5. 留组检查仍只有 1 枚目标电池、5 组 RPT（6 个点），不能形成独立逐圈精度结论。", "",
        "## 4. 数据问题处理", "",
        "- 原已批准的单条非充电 XML 记录仅在内存读取时隔离；其余行完整保留，原始 CRC 失败仍记录。",
        "- 第 187 圈在约 3.49 V 暂停，位于所用窗口之前；不用非充电点跨越片段插值。",
        "- 第 188 圈一条时间戳约错位 1 秒，以连续采集记录号恢复原顺序，未改变容量数值。",
        "- 每圈完整工步容量与汇总核对；全部原文件和冻结实验哈希在处理前后核对。", "",
        "## 5. 源数据与补充设置", "",
        *["- 训练：`"+name+"`。" for name in TRAIN_CELLS],
        "- 源域验证：`"+VAL_CELL+"`。", "",
        "源域分别保留 377、391、480 个有效循环，验证保留 491 圈；逐圈排除原因见 source_features_audit.json。源容量标签来自负电流放电段的实测容量，范围检查固定为额定容量的 0.5–1.2 倍，另要求充电片段完整覆盖特征窗口。C3 容量标签不参与源数据筛选。", "",
        "宽度 16、历史长度 10、三个随机种子、Adam/MSE、源域最多 300 epoch 和 patience=30、目标 Dense 固定 1000 epoch 等均为本任务公开补充设置，不是作者已公布的原参数。容量模型的历史长度不引用为论文寿命预测模型的 10 点设置。", "",
        "## 6. 运行与数据", "", "```bash",
        "dataset/.conda_envs/comac95/bin/python -m experiments.mixed95.c3_che all", "```", "",
        "产物目录：`"+str(out.relative_to(ROOT))+"`。包括逐圈 CSV、留组 CSV/JSON、源域和校准权重、来源与清洗审计、源域 scaler/PCA 和独立指标复算结果。已有结果拒绝覆盖，重跑请用新的 --out。", "",
        "方法依据：Che et al. (2023), DOI 10.1109/TII.2022.3206776，第 III-A 节。原始数据只读，本轮未使用服务器。", ""]
    filename="C3_论文方法重建_20261026.md" if out.resolve()==OUT.resolve() else f"C3_论文方法重建_{out.name}.md"
    path=ROOT/"reports"/filename
    if path.exists() and str(out.relative_to(ROOT)) not in path.read_text():
        raise FileExistsError("refusing to replace report belonging to another experiment")
    with path.open("w") as stream:stream.write("\n".join(lines))
    print(json.dumps(dict(report=str(path),plot=str(out/"C3_neural_reconstruction.png")),ensure_ascii=False),flush=True)


def finalize(out=OUT):
    """Record actual regression evidence and immutable code/artifact fingerprints."""
    if (out/"run_manifest.json").exists():raise FileExistsError("final manifest already exists")
    checks=audit_artifacts(out)
    env=dict(os.environ)
    env.update(MPLCONFIGDIR=str(out/"mplcache"),XDG_CACHE_HOME=str(out/"cache"),
               TRANSFORMERS_CACHE=str(out/"cache/huggingface"))
    command=[sys.executable,"-m","unittest","discover","-s","experiments/mixed95","-p","test_*.py"]
    run=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
    log=run.stdout+run.stderr
    with (out/"test_results.txt").open("x") as stream:stream.write(log)
    if run.returncode:raise ValueError("regression tests failed; see test_results.txt")
    match=re.search(r"Ran (\d+) tests?",log)
    files=[ROOT/"experiments/mixed95"/name for name in
        ("c3_che.py","c3_che_data.py","c3_che_model.py","test_c3_che.py",
         "c3_quarantine.py","c3_evaluate.py")]
    files.append(ROOT/"experiments/comac95/data.py")
    artifacts={str(p.relative_to(ROOT)):sha(p) for p in out.iterdir() if p.is_file()}
    save(out/"run_manifest.json",dict(method="Che III-A inspired C3 reconstruction; not strict replication",
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},artifact_sha256=artifacts,
        python=sys.version,platform=platform.platform(),torch=str(torch.__version__),numpy=np.__version__,
        test_command=command,tests_run=int(match.group(1)) if match else None,
        test_exit_code=run.returncode,checks=checks,
        result_interpretation="Generated neural estimates; did not outperform linear interpolation at common interior heldout RPTs; no measured truth replacement"))
    print(json.dumps(dict(finalized=True,tests=int(match.group(1)) if match else None,checks=checks)),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("stage",choices=("extract","source","train","reconstruct","report","audit","finalize","all"))
    parser.add_argument("--out",type=Path,default=OUT)
    args=parser.parse_args();out=args.out
    if args.stage in ("all","extract") and not (out/"target_features_audit.json").exists():target_features(out)
    if args.stage in ("all","source") and not (out/"source_features_audit.json").exists():source_features(out)
    if args.stage in ("all","train"):train_sources(out)
    if args.stage in ("all","reconstruct"):reconstruct(out)
    if args.stage in ("all","report"):report(out)
    if args.stage in ("all","finalize"):finalize(out)
    if args.stage=="audit":print(json.dumps(audit_artifacts(out)))


if __name__=="__main__":main()
