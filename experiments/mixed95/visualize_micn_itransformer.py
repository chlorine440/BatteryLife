"""Read-only post-hoc figures from audited predictions; no model inference.

Does not edit the benchmark, select checkpoints, or evaluate new target windows.
PNG/SVG figures and JSON are new outputs. All plotted residuals are computed in
float64; formal total-life metrics are cross-checked against the frozen report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("MICN", "iTransformer")
SEEDS = (2021, 42, 2024)
COLORS = {"MICN": "#0072B2", "iTransformer": "#D55E00"}
MARKERS = {"MICN": "o", "iTransformer": "s"}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(p, y):
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    assert p.shape == y.shape and np.isfinite(p).all() and np.isfinite(y).all()
    assert (y > 0).all()
    e = p - y
    return dict(mae_cycles=float(np.abs(e).mean()), rmse_cycles=float(np.sqrt((e * e).mean())),
                mape=float((np.abs(e) / y).mean()), acc15=float((np.abs(e) / y <= .15).mean() * 100),
                acc10=float((np.abs(e) / y <= .10).mean() * 100), bias_cycles=float(e.mean()))


def independent(p, y):
    # Different arithmetic implementation from the figure aggregations.
    pairs = list(zip(np.asarray(p).ravel().tolist(), np.asarray(y).ravel().tolist()))
    n = len(pairs)
    e = [abs(a - b) for a, b in pairs]
    rel = [d / b for d, (_, b) in zip(e, pairs)]
    return dict(mae_cycles=math.fsum(e) / n, rmse_cycles=math.sqrt(math.fsum(d*d for d in e) / n),
                mape=math.fsum(rel) / n, acc15=sum(d <= .15 for d in rel) / n * 100,
                acc10=sum(d <= .1 for d in rel) / n * 100)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "dataset/mixed95/benchmarks/benchmark95_20261013_autoformer_retry_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/micn_itransformer_visualization_20261013_v2")
    args = parser.parse_args()
    out, suite = args.output.resolve(), args.suite.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "analysis.json").exists():
        raise FileExistsError("Choose an independent output directory; completed outputs are not overwritten")
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import TwoSlopeNorm
    font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = "Arial Unicode MS"
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    inputs = [suite / "report.json", ROOT / "experiments/mixed95/benchmark_baseline.json"]
    inputs += [suite / "runs" / m / f"seed_{s}" / "source_test_predictions.npz" for m in MODELS for s in SEEDS]
    before = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    report, baseline = read(inputs[0]), read(inputs[1])
    cells = [r for r in baseline["cells"] if r["split"] == "test" and r["life"] is not None]
    names = np.array([c["cell"] for c in cells])
    datasets = np.array([c["dataset"] for c in cells])
    assert len(cells) == 96
    w = np.arange(1, 101, dtype=np.float64)
    truth, pred, delta = None, {}, []
    source_summary, target_summary, stages = {}, {}, {}
    target_rows = [r for r in report["predictions"] if r["model"] in MODELS]
    assert len(target_rows) == 12
    for m in MODELS:
        matrices = []
        source_summary[m] = {}
        for seed in SEEDS:
            path = suite / "runs" / m / f"seed_{seed}" / "source_test_predictions.npz"
            with np.load(path, allow_pickle=False) as a:
                p, y = a["predicted"].astype(np.float64), a["truth"].astype(np.float64)
                assert np.array_equal(a["observation_cycles"], np.tile(w, 96))
                assert np.array_equal(a["cell"], np.repeat(names, 100))
            assert p.shape == y.shape == (9600,) and np.isfinite(p).all()
            y = y.reshape(96, 100)
            assert np.max(np.abs(y[:, 0] - np.array([c["life"] for c in cells]))) < .001
            assert np.array_equal(y, np.repeat(y[:, :1], 100, axis=1))
            if truth is None:
                truth = y
            assert np.array_equal(truth, y)
            p = p.reshape(96, 100)
            matrices.append(p)
            source_summary[m][str(seed)] = {}
            for protocol, idx in (("all_windows", slice(None)), ("cycle100_only", slice(99, 100))):
                val = metric(p[:, idx], y[:, idx])
                check = independent(p[:, idx], y[:, idx])
                for k in check:
                    delta.append(abs(val[k] - check[k]))
                source_summary[m][str(seed)][protocol] = val
            source_summary[m][str(seed)]["rul100"] = metric(p[:, 99] - 100, y[:, 99] - 100)
        pred[m] = np.stack(matrices)
        for protocol in ("all_windows", "cycle100_only"):
            for k in ("mae_cycles", "rmse_cycles", "mape", "acc15", "acc10"):
                measured = np.mean([source_summary[m][str(s)][protocol][k] for s in SEEDS])
                expected = next(row for row in report["summary"] if row["model"] == m)[f"source_{protocol}_{k}_mean"]
                delta.append(abs(measured - expected))
        target_summary[m] = {}
        for seed in SEEDS:
            rows = sorted([r for r in target_rows if r["model"] == m and r["seed"] == seed], key=lambda r: r["cell"])
            assert [r["cell"] for r in rows] == ["C1", "C2"]
            p = np.array([r["predicted_life_cycle"] for r in rows])
            y = np.array([r["proxy_label_life_cycle"] for r in rows])
            assert np.array_equal(y, [379, 419])
            for row, pp, yy in zip(rows, p, y):
                delta.extend([abs(abs(pp-yy) - row["absolute_error_cycles"]),
                              abs(100*abs(pp-yy)/yy - row["ape_percent"])])
                row.update(observation_physical_cycle=100, predicted_rul_cycles=float(pp-100),
                           proxy_true_rul_cycles=float(yy-100), signed_error_cycles=float(pp-yy),
                           rul_ape_percent=float(100*abs(pp-yy)/(yy-100)))
            target_summary[m][str(seed)] = dict(total_life=metric(p, y), rul100=metric(p-100, y-100))
        for k in ("mae_cycles", "rmse_cycles", "mape", "acc15", "acc10"):
            measured = np.mean([target_summary[m][str(s)]["total_life"][k] for s in SEEDS])
            delta.append(abs(measured-report["models"][m]["target_metrics"][k]["mean"]))
        stages[m] = {}
        for label, ix in (("1–20", slice(0, 20)), ("21–40", slice(20, 40)),
                          ("41–60", slice(40, 60)), ("61–80", slice(60, 80)), ("81–100", slice(80, 100))):
            stages[m][label] = dict(total_life=[metric(p[:, ix], truth[:, ix]) for p in pred[m]],
                                   remaining_life=[metric(p[:, ix]-w[ix], truth[:, ix]-w[ix]) for p in pred[m]])
    assert max(delta) < 1e-9, max(delta)
    figures = []

    def save(fig, name, caption):
        fig.savefig(out / (name + ".png"), dpi=180, bbox_inches="tight", facecolor="white")
        fig.savefig(out / (name + ".svg"), bbox_inches="tight", facecolor="white")
        figures.append(dict(file=name + ".png", svg=name + ".svg", caption=caption))
        plt.close(fig)

    def axes_clean(ax, x=None, y=None):
        if x: ax.set_xlabel(x)
        if y: ax.set_ylabel(y)
        ax.grid(alpha=.16)
        ax.set_axisbelow(True)

    # Only the genuine target observation point is drawn, not synthetic trajectories.
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.1), constrained_layout=True, sharey=True)
    for ax, cell, yy in zip(axs, ("C1", "C2"), (279, 319)):
        ax.axhspan(yy*.85, yy*1.15, color="#009E73", alpha=.1, label="RUL ±15% 范围")
        ax.axhline(yy, color="#222222", lw=2, label=f"代理真实 RUL：{yy} 圈")
        ax.axhline(0, color="#999999", lw=1)
        for m, offset in zip(MODELS, (-.13, .13)):
            vals = [next(r["predicted_rul_cycles"] for r in target_rows if r["model"]==m and r["seed"]==s and r["cell"]==cell) for s in SEEDS]
            xx = np.arange(3) + offset
            ax.scatter(xx, vals, color=COLORS[m], marker=MARKERS[m], s=65, label=m, zorder=3)
            for x, v in zip(xx, vals):
                ax.annotate(f"{v:.1f}", (x,v), xytext=(0, 9 if v>=0 else -16), textcoords="offset points", ha="center", fontsize=11)
        ax.set_title(f"{cell}：只使用物理前 100 圈")
        ax.set_xticks(range(3), [str(s) for s in SEEDS])
        axes_clean(ax, "训练随机种子", "第 100 圈的剩余寿命 RUL（圈）")
        ax.set_ylim(-65, 665)
    axs[1].legend(loc="upper right", fontsize=10)
    fig.suptitle("商飞：预测 RUL 与代理真实 RUL（各点均为已保存的正式预测）")
    save(fig, "01_comac_rul", "商飞只有第100圈一个观察点，不画虚构的逐圈预测。绿色范围以剩余寿命279/319为分母，与正式总寿命15%-Acc不同。iTransformer seed42出现负RUL，保留不截断。")

    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.3), constrained_layout=True)
    for ax, denom in zip(axs, ("total", "rul")):
        a = np.array([[r["ape_percent"] if denom=="total" else r["rul_ape_percent"] for cell in ("C1","C2")
                       for r in target_rows if r["model"]==m and r["seed"]==s and r["cell"]==cell] for m in MODELS for s in SEEDS])
        im=ax.imshow(a, cmap="YlOrRd", vmin=0, vmax=max(115,float(a.max())), aspect="auto")
        for (i,j), val in np.ndenumerate(a):
            ax.text(j,i,f"{val:.2f}%\n{'≤15% 达标' if val<=15 else '>15%'}", ha="center",va="center",color="white" if val>65 else "#222222",fontsize=11)
        ax.set_yticks(range(6), [f"{m} / {s}" for m in MODELS for s in SEEDS])
        ax.set_xticks([0,1],["C1","C2"])
        ax.set_title("正式口径：总寿命相对误差" if denom=="total" else "补充口径：剩余寿命相对误差")
        ax.set_xlabel("商飞测试电芯")
    fig.suptitle("商飞：同一个绝对误差，换成 RUL 分母后相对误差会变大")
    save(fig,"02_comac_errors","总寿命分母379/419；第100圈剩余寿命分母279/319。不能把MICN总寿命33.33%直接当作RUL的15%-Acc，后者16.67%。六条为两个电芯×三个种子，不是六枚独立电芯。")

    errors = {m:pred[m]-truth for m in MODELS}
    true_rul = truth-w
    window_metrics = {}
    specs = [("mae", "MAE（圈）"), ("rmse", "RMSE（圈）"), ("total_mape", "总寿命 MAPE（%）"),
             ("rul_mape", "RUL MAPE（%）"), ("rul_acc15", "RUL 15%-Acc（%）"), ("bias", "平均有符号误差（圈）")]
    fig,axs=plt.subplots(2,3,figsize=(14.3,8.4),constrained_layout=True)
    for ax,(key,label) in zip(axs.flat,specs):
        for m in MODELS:
            e=errors[m]
            values={"mae":np.abs(e).mean(1),"rmse":np.sqrt((e*e).mean(1)),
                "total_mape":(100*np.abs(e)/truth).mean(1), "rul_mape":(100*np.abs(e)/true_rul).mean(1),
                "rul_acc15":(np.abs(e)/true_rul<=.15).mean(1)*100,"bias":e.mean(1)}[key]
            mean,std=values.mean(0),values.std(0)
            window_metrics.setdefault(m,{})[key]=dict(mean=mean.tolist(), std_population=std.tolist(), seeds=values.tolist())
            ax.plot(w,mean,label=m,color=COLORS[m],lw=2)
            ax.fill_between(w,mean-std,mean+std,color=COLORS[m],alpha=.13)
        if key=="bias":ax.axhline(0,color="#999999",lw=1)
        axes_clean(ax,"已观察源域循环数（1–100）",label)
        ax.set_title(label)
        if key=="mae":ax.legend()
    fig.suptitle("源域 96 电芯：误差随观察窗口变化；线为三种子均值，阴影为种子总体标准差")
    save(fig,"03_source_window_metrics","横轴是可见输入从前1圈扩展至前100圈，不是沿全寿命滚动窗口。RUL误差分母T−t，越接近终点越小；因此应同时看绝对MAE和RUL MAPE。")

    fig,axs=plt.subplots(2,3,figsize=(14.2,8.9),constrained_layout=True,sharex=True,sharey=True)
    ru=truth[:,99]-100
    lo=min(-50,float(min((pred[m][:,:,99]-100).min() for m in MODELS))-30)
    hi=max(float(ru.max()),float(max((pred[m][:,:,99]-100).max() for m in MODELS)))+65
    for i,m in enumerate(MODELS):
        for j,s in enumerate(SEEDS):
            ax=axs[i,j];p=pred[m][j,:,99]-100
            x=np.linspace(0,hi,100)
            ax.fill_between(x,.85*x,1.15*x,color="#009E73",alpha=.1)
            ax.plot(x,x,color="#222222",lw=1.2)
            ax.scatter(ru,p,c=COLORS[m],marker=MARKERS[m],s=23,alpha=.6)
            flagged=names==baseline['warning_cell']
            ax.scatter(ru[flagged],p[flagged],facecolors="none",edgecolors="#CC79A7",s=100,lw=1.8)
            vals=source_summary[m][str(s)]['rul100']
            ax.set_title(f"{m} / {s}\nMAE {vals['mae_cycles']:.1f} 圈；RUL MAPE {100*vals['mape']:.1f}%")
            axes_clean(ax,"标签真实 RUL（圈，T−100）","预测 RUL（圈，预测T−100）")
            ax.set_xlim(-25,hi);ax.set_ylim(lo,hi)
    fig.suptitle("源域：第 100 圈 RUL 散点；对角线理想预测，绿色 ±15%，紫圈为已知标签质量警告电芯")
    save(fig,"04_source_rul_scatter","每面板96枚电芯，各种子单列，不将288次重复预测当作288枚电芯。负RUL不裁剪。")

    fig,axs=plt.subplots(2,2,figsize=(12.6,9),constrained_layout=True)
    all_e=np.concatenate([errors[m][:,:,99].ravel() for m in MODELS])
    edges=np.linspace(float(all_e.min())-10,float(all_e.max())+10,33)
    for m in MODELS:
        e=errors[m][:,:,99].ravel()
        axs[0,0].hist(e,bins=edges,histtype="step",lw=2,label=m,color=COLORS[m])
        z=np.sort(np.abs(e));axs[0,1].plot(z,np.arange(1,len(z)+1)/len(z)*100,color=COLORS[m],label=m,lw=2)
        r=np.sort((100*np.abs(errors[m][:,:,99])/(truth[:,99]-100)).ravel())
        axs[1,0].plot(r,np.arange(1,len(r)+1)/len(r)*100,color=COLORS[m],label=m,lw=2)
        x=ru;p=errors[m][:,:,99].mean(0)
        axs[1,1].scatter(x,p,color=COLORS[m],marker=MARKERS[m],alpha=.6,s=26,label=m)
    axs[0,0].axvline(0,color="#999999",lw=1);axs[1,1].axhline(0,color="#999999",lw=1)
    axs[1,0].axvline(15,color="#999999",ls="--",lw=1,label="15%")
    # Log scale keeps large near-endpoint errors visible instead of clipping.
    axs[1,0].set_xscale("symlog",linthresh=15)
    largest_relative=max(float((100*np.abs(errors[m][:,:,99])/(truth[:,99]-100)).max()) for m in MODELS)
    axs[1,0].set_xlim(0,largest_relative*1.1)
    for ax,labx,laby in zip(axs.flat,["预测T−真实T（圈）：正值高估","绝对误差（圈）","RUL 相对绝对误差（%，对称对数轴）","标签真实 RUL（圈）"],
                            ["重复预测条数（96电芯×3种子）","累计占比（%）","累计占比（%）","三种子平均有符号误差（圈）"]):
        axes_clean(ax,labx,laby);ax.legend()
    fig.suptitle("源域第 100 圈：误差方向、尾部以及与真实剩余寿命的关系")
    save(fig,"05_source_error_distribution","前三面板保留每次预测；右下只是电芯级种子平均偏差，不能替代正式逐种子MAE。CDF和直方图不是置信区间。")

    order=np.argsort(truth[:,0],kind="stable")
    med={m:np.median(errors[m],axis=0)[order] for m in MODELS}
    limit=max(float(np.abs(v).max()) for v in med.values())
    fig,axs=plt.subplots(1,2,figsize=(13.4,9.1),constrained_layout=True,sharey=True)
    for ax,m in zip(axs,MODELS):
        im=ax.imshow(med[m],aspect="auto",cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),extent=(.5,100.5,95.5,-.5))
        ticks=[0,19,38,57,76,95]
        ax.set_yticks(ticks,[f"#{k+1}  T={truth[order[k],0]:.0f}" for k in ticks])
        ax.set_xlabel("已观察源域循环数（1–100）");ax.set_title(m)
    axs[0].set_ylabel("96枚电芯，按寿命T从短到长排序")
    fig.colorbar(im,ax=axs,label="三种子有符号误差中位数（圈）：红色高估，蓝色低估",shrink=.8)
    fig.suptitle("源域：逐电芯 × 观察圈数误差热力图（两个模型相同排序、相同色标，不截断极值）")
    save(fig,"06_source_error_heatmap","图中每行是同一枚电芯，左右顺序一致。三种子中位数只用于图像摘要；三种子全量数值保留在原NPZ，不用于替换正式指标。完整排序见analysis.json。")

    representative=list(order[np.rint(np.linspace(0,95,11)).astype(int)])
    warning_index=int(np.flatnonzero(names==baseline['warning_cell'])[0])
    if warning_index not in representative:representative.append(warning_index)
    else:representative.append(next(int(k) for k in order if int(k) not in representative))
    representative=sorted(representative,key=lambda k:truth[k,0])
    fig,axs=plt.subplots(4,3,figsize=(15.5,14),constrained_layout=True)
    for ax,k in zip(axs.flat,representative):
        ax.plot(w,true_rul[k],color="#222222",lw=2,label="标签真实 RUL")
        for m in MODELS:
            p=pred[m][:,k,:]-w
            ax.plot(w,p.mean(0),color=COLORS[m],lw=1.6,label=m)
            ax.fill_between(w,p.min(0),p.max(0),color=COLORS[m],alpha=.12)
        ax.axhline(0,color="#999999",lw=.7)
        title=names[k].removesuffix('.pkl')
        if k==warning_index:title+=' [标签警告]'
        ax.set_title(f"{title}\nT={truth[k,0]:.0f} 圈",fontsize=11)
        axes_clean(ax,"观察圈数（1–100）","剩余寿命 RUL（圈）")
    axs[0,0].legend(fontsize=9)
    fig.suptitle("源域 12 个固定示例：真实 RUL 与预测 RUL；阴影为三个种子的最小–最大值（非置信区间）")
    save(fig,"07_source_rul_trajectories","按标签寿命排序等距选11枚，再加入已知警告电芯，不按模型好坏选示例。每条预测是该观察窗口下重新输出的已缓存总寿命减去t；没有伪造100圈以后的轨迹。")

    dataset_groups=[(d,datasets==d) for d in sorted(set(datasets))]
    life_groups=[(label,(truth[:,0]>a)&(truth[:,0]<=b)) for label,a,b in
                 [("101–200",100,200),("201–400",200,400),("401–800",400,800),(">800",800,np.inf)]]
    groups_json={}
    fig,axs=plt.subplots(2,2,figsize=(14.4,9.2),constrained_layout=True)
    for row,(label,groups) in enumerate((("数据集",dataset_groups),("标签寿命区间",life_groups))):
        groups_json[label]={}
        for col,(key,ylabel) in enumerate((("mae_cycles","MAE（圈）"),("mape","RUL MAPE（%）"))):
            ax=axs[row,col]
            for m,offset in zip(MODELS,(-.16,.16)):
                means,stds=[],[]
                for name,mask in groups:
                    vals=[metric(p[mask,99]-100,truth[mask,99]-100) for p in pred[m]]
                    groups_json[label].setdefault(name,{"cells":int(mask.sum())})[m]=vals
                    values=np.array([v[key] for v in vals])*(100 if key=="mape" else 1)
                    means.append(values.mean());stds.append(values.std())
                ax.errorbar(np.arange(len(groups))+offset,means,yerr=stds,fmt=MARKERS[m],color=COLORS[m],capsize=3,label=m)
            ax.set_xticks(range(len(groups)),[f"{n}\nn={int(mask.sum())}" for n,mask in groups],rotation=15 if row==0 else 0)
            axes_clean(ax,label,ylabel);ax.set_title(f"第 100 圈：按{label}分组");ax.legend()
    fig.suptitle("源域：误差集中在哪些数据集与短／长寿命电芯？误差棒为种子标准差")
    save(fig,"08_source_groups","n是电芯数不是种子数。小样本组仅作描述，不用于可靠排名；短寿命电芯在第100圈剩余寿命很小，RUL百分比误差可能很大。")

    fraction=w/truth
    bounds=np.array([0,.05,.10,.20,.40,.60,.80,1.0])
    age_summary=[]
    fig,axs=plt.subplots(1,3,figsize=(15,5.4),constrained_layout=True)
    labels=[]
    for i,(a,b) in enumerate(zip(bounds[:-1],bounds[1:])):
        mask=(fraction>a)&(fraction<=b)
        labels.append(f"{a*100:.0f}–{b*100:.0f}%")
        item=dict(age_interval=[float(a),float(b)],cell_count=int(mask.any(1).sum()),cell_window_count=int(mask.sum()),models={})
        for m,offset in zip(MODELS,(-.12,.12)):
            vals=[metric(p[mask]-np.broadcast_to(w,truth.shape)[mask],true_rul[mask]) for p in pred[m]]
            item['models'][m]=vals
            for ax,key,mult in ((axs[0],'mae_cycles',1),(axs[1],'mape',100)):
                v=np.array([q[key]*mult for q in vals])
                ax.errorbar(i+offset,v.mean(),yerr=v.std(),fmt=MARKERS[m],color=COLORS[m],capsize=3,label=m if i==0 else None)
        age_summary.append(item)
    for ax,y in zip(axs[:2],("MAE（圈）","RUL MAPE（%）")):
        ax.set_xticks(range(len(labels)),labels,rotation=30);axes_clean(ax,"相对年龄 t／标签寿命T",y);ax.legend()
    axs[2].bar(range(len(labels)),[q['cell_count'] for q in age_summary],color="#999999")
    for i,q in enumerate(age_summary):axs[2].text(i,q['cell_count']+.7,str(q['cell_count']),ha="center")
    axs[2].set_xticks(range(len(labels)),labels,rotation=30)
    axes_clean(axs[2],"相对年龄 t／标签寿命T","覆盖的不同电芯数")
    fig.suptitle("源域相对早／后期的补充描述：仅有前 100 圈，后期分组是短寿命电芯，并非同一批电芯的全寿命")
    save(fig,"09_source_relative_age","相对年龄分析混合不同寿命电芯，且每枚电芯可贡献多个窗口；分组组成变化，不能证明模型在同一电芯真正后期变好或变坏。右图给出实际电芯覆盖。")

    endpoint_diagnostics={}
    short=truth[:,99]<=200
    for m in MODELS:
        relative=100*np.abs(errors[m][:,:,99])/true_rul[:,99]
        endpoint_diagnostics[m]=dict(short_life_cells=int(short.sum()),other_cells=int((~short).sum()),
            short_life_share_of_summed_RUL_APE_percent=float(relative[:,short].sum()/relative.sum()*100),
            median_RUL_APE_percent_per_seed=np.median(relative,axis=1).tolist(),
            mean_RUL_APE_other81_percent=float(relative[:,~short].mean()),
            negative_RUL_count_per_seed=[int((p[:,99]-100<0).sum()) for p in pred[m]])
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in inputs},"frozen inputs modified"
    result=dict(suite=str(suite.relative_to(ROOT)),no_new_inference=True, source_cells=96, source_arrays=6,
        source_windows_per_seed=9600, observation_range=[1,100], proxy_target_labels={"C1":379,"C2":419},
        target_formal_predictions=target_rows,source_seed_metrics=source_summary,target_seed_metrics=target_summary,
        source_observation_stages=stages, source_per_window=window_metrics, source_groups_at100=groups_json,
        source_relative_age=age_summary, source_endpoint_diagnostics=endpoint_diagnostics,
        heatmap_order=[dict(rank=i+1,cell=str(names[k]),life=float(truth[k,0])) for i,k in enumerate(order)],
        representative_cells=[str(names[k]) for k in representative],figures=figures,
        validation=dict(passed=True, input_hashes_unchanged=True,max_metric_delta=max(delta), input_sha256=before),
        notes=["RUL_hat(t)=T_hat(prefix_t)-t; true_proxy_RUL(t)=T95-t", "No target predictions before or after physical100 were inferred",
               "Source all windows are repeated measurements of 96 cells, not 9600 independent cells",
               "Target has two cells; seeds are repetitions, not independent cells", "Negative RUL preserved; seed spread is not a confidence interval",
               "All source95 test labels include the known flagged cell; no exclusions in these formal figures"])
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out),figures=len(figures),validation=result['validation']['passed'],max_delta=max(delta),
                         stage_metrics={m:{q:{key:np.mean([v[key] for v in d['total_life']]) for key in ('mae_cycles','mape')} for q,d in stages[m].items()} for m in MODELS}),ensure_ascii=False))


if __name__=='__main__':
    main()
