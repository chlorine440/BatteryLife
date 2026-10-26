"""Independent v2 suite: Q-V candidates -> fold-local HIs -> LSTM -> SVD Dense.

python -m experiments.mixed95.c3_che_v2 all --suite-id c3_che_20261026_v2
No v1 entry point or artifact is overwritten. Only report/audit inspect hidden
capacity outcomes. Their results never feed feature/model selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import re
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator
import torch

from experiments.mixed95.c3_che_data import ROOT, OUT as V1, read, save
from experiments.mixed95.c3_che_v2_data import (
    BASE, DEFAULT_SUITE, PROTOCOL, SEEDS, GROUPS, extract_candidates,
    load_candidates, select_features, selected_arrays, fingerprint,
)
from experiments.mixed95.c3_che_v2_model import train_grid, calibrate_checkpoint
from experiments.mixed95.c3_che import metrics, crossing
from experiments.mixed95.c3_evaluate import sha

CONTEXTS=[(f'holdout_{g[0]}',g) for g in GROUPS]+[('full',())]
CODE_FILES=['c3_che_v2.py','c3_che_v2_data.py','c3_che_v2_model.py','test_c3_che_v2.py']


def suite_path(suite_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*_v2[A-Za-z0-9_-]*',suite_id):
        raise ValueError('suite-id must be a plain v2 name, not a path')
    return BASE/suite_id


def write_once(path,value):
    if path.exists():
        if read(path)!=value:raise ValueError(f'existing record differs: {path}')
    else:save(path,value)


def code_hashes():
    return {str(ROOT/'experiments/mixed95'/name):sha(ROOT/'experiments/mixed95'/name) for name in CODE_FILES}


def preflight(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists() and read(out/'protocol.json')!=PROTOCOL:
        raise ValueError('protocol changed; choose another suite-id')
    write_once(out/'protocol.json',PROTOCOL)
    # Freeze v2 code before any source training. Extraction can be reused after
    # a pre-training code fix, but training artifacts require identical code.
    p=out/'execution_manifest.json'
    if p.exists():
        if read(p)['code_sha256']!=code_hashes():raise ValueError('v2 code changed after training start')
    if (out/'candidates.json').exists():
        _,meta=load_candidates(out)
        for path,expected in meta['immutable_sha256'].items():
            if sha(Path(path))!=expected:raise ValueError(f'protected data changed: {path}')


def select_all(out):
    arrays,meta=load_candidates(out);results=[]
    for name,hidden in CONTEXTS:
        folder=out/name;folder.mkdir(exist_ok=True)
        # Hidden labels are removed BEFORE calling the selection API.
        calibration=[r for r in meta['rpt'] if r['cycle'] not in hidden]
        selection=select_features(arrays,meta,calibration)
        write_once(folder/'selection.json',selection)
        results.append(dict(context=name,hidden_cycles=list(hidden),
            status=selection['status'],selected_ids=selection['selected_ids']))
    write_once(out/'selection_summary.json',dict(contexts=results))
    return results


def train_all(out):
    preflight(out)
    summary=read(out/'selection_summary.json')['contexts']
    if any(r['status']!='selected' for r in summary):
        raise ValueError('a fold has no qualified HIs; stop without lowering threshold')
    write_once(out/'execution_manifest.json',dict(code_sha256=code_hashes(),
        protocol_sha256=fingerprint(PROTOCOL),python=sys.version,torch=str(torch.__version__),
        numpy=str(np.__version__),platform=platform.platform(),device='cpu'))
    arrays,meta=load_candidates(out)
    for name,_ in CONTEXTS:
        selection=read(out/name/'selection.json')
        train_grid(arrays,selection,out/name/'source',name)


def calibrate_all(out):
    preflight(out);arrays,meta=load_candidates(out)
    for name,hidden in CONTEXTS:
        folder=out/name
        if (folder/'calibration.json').exists():
            stored=read(folder/'calibration.json')
            for r in stored['per_seed']:
                if sha(folder/f"C3_seed_{r['seed']}.pt")!=r['checks']['checkpoint_sha256']:
                    raise ValueError('calibrated checkpoint hash changed')
            continue
        selection=read(folder/'selection.json');chosen=read(folder/'source/selected_model.json')
        _,state=selected_arrays(arrays,selection)
        if state!=chosen['source_transform']:raise ValueError('source-only scaler changed')
        x=(arrays['target_x'][:,selection['selected_indices']]-state['mean'])/state['scale']
        if not np.isfinite(x).all():raise ValueError('missing target HI; no interpolation allowed')
        calibration=[r for r in meta['rpt'] if r['cycle'] not in hidden]
        indices=[r['cycle']-1 for r in calibration];labels=[r['soh'] for r in calibration]
        rows=[];curves=[]
        for run in chosen['runs']:
            seed=run['seed']
            pred,checks=calibrate_checkpoint(run['checkpoint'],x,arrays['target_cycles'],
                indices,labels,folder/f'C3_seed_{seed}.pt')
            curves.append(pred)
            rows.append(dict(seed=seed,checks=checks,calibration_metrics=metrics(labels,pred[indices]),
                source_checkpoint=run['checkpoint'],calibration_cycles=[n+1 for n in indices],
                prediction_range=[float(pred.min()),float(pred.max())]))
        # All predictions are produced before any hidden label is evaluated.
        np.savez_compressed(folder/'predictions.npz',cycles=arrays['target_cycles'],
            seeds=np.array(SEEDS),predictions=np.asarray(curves))
        save(folder/'calibration.json',dict(context=name,hidden_cycles=list(hidden),
            selected_ids=selection['selected_ids'],hidden_size=chosen['hidden_size'],history=chosen['history'],
            source_transform=state,per_seed=rows,predictions_sha256=sha(folder/'predictions.npz'),
            hidden_capacities_read=False))


def validate(out):
    arrays,meta=load_candidates(out);heldout=[];source_summary=[]
    old=read(V1/'reconstruction.json')
    for name,hidden in CONTEXTS:
        r=read(out/name/'calibration.json');chosen=read(out/name/'source/selected_model.json')
        if sha(out/name/'predictions.npz')!=r['predictions_sha256']:raise ValueError('predictions changed')
        source_summary.append(dict(context=name,selected_ids=r['selected_ids'],
            hidden_size=r['hidden_size'],history=r['history'],
            mean_source_validation_rmse_pp=100*math.sqrt(chosen['mean_validation_mse']),
            grid_configs=len(chosen['grid']),excluded_nonconverged=chosen['excluded_nonconverged_configs']))
        if not hidden:continue
        with np.load(out/name/'predictions.npz') as d:pred=d['predictions'].copy()
        retained=[q for q in meta['rpt'] if q['cycle'] not in hidden]
        kept_n=[q['cycle'] for q in retained];kept_y=[q['soh'] for q in retained]
        for k,seed in enumerate(SEEDS):
            for q in meta['rpt']:
                n=q['cycle']
                if n not in hidden:continue
                interior=min(kept_n)<=n<=max(kept_n)
                old_row=next(a for a in old['heldout'] if a['seed']==seed and a['cycle']==n)
                heldout.append(dict(context=name,seed=seed,cycle=n,measured_soh=q['soh'],
                    predicted_soh=float(pred[k,n-1]),v1_soh=old_row['predicted_soh'],
                    linear_soh=float(np.interp(n,kept_n,kept_y)) if interior else None,
                    pchip_soh=float(PchipInterpolator(kept_n,kept_y)(n)) if interior else None,
                    interior=interior,calibration_cycles=kept_n,
                    v1_feature_selection_not_nested=True))
    per_seed=[]
    for seed in SEEDS:
        rr=[r for r in heldout if r['seed']==seed]
        inside=[r for r in rr if r['interior']]
        ends=[r for r in rr if not r['interior']]
        per_seed.append(dict(seed=seed,all_heldout=metrics([r['measured_soh'] for r in rr],[r['predicted_soh'] for r in rr]),
            endpoint_extrapolation=metrics([r['measured_soh'] for r in ends],[r['predicted_soh'] for r in ends]),
            common_interior={key:metrics([r['measured_soh'] for r in inside],[r[key] for r in inside])
                for key in ('predicted_soh','v1_soh','linear_soh','pchip_soh')}))
    full=read(out/'full/calibration.json')
    with np.load(out/'full/predictions.npz') as d:curves=d['predictions'].copy()
    mean=curves.mean(axis=0);std=curves.std(axis=0)
    measured={r['cycle']:r for r in meta['rpt']};rows=[]
    for j,n in enumerate(arrays['target_cycles']):
        q=measured.get(int(n))
        rows.append(dict(physical_cycle=int(n),estimated_soh_mean=float(mean[j]),
            estimated_soh_seed_std=float(std[j]),estimated_capacity_mean_Ah=float(mean[j]*183),
            **{f'estimated_soh_seed_{s}':float(curves[k,j]) for k,s in enumerate(SEEDS)},
            measured_rpt_soh=q['soh'] if q else None,measured_rpt_capacity_Ah=q['discharge_ah'] if q else None,
            beyond_last_calibration_checkpoint=bool(n>402),label_kind='model_estimate_not_measured_truth'))
    result=dict(status='completed_with_validation_limits',strict_paper_replication=False,
        historical_reconstruction_only=True,results_used_to_modify_protocol=False,
        all_404_cycles=True,nominal_Ah=183.,target_label_div094=False,
        final_selected_ids=full['selected_ids'],source_configurations=source_summary,
        heldout=heldout,per_seed=per_seed,rows=rows,
        estimated_first_95_cycle=crossing(arrays['target_cycles'],mean),
        per_seed_first_95={str(s):crossing(arrays['target_cycles'],curves[k]) for k,s in enumerate(SEEDS)},
        exact_measured_first_95_cycle=None,original_RPT95_bracket_assuming_monotone=[202,301],
        one_target_cell=True,seed_std_is_not_confidence_interval=True,
        process_current_confounding_not_eliminated=True,
        v1_comparison_is_diagnostic_not_a_nested_feature_selection_result=True,
        rpt=meta['rpt'])
    write_once(out/'reconstruction.json',result)
    # Plain machine-readable experiment tables, not an authored Excel workbook.
    for filename,data in [('C3_reconstructed_soh.csv',rows),('heldout_rpt.csv',heldout)]:
        path=out/filename
        if not path.exists():
            with path.open('x',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    return result


def audit(out):
    preflight(out);arrays,meta=load_candidates(out)
    # Repeat feature selection without a hidden label ever entering the API.
    for name,hidden in CONTEXTS:
        calibration=[r for r in meta['rpt'] if r['cycle'] not in hidden]
        stored=read(out/name/'selection.json');replay=select_features(arrays,meta,calibration)
        if replay!=stored:raise ValueError('feature selection replay differs')
        changed=[dict(r,soh=r['soh']+100 if r['cycle'] in hidden else r['soh']) for r in meta['rpt']]
        isolated=select_features(arrays,meta,[r for r in changed if r['cycle'] not in hidden])
        if isolated!=stored:raise ValueError('hidden capacity affected HI screening')
        if not (out/name/'calibration.json').exists():continue
        chosen=read(out/name/'source/selected_model.json')
        rr=read(out/name/'calibration.json')
        if not all(run['converged'] for run in chosen['runs']):raise ValueError('unconverged source model accepted')
        for record in rr['per_seed']:
            c=record['checks']
            if not c['accepted'] or c['encoder_sha256_before']!=c['encoder_sha256_after']:
                raise ValueError('Dense-only optimality/freeze check failed')
            if set(record['calibration_cycles'])&set(hidden):raise ValueError('held-out label calibrated Dense')
            if sha(out/name/f"C3_seed_{record['seed']}.pt")!=c['checkpoint_sha256']:
                raise ValueError('target weight changed')
        if sha(out/name/'predictions.npz')!=rr['predictions_sha256']:raise ValueError('target predictions changed')
    if (out/'reconstruction.json').exists():
        r=read(out/'reconstruction.json')
        if [q['physical_cycle'] for q in r['rows']]!=list(range(1,405)):raise ValueError('coverage not 404')
        if sum(q['measured_rpt_soh'] is not None for q in r['rows'])!=6:raise ValueError('invented RPT truth')
        for seed in SEEDS:
            errors=[q['predicted_soh']-q['measured_soh'] for q in r['heldout'] if q['seed']==seed]
            record=next(q for q in r['per_seed'] if q['seed']==seed)['all_heldout']
            if abs(record['mae_pp']-100*math.fsum(abs(e) for e in errors)/len(errors))>1e-10:
                raise ValueError('independent MAE mismatch')
            if abs(record['rmse_pp']-100*math.sqrt(math.fsum(e*e for e in errors)/len(errors)))>1e-10:
                raise ValueError('independent RMSE mismatch')
    checks=dict(immutable_files=len(meta['immutable_sha256']),immutable_hashes_unchanged=True,
        fold_selection_replay=True,hidden_label_perturbation=True,
        no_heldout_label_in_calibration=True,source_only_PCA_and_scaler=True)
    return checks


def report(out):
    r=read(out/'reconstruction.json');checks=audit(out)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'mplcache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts={f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({'font.family':next((n for n in ('Arial Unicode MS','PingFang SC','Heiti TC') if n in fonts),'DejaVu Sans'),
                         'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    a=np.arange(1,405);rr=r['rows'];ac=np.array([q['cycle'] for q in r['rpt']]);ay=np.array([q['soh'] for q in r['rpt']])
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for seed in SEEDS:ax.plot(a,[q[f'estimated_soh_seed_{seed}']*100 for q in rr],lw=.8,alpha=.4,label=f'seed={seed}')
    mean=np.array([q['estimated_soh_mean'] for q in rr]);std=np.array([q['estimated_soh_seed_std'] for q in rr])
    ax.plot(a,mean*100,c='#176b9b',lw=2,label='v2 重建均值（模型估计）')
    ax.fill_between(a,(mean-std)*100,(mean+std)*100,color='#176b9b',alpha=.12,label='种子标准差，非置信区间')
    old=read(V1/'reconstruction.json')
    ax.plot(a,[q['estimated_soh_mean']*100 for q in old['rows']],c='#b88242',ls='--',alpha=.7,label='v1 均值')
    ax.plot(a,PchipInterpolator(ac,ay,extrapolate=False)(a)*100,c='#888',ls=':',label='RPT 插值参考，非逐圈真值')
    ax.scatter(ac,ay*100,c='#b52236',s=40,zorder=5,label='实测 RPT')
    ax.axhline(95,c='#333',ls=':',lw=1);ax.axvspan(402,404,color='#aaa',alpha=.1)
    ax.set(xlabel='物理循环号',ylabel='SOH (%)',title='C3 历史容量重建 v2：重新筛选特征，仅校准 Dense')
    ax.grid(alpha=.15);ax.legend(fontsize=9,ncol=2)
    fig.savefig(out/'C3_reconstruction_v2.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for seed in SEEDS:
        q=[q for q in r['heldout'] if q['seed']==seed]
        axes[0].plot(range(6),[v['predicted_soh']*100 for v in q],'o--',label=f'seed={seed}')
        axes[1].plot(range(6),[(v['predicted_soh']-v['measured_soh'])*100 for v in q],'o--',label=f'seed={seed}')
    axes[0].scatter(range(6),ay*100,c='#b52236',marker='x',s=65,zorder=6,label='被隐藏的 RPT 实测值')
    axes[1].axhline(0,c='#555',lw=1)
    for ax in axes:ax.set_xticks(range(6),ac);ax.set_xlabel('被隐藏的 RPT 圈号（401/402 同组）');ax.grid(alpha=.15);ax.legend(fontsize=8)
    axes[0].set(ylabel='SOH (%)',title='隐藏容量后重新筛选、训练和校准')
    axes[1].set(ylabel='误差（SOH 百分点）',title='隐藏点预测减去实测值')
    fig.savefig(out/'C3_heldout_v2.png',dpi=180);plt.close(fig)
    common={key:float(np.mean([q['common_interior'][key]['mae_pp'] for q in r['per_seed']])) for key in
            ('predicted_soh','v1_soh','linear_soh','pchip_soh')}
    improved=common['predicted_soh']<common['linear_soh']
    lines=['# C3 容量重建 v2','',
        '**结论：** 已完成重新选特征、源域预训练和 SVD 末层校准；校准通过最优性及重载验收。',
        f"最终特征：{', '.join(r['final_selected_ids'])}。",
        f"在共同隐藏的 101/201/301 圈，三种子平均 MAE：v2 **{common['predicted_soh']:.3f}**、旧版 **{common['v1_soh']:.3f}**、线性插值 **{common['linear_soh']:.3f}**、PCHIP **{common['pchip_soh']:.3f}** 个 SOH 百分点。",
        ('v2 在这三个隐藏点上优于线性插值，但样本太少，不能证明逐圈准确。' if improved else
         'v2 仍未优于线性插值，当前方法尚未体现优势；不继续根据这些测试结果调参。'),
        '曲线仅用于研究性历史估计，不能替代逐圈实测标签。','',
        f"![曲线](../{out.relative_to(ROOT)}/C3_reconstruction_v2.png)",'',
        '## 验证结果','',
        '| 种子 | 全部隐藏 RPT MAE（百分点） | 中间三点 MAE | 端点外推 MAE |',
        '|---|---:|---:|---:|']
    for q in r['per_seed']:
        lines.append(f"| {q['seed']} | {q['all_heldout']['mae_pp']:.3f} | {q['common_interior']['predicted_soh']['mae_pp']:.3f} | {q['endpoint_extrapolation']['mae_pp']:.3f} |")
    lines+=['',f"![隐藏点](../{out.relative_to(ROOT)}/C3_heldout_v2.png)",'',
        '## 特征与训练细节','',
        '- 同一充电 Q-V 路线，扩展总量、均值、标准差、四分位数、四分位距、熵、PC1；当前电流和圈号不作为输入。',
        '- 0.8 相关性门槛、方向一致性、0.98 冗余门槛、最多三个特征及窗口枚举均为任务适配，不是作者公布的自动搜索算法。',
        '- 每折先剔除隐藏容量，再筛选特征。401/402 同组，相关性按组等权。PCA、标准化只用三枚源训练电池。',
        '- 一层 ReLU-LSTM＋线性 Dense；PyTorch 按论文 ReLU 文字定义实现。论文一般公式仍写 tanh，此歧义保留披露。',
        '- 每折 12 组宽度/历史长度配置、每组三种子，只按独立源电池验证 MSE 选型，不按 C3 成绩选型。',
        '- Adam/MSE、LR=0.001、batch=64、最多3000 epoch、patience=100。未收敛配置不进入选型，逐项保留记录。',
        '- 冻结 LSTM；FP64 SVD 求最小二乘最优的 Dense 权重更新，独立 QR 复核，不加正则、不做插值校正或单调投影。','',
        '| 场景 | 特征 | 宽度 | 历史长度 | 源域验证 RMSE（百分点，三种子 MSE 均值开方） |',
        '|---|---|---:|---:|---:|']
    for q in r['source_configurations']:
        lines.append(f"| {q['context']} | {', '.join(q['selected_ids'])} | {q['hidden_size']} | {q['history']} | {q['mean_source_validation_rmse_pp']:.3f} |")
    full=read(out/'full/calibration.json')
    lines+=['','## 末层校准验收','','| 种子 | 校准点 RMSE（百分点） | 正规方程残差 | 矩阵秩 | 条件数 |','|---|---:|---:|---:|---:|']
    for q in full['per_seed']:
        c=q['checks'];condition=c['condition_number']
        lines.append(f"| {q['seed']} | {c['after_rmse_pp']:.8f} | {c['normal_equation_residual_max']:.2e} | {c['rank']} | {condition:.3g} |" if condition is not None else
                     f"| {q['seed']} | {c['after_rmse_pp']:.8f} | {c['normal_equation_residual_max']:.2e} | {c['rank']} | 秩亏 |")
    lines+=['','## 解释边界与可复查产物','',
        '- 这是 Che 2023 III-A 的 C3 任务适配，不是作者原始实验复现；源数据为固定 MICH NMC111，电压窗口和补充参数不同。',
        '- C3 SOH＝完整 RPT Ah÷183，不使用0.94。六个 RPT 的全点校准仅生成最终曲线，不当作独立测试成绩。',
        '- 隐藏点验证只有一枚电池、五组容量。充电设定随 RPT 更新的混杂仍在，不能证明两个 RPT 间的每圈都准确。',
        '- 旧版未做同样的嵌套特征筛选，旧版对比仅用于诊断。三种子标准差不是置信区间。',
        f"- 模型均值首次≤95%：第 {r['estimated_first_95_cycle']} 圈，仅为估计；实测 RPT 在单调假设下只支持202–301圈。403/404圈位于最后校准点之后。",
        '- 候选、选择、完整训练日志、权重和独立检查均保存在实验目录；原始数据、v1和benchmark不变。','',
        '```bash',f'python -m experiments.mixed95.c3_che_v2 all --suite-id {out.name}','```','',
        f"产物目录：`{out.relative_to(ROOT)}`。原始论文 DOI：10.1109/TII.2022.3206776。",'']
    path=ROOT/'reports'/f'C3_论文路线重建_{out.name}.md'
    # Report is a deterministic view of frozen results, not a training input.
    path.write_text('\n'.join(lines))
    write_once(out/'cross_check.json',checks)
    artifacts={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and
               'cache' not in str(p.relative_to(out)) and p.name!='run_manifest.json'}
    write_once(out/'run_manifest.json',dict(execution=read(out/'execution_manifest.json'),
        checks=checks,artifact_sha256=artifacts,report_path=str(path),report_sha256=sha(path)))
    print(json.dumps(dict(report=str(path),common_interior_mae_pp=common),ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('stage',choices=('extract','select','train','calibrate','validate','report','audit','all'))
    p.add_argument('--suite-id',default=DEFAULT_SUITE)
    args=p.parse_args();out=suite_path(args.suite_id)
    torch.set_num_threads(PROTOCOL['threads']);torch.use_deterministic_algorithms(True)
    preflight(out)
    if args.stage in ('all','extract'):extract_candidates(out)
    if args.stage in ('all','select'):select_all(out)
    if args.stage in ('all','train'):train_all(out)
    if args.stage in ('all','calibrate'):calibrate_all(out)
    if args.stage in ('all','validate'):validate(out)
    if args.stage in ('all','report'):report(out)
    if args.stage=='audit':print(json.dumps(audit(out)),flush=True)


if __name__=='__main__':main()
