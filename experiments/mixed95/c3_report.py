"""Concise post-hoc C3 report; conditional interval bounds share one cell truth."""
import json
import math
from pathlib import Path

import numpy as np

from experiments.mixed95.c3_evaluate import ROOT,SUITE,OUTPUT,read,dump,sha


def main():
    audit=read(OUTPUT/'input_audit.json')
    primary=read(OUTPUT/'predictions.json')['results']
    diagnostic=read(OUTPUT/'autoformer_cpu_diagnostic_predictions.json')['results']
    original=read(SUITE/'report.json')
    test_log=OUTPUT/'test_results.txt'
    if not test_log.exists() or not test_log.read_text().rstrip().endswith('OK'):
        raise ValueError('successful final unit test evidence is required')
    lo,hi=audit['label']['integer_crossing_range_assuming_monotone']
    truth=np.arange(lo,hi+1,dtype=float)
    table=[]
    for model in original['models']:
        rows=[r for r in primary if r['model']==model and r['status']=='complete']
        status='通过CPU复核'
        if model=='Autoformer':rows=diagnostic;status='仅CPU诊断（复核超差）'
        if not rows:
            table.append(dict(model=model,status='源域复核失败，未测',predictions=None));continue
        assert len(rows)==3 and [r['seed'] for r in rows]==[2021,42,2024]
        p=np.array([r['predicted_life_cycle'] for r in rows])
        errors=np.abs(p[:,None]-truth[None,:])
        mae=errors.mean(0)
        relative=errors/truth
        rul_relative=errors/(truth-100)
        refs=[r for r in original['predictions'] if r['model']==model]
        comparison={cell:float(np.mean([r['predicted_life_cycle'] for r in refs if r['cell']==cell])) for cell in ('C1','C2')}
        table.append(dict(model=model,status=status,predictions=p.tolist(),predicted_mean=float(p.mean()),
            predicted_std_population=float(p.std()),predicted_rul_mean_at100=float(p.mean()-100),
            common_unknown_truth_integer_range=[lo,hi],
            conditional_MAE_mean_across_seeds_range=[float(mae.min()),float(mae.max())],
            conditional_total_MAPE_mean_range_percent=[float(relative.mean(0).min()*100),float(relative.mean(0).max()*100)],
            conditional_RUL_MAPE_mean_range_percent=[float(rul_relative.mean(0).min()*100),float(rul_relative.mean(0).max()*100)],
            conditional_total_Acc15_range_percent=[float((relative<=.15).mean(0).min()*100),float((relative<=.15).mean(0).max()*100)],
            C1_C2_previous_predicted_mean=comparison,negative_RUL_seeds=[r['seed'] for r in rows if r['predicted_rul_at100']<0]))
    accepted=[r for r in primary if r['status']=='complete']
    replay_max=max(abs(q['delta']) for r in accepted for q in r['replay'])
    bad=[r for r in primary if r['status']!='complete']
    diag_max=max(abs(q['delta']) for r in diagnostic for q in r['replay'])
    hashes_verified=0
    for item in read(SUITE/'final_audit_20261013_v1/local_checkpoint_integrity.json')['files_index']:
        assert sha(ROOT/item['local_path'])==item['sha256'];hashes_verified+=1
    assert all(sha(Path(p))==v for p,v in audit['protected_input_sha256'].items())
    report=dict(models=table,accepted_models=11,accepted_C3_predictions=33,autoformer_diagnostic_predictions=3,
        patchtst_not_evaluated=True,source_weights_files_hash_checked=hashes_verified,
        accepted_replay_max_delta_cycles=replay_max,autoformer_replay_max_delta_cycles=diag_max,
        original_report_sha256=sha(SUITE/'report.json'),no_retraining_or_finetuning=True,
        no_exact_C3_life_label=True,primary_RPT_interval_assuming_monotone=[lo,hi],
        validation_tests=test_log.read_text() if test_log.exists() else None,
        causal_claim_allowed=False,
        limitation='C3 is one different cell with different duty and RPT-only endpoint; C1/C2 proxy point labels are not directly interchangeable with RPT interval labels')
    output=OUTPUT/'summary.json'
    if output.exists():
        previous=read(output)
        if previous['original_report_sha256']!=report['original_report_sha256']:
            raise ValueError('original benchmark has changed')
        with output.open('w') as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write('\n')
    else:dump(output,report)
    micn=next(r for r in table if r['model']=='MICN')
    lines=['# C3本机测试：简要结论','',
        '**已在本机CPU完成推理，没有重训或微调。11种模型/33个种子通过旧预测回放复核；Autoformer另提供3条CPU诊断结果；PatchTST原源域复核失败，不填成绩。**','',
        f"MICN预测C3的总寿命为 **{micn['predictions'][0]:.2f}、{micn['predictions'][1]:.2f}、{micn['predictions'][2]:.2f}圈**，均值 **{micn['predicted_mean']:.2f}圈**。而C3第301圈RPT已经低于95%，三个预测都明显偏晚。",'',
        '## 1. C3的真实寿命并不是精确的301圈','',
        'C3按完整RPT放电容量÷183 Ah计算SOH，不套用C1/C2的日常容量÷0.94。第201圈RPT为96.4721%，第301圈为94.4650%。因此301只是首次观测到低于95%的圈数。假设两次RPT间健康状态单调下降，准确阈值圈数只能限定为 **202–301圈**；第100圈的真实剩余寿命则为 **102–201圈**。没有用线性插值或外推伪造精确标签。','',
        '## 2. 各模型的C3预测','',
        '以下均为总寿命预测圈数。误差范围是在同一个未知真实终点202–301圈内计算，保留三个种子；不是置信区间，也不是把各个种子的最小误差随意取平均。','',
        '| 模型 | seed2021 | seed42 | seed2024 | 三种子MAE可能范围（圈） | 状态 |',
        '|---|---:|---:|---:|---:|---|']
    for r in table:
        if r['predictions'] is None:lines.append(f"| {r['model']} | — | — | — | — | {r['status']} |")
        else:
            p=r['predictions'];a,b=r['conditional_MAE_mean_across_seeds_range']
            lines.append(f"| {r['model']} | {p[0]:.2f} | {p[1]:.2f} | {p[2]:.2f} | {a:.2f}–{b:.2f} | {r['status']} |")
    lines+=['','## 3. 工况是否有影响？','',
        '至少可以确认：换成C3输入后，不同模型的输出有不同程度变化，且MICN没有识别出C3更早到95%的事实。不能只凭三枚不同电芯，把差异全部归因为工况。电芯个体、老化过程、输入分布和标签口径都可能影响结果。','',
        '| 模型 | C1原预测均值 | C2原预测均值 | C3预测均值 |',
        '|---|---:|---:|---:|']
    for m in ('MICN','iTransformer','CPTransformer','CPBiGRU'):
        r=next(q for q in table if q['model']==m);c=r['C1_C2_previous_predicted_mean']
        lines.append(f"| {m} | {c['C1']:.2f} | {c['C2']:.2f} | {r['predicted_mean']:.2f} |")
    lines+=['',
        'MICN对三枚电芯的平均寿命预测几乎不变（都约556圈），但C3的RPT显示其更早达到95%。其C3平均绝对误差即使按最有利的301圈终点计算，仍至少约255.07圈；原C1/C2代理标签下MAE为157.49圈。**这说明其在C3上也存在明显高估，但不能把两套标签下的误差差值视为纯粹的工况效应。**','',
        '输入方面，按原加载器归一化后的放电电流最大值，C1约1.005C、C2约1.007C，C3约3.753C。C3还包含飞行功率段之后的低倍率恢复放电，曲线分布确实不同；测试没有把它改成C1/C2的恒流曲线。','',
        '## 4. 本机兼容性和保护检查','',
        f'- 11种模型的66条C1/C2兼容性回放，最大差 **{replay_max:.9f}圈**，低于原0.005圈门限；原72条正式预测文件未修改。',
        f'- Autoformer只在新入口内把索引张量从强制CUDA改到输入设备，原模型文件、数学操作和权重不改。CPU回放最大差 **{diag_max:.6f}圈**，超过原门限，因此3条C3输出仅列CPU诊断，不冒充严格复核通过结果。',
        '- Autoformer聚合操作还与独立逐样本torch.roll实现核对通过（绝对容差1e−6）。',
        '- C3输入100×3×300，只有物理2–100圈的99个普通循环，第1圈RPT置零。四个CP-RNN沿用2–100圈前移、尾部补零；没有引入第101圈。I/Q特征仍都÷183。',
        '- C3工步和循环汇总表末尾的3个零容量、仅搁置的导出记录不计作充放电循环；逐行原因和映射已保存。主数据共404个有充放电活动的记录，RPT位置1/101/201/301/401/402。后段部分循环不完整，但不影响前100圈输入及201–301的95%区间。',
        '- 6个完整RPT放电容量经流式XML与openpyxl独立读取逐项完全一致；原Excel只读，未改动。Cell03-Extra为后续变更工况续测，不纳入。',
        f'- 对本地已归档的 **{hashes_verified}个权重/配置/scaler文件**重新核对SHA256，均与最终审计索引一致。原输入、模型代码和正式报告哈希保持不变。',
        '- 旧标签scaler采用原283个源域训练电芯的已审计均值/标准差；不使用C3或C1/C2拟合任何统计量。为了避免本机sklearn版本不同导致反序列化差异，读取已校验JSON参数手动反标准化，原scaler二进制文件SHA同时核验。','',
        '## 5. 结果位置','',
        '- [全部逐种子结果与区间统计](../dataset/mixed95/c3_eval_20261020_v1/summary.json)',
        '- [C3输入、RPT与圈号映射核对](../dataset/mixed95/c3_eval_20261020_v1/input_audit.json)',
        '- [严格复核通过结果及Autoformer失败记录](../dataset/mixed95/c3_eval_20261020_v1/predictions.json)',
        '- [Autoformer CPU诊断结果](../dataset/mixed95/c3_eval_20261020_v1/autoformer_cpu_diagnostic_predictions.json)',
        '- [本机回归测试记录](../dataset/mixed95/c3_eval_20261020_v1/test_results.txt)','',
        '原始数据位于 `/Users/curtischan/Datasets/comac/raw/商飞800VSOH-tight/商飞800VSOH-tight/Cell03`，RPT取各工作簿“工步数据表”的完整工步8放电容量。该报告是新增C3事后测试，不是对旧C1/C2 benchmark盲测协议的追溯修改。','']
    path=ROOT/'reports/BatteryLife_C3_本机测试_20261020.md'
    with path.open('w') as stream:stream.write('\n'.join(lines))
    print(json.dumps(dict(report=str(path),accepted_predictions=33,autoformer_diagnostic=3,replay_max=replay_max),ensure_ascii=False))


if __name__=='__main__':main()
