"""Separate, frozen-checkpoint C3 inference; preserve the original benchmark.

No training and no daily-capacity proxy label for C3. The threshold is interval
censored between full RPTs. Empty final rest-only export rows are not cycles.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "dataset/mixed95/benchmarks/benchmark95_20261013_autoformer_retry_v1"
RAW = Path("/Users/curtischan/Datasets/comac/raw/商飞800VSOH-tight/商飞800VSOH-tight")
OUTPUT = ROOT / "dataset/mixed95/c3_eval_20261020_v1"
BOOKS = ("C3-100-1.xlsx", "C3-200-1.xlsx", "C3-300-1.xlsx", "C3-400.xlsx", "C3-500-1.xlsx", "C3-500-2.xlsx")


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def interval_errors(prediction, last_above, first_below, observation=100):
    """Conditional on monotone crossing between adjacent observed full RPTs.

    Discrete possible first crossing is last_above+1 .. first_below inclusive.
    Min/max metrics are bounds, not errors against an observed exact label.
    """
    possible = np.arange(last_above+1, first_below+1, dtype=np.float64)
    assert np.isfinite(prediction) and last_above >= observation and len(possible)
    absolute = np.abs(prediction-possible)
    total_rel = absolute/possible
    rul_rel = absolute/(possible-observation)
    return dict(absolute_error_cycles_min=float(absolute.min()), absolute_error_cycles_max=float(absolute.max()),
                total_life_ape_percent_min=float(100*total_rel.min()), total_life_ape_percent_max=float(100*total_rel.max()),
                rul_ape_percent_min=float(100*rul_rel.min()), rul_ape_percent_max=float(100*rul_rel.max()),
                total_life_15pct_possible=bool((total_rel<=.15).any()), total_life_15pct_guaranteed=bool((total_rel<=.15).all()))


def rows_named(book, sheet):
    rows = iter(book.rows(sheet))
    columns = {str(v).strip(): k for k,v in next(rows).items()}
    for row in rows:
        yield {name:row.get(k) for name,k in columns.items()}


def capacity_audit(raw):
    from experiments.comac95.data import XlsxRows, _number
    mapping, summaries, rpt, excluded = [], [], [], []
    cursor, previous_end = 0, -math.inf
    for name in BOOKS:
        path=raw/"Cell03"/name
        with XlsxRows(path) as book:
            grouped=defaultdict(list)
            for row in rows_named(book,"工步数据表"):
                local=_number(row.get("循环号"))
                if local is not None:grouped[int(local)].append(row)
            by_local={int(float(r["循环序号"])):r for r in rows_named(book,"循环数据表")}
        for local,steps in sorted(grouped.items()):
            active=[r for r in steps if (_number(r.get("容量(Ah)")) or 0)>0
                    and "搁置" not in str(r.get("工步名称"))]
            if not active:
                excluded.append(dict(file=name,local_cycle=local,reason="rest_only_zero_capacity_export_tail"))
                continue
            start=min(float(r["绝对时间"]) for r in steps)
            end=max(float(r["绝对时间"]) for r in steps)
            if start<=previous_end:raise ValueError("overlap or unordered C3 summary books")
            previous_end=end;cursor+=1
            mapping.append(dict(source_file=name,local_cycle=local,physical_cycle=cursor,
                                first_step_end_excel=start,last_step_end_excel=end))
            full=[r for r in steps if _number(r.get("工步号"))==8 and (_number(r.get("容量(Ah)")) or 0)>100]
            calibrated=any(_number(r.get("工步号"))==6 and (_number(r.get("容量(Ah)")) or 0)>100 for r in steps)
            if calibrated:
                if len(full)!=1 or abs(float(full[0]['终止电压(V)'])-2.5)>.02:
                    raise ValueError("RPT must include unique complete step8 discharge to 2.5V")
                q=float(full[0]["容量(Ah)"])
                rpt.append(dict(cycle=cursor,discharge_ah=q,soh=q/183,source_file=name,source_local_cycle=local,
                                label_source="RPT_full_discharge_div_nominal183"))
            summary=by_local[local]
            summaries.append(dict(cycle=cursor,source_file=name,source_local_cycle=local,
                charge_ah=float(summary['充电容量(Ah)']),discharge_ah=float(summary['放电容量(Ah)']),
                is_rpt=calibrated,used_as_daily_soh_label=False))
    assert len(mapping)==404 and len(excluded)==3
    # Independent XLSX implementation verifies actual RPT row contents.
    import openpyxl
    independent=[]
    for name in BOOKS:
        book=openpyxl.load_workbook(raw/'Cell03'/name,read_only=True,data_only=True)
        try:
            rows=iter(book['工步数据表'].iter_rows(values_only=True));head=next(rows)
            col={str(v).strip():i for i,v in enumerate(head) if v is not None}
            for row in rows:
                if _number(row[col['工步号']])==8 and (_number(row[col['容量(Ah)']]) or 0)>100:
                    independent.append((name,int(row[col['循环号']]),float(row[col['容量(Ah)']])))
        finally:book.close()
    expected=[(r['source_file'],r['source_local_cycle'],r['discharge_ah']) for r in rpt]
    assert independent==expected
    below=next(r for r in rpt if r['soh']<=.95)
    prior=max((r for r in rpt if r['cycle']<below['cycle']),key=lambda r:r['cycle'])
    assert prior['soh']>.95
    return dict(mapping=mapping,summary_rows=summaries,rpt=rpt,excluded=excluded,
        label=dict(exact_life_cycle=None,last_rpt_above95=prior,first_rpt_at_or_below95=below,
                   integer_crossing_range_assuming_monotone=[prior['cycle']+1,below['cycle']],
                   first_observed_rpt95_cycle=below['cycle'],no_daily_div094_label=True,
                   caveat="Interval requires monotone underlying crossing between RPTs; no observed exact crossing label"),
        independent_rpt_openpyxl_exact=True,extra_excluded="Cell03-Extra is same cell later changed-condition continuation; not needed for first95 RPT bracket")


def prepare(raw,out):
    from experiments.comac95.data import XlsxRows, _number
    from experiments.mixed95.comac_input import original_loader_features, PREPROCESSING_VERSION
    if (out/'input_audit.json').exists() or (out/'C3_input.npz').exists():
        raise FileExistsError('prepared artifacts already exist; choose a new directory')
    audit=capacity_audit(raw)
    paths=[raw/'Cell03'/f for f in BOOKS]+[raw/'Cell03/C3-100-2.xlsx']
    protected=[SUITE/'report.json',SUITE/'manifest.json',ROOT/'experiments/mixed95/benchmark_baseline.json',
               ROOT/'experiments/mixed95/comac_input.py',ROOT/'layers/AutoCorrelation.py',
               ROOT/'dataset/mixed95/comac_loader_v2/C1_input.npz',ROOT/'dataset/mixed95/comac_loader_v2/C2_input.npz']
    hashes={str(p):sha(p) for p in paths+protected}
    cycles=defaultdict(list);duplicates={};parsed=[]
    fields=('绝对时间','电压(V)','电流(A)','累计充电容量(Ah)','累计放电容量(Ah)')
    for path in [raw/'Cell03/C3-100-1.xlsx',raw/'Cell03/C3-100-2.xlsx']:
        with XlsxRows(path) as book:
            for sheet in book.sheets:
                if not sheet.startswith('详细数据'):continue
                rows=iter(book.rows(sheet));col={str(v).strip():k for k,v in next(rows).items()}
                kept=0
                for row in rows:
                    n=_number(row.get(col['循环序号']))
                    if n is None or not 2<=n<=100:continue
                    assert n==int(n)
                    point=tuple(_number(row.get(col[k])) for k in fields)
                    if any(v is None for v in point):raise ValueError(f'invalid record {path.name}/{sheet}/{n}')
                    cycles[int(n)].append(point);kept+=1
                parsed.append(dict(file=path.name,sheet=sheet,kept=kept))
                print(json.dumps(parsed[-1]),flush=True)
    cleaned={}
    for n,points in cycles.items():
        unique=sorted(dict.fromkeys(points),key=lambda p:p[0]);duplicates[n]=len(points)-len(unique);cleaned[n]=unique
    assert set(cleaned)==set(range(2,101))
    x,mask=original_loader_features(cleaned)
    np.savez_compressed(out/'C3_input.npz',features=x,mask=mask,physical_cycles=np.arange(1,101),
                        preprocessing_version=np.array(PREPROCESSING_VERSION))
    stats={}
    for n,points in cleaned.items():
        a=np.asarray(points);negative=a[:,2][a[:,2]<-1.83]
        stats[n]=dict(records=len(points),duplicate_records=duplicates[n],charge_current_peak_C=float(a[:,2].max()/183),
                      discharge_current_peak_C=float(-negative.min()/183), discharge_current_median_C=float(-np.median(negative)/183))
    audit.update(input_shape=list(x.shape),input_dtype=str(x.dtype),mask=mask.tolist(),physical_cycle_by_slot=list(range(1,101)),
        no_input_beyond100=True,parsed_sheets=parsed,cycle_stats=stats,
        input_capacity_current_denominator_Ah=183,feature_algorithm='unchanged original_loader_features',
        protected_input_sha256=hashes,input_sha256=sha(out/'C3_input.npz'),duplicates_removed=sum(duplicates.values()))
    assert hashes=={str(p):sha(p) for p in paths+protected}
    dump(out/'input_audit.json',audit)
    print(json.dumps(dict(prepared=True,rpt=audit['rpt'],label=audit['label'],duplicates=audit['duplicates_removed'])),flush=True)


def cpu_autoformer_aggregation(self,values,corr):
    """Original inference aggregation; ONLY placement of integer arange differs.

    Kept outside original layers to preserve all benchmark code/checkpoint hashes.
    A frozen C1/C2 diagnostic replay must agree before accepting C3 inference.
    """
    import torch
    batch,head,channel,length=values.shape
    init_index=torch.arange(length,device=values.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch,head,channel,1)
    top_k=int(self.factor*math.log(length))
    mean_value=torch.mean(torch.mean(corr,dim=1),dim=1)
    weights,delay=torch.topk(mean_value,top_k,dim=-1)
    tmp_corr=torch.softmax(weights,dim=-1)
    tmp_values=values.repeat(1,1,1,2)
    delays_agg=torch.zeros_like(values).float()
    for i in range(top_k):
        tmp_delay=init_index+delay[:,i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,head,channel,length)
        pattern=torch.gather(tmp_values,dim=-1,index=tmp_delay)
        delays_agg=delays_agg+pattern*(tmp_corr[:,i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,head,channel,length))
    return delays_agg


def evaluate(out,autoformer_diagnostic=False):
    import torch
    from safetensors.torch import load_file
    from experiments.mixed95.benchmark_config import instantiate, SEEDS
    from experiments.mixed95.benchmark_inference import adapt_input,configure_inference_backend
    from experiments.mixed95.comac_input import load_target_input
    torch.set_num_threads(4)
    configure_inference_backend()
    frozen=read(SUITE/'report.json');baseline=read(ROOT/'experiments/mixed95/benchmark_baseline.json')
    for name,expected_hash in baseline['core_sha256'].items():
        if name.startswith(('models/','layers/')) and sha(ROOT/name)!=expected_hash:
            raise ValueError(f'original model code changed: {name}')
    index=read(SUITE/'final_audit_20261013_v1/local_checkpoint_integrity.json')['files_index']
    audit=read(out/'input_audit.json')
    assert all(sha(Path(p))==v for p,v in audit['protected_input_sha256'].items())
    features,mask=load_target_input(out/'C3_input.npz')
    results=[]
    for model_name,record in frozen['models'].items():
        if autoformer_diagnostic and model_name!='Autoformer':continue
        if any(r['status']!='complete' for r in record['seed_status'].values()):
            dump(out/f'{model_name}_skipped.json',dict(model=model_name,status='not_evaluated',reason='original source audit did not pass for all seeds'))
            continue
        for seed in SEEDS:
            suffix='_cpu_diagnostic' if autoformer_diagnostic else ''
            destination=out/f'{model_name}_seed_{seed}{suffix}.json'
            if destination.exists():
                results.append(read(destination));continue
            start=time.monotonic();entry=dict(model=model_name,seed=seed,status='running',device='cpu',precision='float32')
            print(json.dumps(entry),flush=True)
            files=[f for f in index if f['model']==model_name and f['seed']==seed]
            args_path=next(ROOT/f['local_path'] for f in files if f['local_path'].endswith('/args.json'))
            folder=args_path.parent
            for f in files:
                path=ROOT/f['local_path']
                if sha(path)!=f['sha256']:raise ValueError(f'checkpoint hash mismatch: {path}')
            cfg=read(args_path)
            expected=record['seed_status'][str(seed)].get('args',dict(baseline['configs'][model_name],seed=seed))
            for k,v in expected.items():
                if k not in {'root_path','checkpoints','model_comment'} and cfg.get(k)!=v:
                    raise ValueError(f'config mismatch {model_name}/{seed}/{k}')
            net=instantiate(cfg).float().eval()
            net.load_state_dict(load_file(str(folder/'model.safetensors'),device='cpu'),strict=True)
            if model_name=='Autoformer':
                from layers.AutoCorrelation import AutoCorrelation
                adapted=0
                for layer in net.modules():
                    if isinstance(layer,AutoCorrelation):
                        layer.time_delay_agg_inference=types.MethodType(cpu_autoformer_aggregation,layer);adapted+=1
                assert adapted==cfg['e_layers']
                entry['cpu_compatibility_adapter']='arange CUDA placement only; original layer file unchanged'
            # Use the frozen training-only scaler numbers; avoid sklearn 1.4 vs
            # local1.9 unpickling. The exact original scaler blob hash is checked.
            assert next(f['sha256'] for f in files if f['local_path'].endswith('/label_scaler'))=='534ecde69bf48c82460a2c6c0eb178655c608415f2e6a6addfba4fd1bc1e1a6d'
            mu,sigma=baseline['scaler']['mean'],baseline['scaler']['scale']
            def predict(x,m):
                xx,mm,_=adapt_input(model_name,x,m,np.arange(1,101))
                with torch.inference_mode():z=net(torch.from_numpy(xx[None]),torch.from_numpy(mm[None]))
                if z.shape!=(1,1) or not torch.isfinite(z).all():raise ValueError('nonfinite scalar prediction')
                return float(z.item())*sigma+mu
            replay=[]
            for cell in ('C1','C2'):
                xx,mm=load_target_input(ROOT/f'dataset/mixed95/comac_loader_v2/{cell}_input.npz')
                actual=predict(xx,mm)
                original=next(r['predicted_life_cycle'] for r in frozen['predictions'] if r['model']==model_name and r['seed']==seed and r['cell']==cell)
                replay.append(dict(cell=cell,predicted=actual,saved_prediction=original,delta=actual-original))
            tolerance=.005
            entry.update(replay=replay,replay_tolerance_cycles=tolerance,
                         checkpoint=str(folder.relative_to(ROOT)),checkpoint_sha256=sha(folder/'model.safetensors'),
                         scaler=dict(mean=mu,scale=sigma,source='frozen original 283 source training cells'))
            replay_passed=max(abs(r['delta']) for r in replay)<=tolerance
            entry['strict_cpu_replay_passed']=replay_passed
            if not replay_passed and not autoformer_diagnostic:
                entry.update(status='cpu_replay_failed',elapsed_seconds=time.monotonic()-start)
                dump(destination,entry);results.append(entry);del net;continue
            p=predict(features,mask)
            a=audit['label']['last_rpt_above95']['cycle'];b=audit['label']['first_rpt_at_or_below95']['cycle']
            _,mm,mapping=adapt_input(model_name,features,mask,np.arange(1,101))
            entry.update(status='diagnostic_only_cpu_replay_outside_tolerance' if not replay_passed else 'complete',
                predicted_life_cycle=p,predicted_rul_at100=p-100,
                exact_label_life_cycle=None,rpt_first_observed95_cycle=b,
                interval_errors_assuming_monotone=interval_errors(p,a,b),
                physical_cycle_by_slot=mapping.tolist(),mask=mm.tolist(),
                elapsed_seconds=time.monotonic()-start)
            dump(destination,entry);results.append(entry);del net
            print(json.dumps({k:entry[k] for k in ('model','seed','status','predicted_life_cycle','elapsed_seconds')}),flush=True)
    assert all(sha(Path(p))==v for p,v in audit['protected_input_sha256'].items())
    destination='autoformer_cpu_diagnostic_predictions.json' if autoformer_diagnostic else 'predictions.json'
    dump(out/destination,dict(no_training=True,only_prefix100=True,results=results,
        original_report_sha256=sha(SUITE/'report.json'),no_exact_c3_label=True))


def sync_list(out):
    """Only source/config text; exclude credentials, data, weights, reports, git."""
    tracked=subprocess.check_output(['git','ls-files','-co','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    allowed={'.py','.sh','.md','.json','.txt','.yaml','.yml','.toml','.ini','.cfg'}
    paths=sorted({p for p in tracked if p and Path(p).parts[0] not in {'dataset','reports','results','checkpoints','logs'}
                  and (not Path(p).parts[0].startswith('.') or Path(p).parts[0] in {'.github','.gitignore'})
                  and (Path(p).suffix in allowed or p=='.gitignore') and (ROOT/p).is_file() and not (ROOT/p).is_symlink()})
    (out/'code_files.txt').write_text('\n'.join(paths)+'\n')
    dump(out/'code_manifest.json',{p:sha(ROOT/p) for p in paths})
    print(json.dumps(dict(sync_files=len(paths))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','evaluate','autoformer-diagnostic','sync-list'))
    parser.add_argument('--raw',type=Path,default=RAW)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.command=='prepare':prepare(args.raw,args.output)
    elif args.command=='evaluate':evaluate(args.output)
    elif args.command=='autoformer-diagnostic':evaluate(args.output,autoformer_diagnostic=True)
    else:sync_list(args.output)


if __name__=='__main__':main()
