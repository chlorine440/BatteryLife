"""Post-run independent verification; never trains or changes existing results.

python -m experiments.mixed95.c3_che_v2_verify --suite-id c3_che_20261026_v2
Writes a separate verification record and regression log exactly once.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys

import numpy as np

from experiments.mixed95.c3_che_data import ROOT, OUT as V1, read, save
from experiments.mixed95.c3_che_v2_data import (
    DEFAULT_SUITE, SEEDS, SOURCE, TRAIN_CELLS, VAL_CELL, load_candidates,
)
from experiments.mixed95.c3_che_v2 import suite_path, audit
from experiments.mixed95.c3_evaluate import sha


def verify(out):
    if (out/'verification.json').exists() or (out/'regression_tests.txt').exists():
        raise FileExistsError('verification already recorded; do not overwrite its evidence')
    checks=audit(out);manifest=read(out/'run_manifest.json')
    for relative,expected in manifest['artifact_sha256'].items():
        if sha(out/relative)!=expected:raise ValueError(f'artifact changed: {relative}')
    report=Path(manifest['report_path'])
    if sha(report)!=manifest['report_sha256']:raise ValueError('report changed')
    arrays,meta=load_candidates(out)
    if meta['rpt']!=read(V1/'target_features_audit.json')['rpt']:
        raise ValueError('v2 RPT labels differ from audited source RPT table')
    full_discharge_checks=[]
    for c,name in enumerate((*TRAIN_CELLS,VAL_CELL)):
        valid=np.isfinite(arrays[f'c{c}_x']).any(axis=1)
        used=set(arrays[f'c{c}_cycles'][valid].tolist());checked=0
        with (SOURCE/name).open('rb') as handle:source=pickle.load(handle)
        for row in source['cycle_data']:
            if int(row['cycle_number']) not in used:continue
            current=np.asarray(row['current_in_A']);voltage=np.asarray(row['voltage_in_V'])
            discharge=np.flatnonzero(current<-.01*source['nominal_capacity_in_Ah'])
            if not len(discharge) or abs(voltage[discharge[-1]]-3.)>.005:
                raise ValueError('accepted source capacity is not a full discharge to 3.0 V')
            checked+=1
        if checked!=len(used):raise ValueError('accepted source cycle not found in raw pkl')
        full_discharge_checks.append(dict(cell=name,checked_cycles=checked,full_discharge_to_3V=True))
    r=read(out/'reconstruction.json')
    with (out/'C3_reconstructed_soh.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    if [int(q['physical_cycle']) for q in rows]!=list(range(1,405)):raise ValueError('CSV cycle mapping mismatch')
    for json_row,csv_row in zip(r['rows'],rows):
        for key,value in json_row.items():
            if isinstance(value,float):
                if abs(float(csv_row[key])-value)>1e-12:raise ValueError(f'CSV/JSON differs: {key}')
            elif value is None and csv_row[key]!='':raise ValueError('missing measurement fabricated in CSV')
    with (out/'heldout_rpt.csv').open(newline='') as stream:held=list(csv.DictReader(stream))
    metric_cross_check={}
    for seed in SEEDS:
        q=[x for x in held if int(x['seed'])==seed and x['interior']=='True']
        if [int(x['cycle']) for x in q]!=[101,201,301]:raise ValueError('common comparison not the three interior points')
        errors=[float(x['predicted_soh'])-float(x['measured_soh']) for x in q]
        actual=dict(mae_pp=100*math.fsum(abs(e) for e in errors)/3,
                    rmse_pp=100*math.sqrt(math.fsum(e*e for e in errors)/3))
        stored=next(x for x in r['per_seed'] if x['seed']==seed)['common_interior']['predicted_soh']
        if any(abs(actual[key]-stored[key])>1e-10 for key in actual):raise ValueError('independent interior metrics differ')
        metric_cross_check[str(seed)]=actual
    env=dict(os.environ)
    env.update(MPLCONFIGDIR=str(out/'verify_mplcache'),XDG_CACHE_HOME=str(out/'verify_cache'),
               TRANSFORMERS_CACHE=str(out/'verify_cache/huggingface'))
    command=[sys.executable,'-m','unittest','discover','-s','experiments/mixed95','-p','test_*.py']
    run=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
    log=run.stdout+run.stderr
    with (out/'regression_tests.txt').open('x') as stream:stream.write(log)
    if run.returncode:raise ValueError('regression failed; inspect saved test log')
    count=re.search(r'Ran (\d+) tests?',log);skips=re.search(r'OK \(skipped=(\d+)\)',log)
    source_logs=list(out.glob('*/source/h*/seed_*/training.json'))
    training=[read(p) for p in source_logs]
    if len(training)!=216:raise ValueError('source grid incomplete')
    supplementary=[ROOT/'experiments/mixed95'/n for n in
                   ('c3_che_v2_verify.py','test_c3_che_v2_regression.py')]
    result=dict(status='passed',checks=checks,tests_run=int(count.group(1)),
        tests_skipped=int(skips.group(1)) if skips else 0,test_exit_code=run.returncode,
        test_command=command,regression_log_sha256=sha(out/'regression_tests.txt'),
        completed_source_runs=len(training),nonconverged_source_runs=sum(not q['converged'] for q in training),
        total_source_training_seconds=sum(q['elapsed_seconds'] for q in training),
        csv_json_match=True,RPT_labels_match_immutable_v1_audit=True,
        source_full_discharge_label_checks=full_discharge_checks,
        independent_interior_metrics=metric_cross_check,
        frozen_artifact_files_checked=len(manifest['artifact_sha256']),
        supplementary_verification_code_sha256={str(p.relative_to(ROOT)):sha(p) for p in supplementary},
        limitation='Passing checks does not establish unknown between-RPT SOH truth or remove process-setpoint confounding.')
    save(out/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--suite-id',default=DEFAULT_SUITE)
    args=p.parse_args();verify(suite_path(args.suite_id))
