# eVTOL → COMAC C1/C2: exploratory 95% life experiment

This is an alternative source-data experiment to the XJTU plan in [README.md](README.md).
It uses the same CPTransformer architecture and COMAC target adapter, but **trains only
on the 22 local eVTOL `VAH*.csv` cells**. C1/C2 are not used for training,
fine-tuning, normalization, checkpoint selection, or model changes. The original
80% experiments and XJTU 95% files are left untouched.

From the repository root, with the Python 3.11 environment described in
[README.md](README.md) activated:

```sh
python -m unittest experiments.comac95.test_data experiments.comac95.test_evtol
python -m experiments.comac95.evtol prepare
python -m experiments.comac95.evtol train --epochs 100 --batch-size 32 --seed 2024
python -m experiments.comac95.evtol evaluate
```

`--raw-root` defaults to `/Users/curtischan/Datasets/comac/raw`; `--output`
defaults to ignored `dataset/comac95/`. Preparation reads the raw eVTOL CSVs
and COMAC Excel files without modifying them. It writes eVTOL cell/RPT audits,
split arrays, COMAC proxy/RPT audits, and early-cycle features into the ignored
output directory. The weights and predictions are likewise separate from XJTU:
`cptransformer_evtol_95.pt`, `training_log_evtol_95.json`,
`evaluation_evtol_95.json`, and `evaluation_evtol_95.csv`.

The eVTOL CSV `cycleNumber` resets around reference performance tests (RPTs).
The adapter first segments chronological contiguous runs, identifies complete
C/5 full-discharge RPTs, and counts valid mission runs. The source label is the
first *observed RPT check* at or below 95% of 3.0 Ah. This is **interval-censored**:
the actual crossing occurred sometime after the preceding above-95% RPT and no
later than the labeled check. Ordinary mission discharge is partial and is not
used as a full-capacity label. The RPTs are excluded from early model curves;
their physical positions remain masked rather than replaced with later curves.

The 22 cells are stratified by observed label into 12 train, 5 validation, and
5 test cells using seed 2024. Only the training labels set target mean/standard
deviation; validation selects the checkpoint. The eVTOL test split and C1/C2
are evaluated after selection. The label values in the local data are only
150, 200, or 250 cycles, so this is a small, coarse-label experiment, not a
precise continuous-life ground truth.

COMAC labels remain the previously agreed ordinary-cycle proxy:
discharge Ah / (183 Ah × 0.94), giving first 95% crossings of C1 cycle 379 and
C2 cycle 419. They are **not full-RPT measured crossings**. COMAC and eVTOL use
different cells and operating profiles, and the source measured-RPT versus
target proxy label distinction is a serious domain/label shift. Interpret
cross-domain errors accordingly; do not use C1/C2 outcomes to pick another
checkpoint or tune this experiment.
