# BatteryLife MIX_large → COMAC C1/C2, 95% SOH

This experiment keeps the original CPTransformer, source curves, cell splits,
feature processing, MSE training loss and validation-based checkpoint selection.
It changes the source life threshold to 95% and adapts COMAC Excel only for
held-out inference. It does not reuse the XJTU/eVTOL 95% weights in
`experiments/comac95` or any 80% weights.

## Data and labels

The preprocessed source files stay at `/Users/curtischan/Datasets/batterylife`.
The source-label script inspects each `MIX_large` cell, computes discharge Ah
divided by the dataset's nominal Ah and SOC span, and labels the first cycle
with SOH ≤95%. For a cell not observed to cross, it follows the paper's
20-cycle linear tail extrapolation only if terminal SOH is in (95%, 97.5%].
Cells with life ≤100, missing crossings, invalid data or failed extrapolation
are excluded. Every decision is recorded in `source_labels_audit_95.json`.
The generated JSON labels and symlinks are under ignored `dataset/mixed95/input`;
the original source files and old 80% labels are untouched.

The local preflight on 843 cells passed: 283 train, 89 validation and 96 test
cells have usable 95% labels. None required extrapolation; 369 crossed by
cycle 100 and 6 never reached the threshold. Although all 13 source folders
are present, only 8 contribute usable cells after early-input/life filtering.
This is not a 13-dataset *effective* training set.

On the repository root:

```sh
source dataset/.miniconda3/etc/profile.d/conda.sh
conda activate "$PWD/dataset/.conda_envs/comac95"
python -m unittest experiments.mixed95.test_mixed95 experiments.comac95.test_data
python -m experiments.mixed95.prepare
python -m experiments.mixed95.check_curves
python -m experiments.mixed95.evaluate prepare-comac
```

`prepare` prints and saves per-split/per-dataset coverage, including observed,
extrapolated, early-crossing and censored cells. It exits nonzero if a source
file cannot be read, a split has no usable label, training labels have no
variation or the source supplies fewer than 256 early-cycle windows. Do not
start training unless `preflight_95.json` says `passed: true`.
`check_curves` then tests every eligible cell with the unchanged BatteryLife
curve loader, saving any per-cell extraction error in `source_curves_audit_95.json`.

COMAC preparation saves the original ordinary-cycle capacities, the separate
full-discharge RPT values, cleaning issues, early input arrays and two QA plots
under `dataset/mixed95/comac`. The labels must be C1=379 and C2=419. These
are **ordinary-cycle proxy crossings, not full-RPT measured 95% lives**.
Only physical cycles 1–100 can enter the model. RPT cycle 1 is zero-filled and
masked. Target life labels use discharge Ah/(183×0.94 Ah), while target model
input uses the original feature normalization: current/183 and capacity/183.

## Original-paper training

The original trainer needs `accelerate`, `evaluate`, `wandb`, `peft`,
`denseweight`, `deepspeed`, `safetensors`, and the repository requirements in
the activated environment. `WANDB_MODE=offline` is set by the launcher so
COMAC/source details are not uploaded. The local Apple M5 environment currently
has no available CUDA or MPS backend. The listed dependencies except
`deepspeed` were installed locally to check imports and the data loader;
DeepSpeed is reserved for the CUDA server.
The launcher deliberately requires **two CUDA GPUs** to preserve the paper's
per-process batch size of 128 and effective batch size of 256. It will not
silently substitute a smaller model, lower batch size or CPU trainer.
The paper and repository do not name a required GPU model. The server must
provide a compatible Linux/CUDA PyTorch environment plus this repository's
`requirements.txt`; install BatteryML from the archived source as described
in [the existing experiment setup](../comac95/README.md), since the pinned
BatteryML package is not published on PyPI. Install `wandb` separately because
the original trainer imports it but it is missing from `requirements.txt`.

On a compatible two-GPU server, transfer the 13 processed source folders,
`dataset/seen_unseen_labels`, and the small `dataset/mixed95/comac` artifacts.
Symlinks generated on this Mac
have absolute local targets, so rerun source preparation on the server with
its source path before training:

```sh
python -m experiments.mixed95.prepare --source /server/path/batterylife
python -m experiments.mixed95.check_curves
python -m experiments.mixed95.train --smoke --run-id resource_check
python -m experiments.mixed95.train --run-id paper95_v1
python -m experiments.mixed95.evaluate evaluate --run-id paper95_v1
```

The smoke run is one epoch with the full model/data batch configuration; its
weights cannot be tested on COMAC. The full run trains seeds 2021, 42 and 2024
with the paper's Li-ion CPTransformer dimensions (256/64/1/12), learning rate
5e-5 and the repository's remaining CPTransformer settings. The source
validation set picks each checkpoint. Source-test metrics are captured from
the original trainer log; its metrics aggregate early windows 1–100. COMAC
evaluation uses the cycle-100 input only and refuses a second evaluation for
the same run ID. Each seed produces one C1 and one C2 prediction.

The experiment launcher never overwrites an existing run ID because the
original trainer deletes pre-existing checkpoint contents. Use a new run ID
for a fresh run. Do not use C1/C2 errors to select seeds, hyperparameters,
checkpoints or preprocessing variants.
