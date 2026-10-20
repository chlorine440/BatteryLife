# BatteryLife MIX_large → COMAC C1/C2, 95% SOH

The frozen 13-model extension is documented in [BENCHMARK.md](BENCHMARK.md).
It preserves the existing single-model entry points and audited results.

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
under `dataset/mixed95/comac_loader_v2` (the old sign-split `comac` artifacts
are preserved separately). The labels must be C1=379 and C2=419. These
are **ordinary-cycle proxy crossings, not full-RPT measured 95% lives**.
Only physical cycles 1–100 can enter the model. RPT cycle 1 is zero-filled and
masked. Target life labels use discharge Ah/(183×0.94 Ah), while target model
input uses the original feature normalization: current/183 and capacity/183.

### Original-loader COMAC adapter correction (2026-10-08)

The original sign-filtered COMAC adapter did not exactly reproduce the source
loader's phase endpoint convention. The mixed95 adapter now preserves full
chronological cycling records, including rests and both lifetime Ah counters,
converts the counters to per-cycle Ah, then calls the unchanged
`Dataset_original.get_charge_discharge_curves` for all phase selection,
resampling and normalization. In particular, the original charge-first method
excludes the final charging record from the charge segment and retains it at
the start of the discharge segment. This legacy convention is retained for
strict source/target consistency, not proposed as a new physical definition.
The original model, source curves, training code, source labels, checkpoints,
183 Ah feature scaling and 0.94 label correction are unchanged.

Run `python -m experiments.mixed95.evaluate prepare-comac` locally to create
**new** versioned artifacts in `dataset/mixed95/comac_loader_v2`. It refuses
to overwrite an existing v2 directory and does not alter the old `comac`
directory. Both inputs store `batterylife_original_loader_v2` metadata;
evaluation rejects old/unversioned artifacts. `input_provenance.json` records
the loader hash, input hashes, masked physical RPT cycle 1 and labels' formula.
Physical cycles 2–100 retain their original positions; no later cycle replaces
the gap. The C1/C2 proxy labels remain 379/419.

If a corrected evaluation is explicitly requested, reuse the already-selected
source checkpoints; no source retraining is required for this adapter fix:

```sh
python -m experiments.mixed95.evaluate evaluate --run-id paper95_20261008_v1
```

This now reads only `comac_loader_v2` and saves
`held_out_comac_evaluation_95_loader_v2.json` and `.csv`. Old
`held_out_comac_evaluation_95.json`/`.csv` results remain untouched. The v2
result is an input-implementation correction, not a new independent held-out
experiment; do not choose models or preprocessing from C1/C2 error changes.
The second v2 evaluation for the same run is refused. Regeneration of inputs
does not itself invoke model inference or train a model. Transfer the new
adapter code and **the new v2 directory**, not only the old `comac` artifacts,
before evaluating on the server.

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
`dataset/seen_unseen_labels`, and the small `dataset/mixed95/comac_loader_v2` artifacts.
Symlinks generated on this Mac
have absolute local targets, so rerun source preparation on the server with
its source path before training:

The current server exposes two 32 GiB GPUs with CUDA capability 8.9 and a
CUDA 11.8 toolkit, so the original Torch 2.4.1 version can be retained. Its
isolated environment is `dataset/.conda_envs/mixed95`. For a fresh environment:

```sh
source /root/miniconda3/etc/profile.d/conda.sh
conda create -y -p "$PWD/dataset/.conda_envs/mixed95" python=3.11 pip
conda activate "$PWD/dataset/.conda_envs/mixed95"
python -m pip install pip==25.2 setuptools==75.1.0 wheel
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --no-build-isolation -r experiments/mixed95/requirements-server.txt
python -m pip install --no-build-isolation deepspeed==0.15.0
python -m pip check
```

The published Ninja 1.11.1.1 and XGBoost 1.7.6 wheels have malformed or
inconsistent internal platform tags, so `pip check` can report these two as
unsupported despite successful installation. On the current Linux server,
Ninja execution, XGBoost import and the original training-entry imports were
verified separately; there are no missing or conflicting dependency versions.
Do not ignore additional dependency errors or CUDA failures.

The server requirements retain every original pinned training dependency;
BatteryML is fetched from the archived commit used locally, and missing or
otherwise unpinned support dependencies are recorded separately. This cu118
setup is for the current sm_89 hardware; RTX 5090 needs a newer CUDA-enabled
Torch build. Changing the installer source does not change the model or batch
configuration.

On 2026-10-08 the current two-sm_89-GPU server passed all 468 eligible
source-curve checks and the full-configuration one-epoch smoke run
`resource_check_20261008_v1`, including validation and checkpoint saving.
The model has 3,857,217 trainable parameters; training compute took 18.16 s,
while loading the three source splits dominated the approximately 9-minute
startup. Ten-second GPU samples reached 4,759 MiB on each GPU (a sampled
maximum, not an exact instantaneous peak). The smoke metrics are not final
experiment results and its weights were not evaluated on COMAC.

The full run `paper95_20261008_v1` was started in the server background.
Its wrapper trains all three seeds before calling the one-time COMAC
evaluation; errors stop the wrapper instead of changing the configuration.
On that server, inspect progress without starting another training process:

```sh
tail -f /root/autodl-tmp/BatteryLife/dataset/mixed95/setup/paper95_console_20261008.log
```

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
