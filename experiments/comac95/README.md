# XJTU → COMAC C1/C2: 95% cycle life

This is a separate CPTransformer experiment. It does not overwrite BatteryLife's
existing 80% life labels or use C1/C2 to train, fine-tune, or select a checkpoint.
It requires Python 3.11+ and the numerical/model dependencies used below,
including BatteryML, PyTorch, SciPy, pandas and matplotlib. The source files are
read from `/Users/curtischan/Datasets/comac/raw` by default; use `--raw-root` to
change this. All generated files go to the ignored `dataset/comac95/` folder by
default; use `--output` to change it.

On this Mac, an isolated Miniconda installation and Python 3.11 environment
were created under the ignored `dataset/.miniconda3` and
`dataset/.conda_envs/comac95` directories. Activate it from the repository
root with:

```sh
source dataset/.miniconda3/etc/profile.d/conda.sh
conda activate "$PWD/dataset/.conda_envs/comac95"
```

BatteryML 0.0.1 is not published on PyPI; this environment installs it from
[Microsoft's archived BatteryML repository](https://github.com/microsoft/BatteryML)
at commit `2861ae3b8c79938c7fc8e6fe9986b799ca71c7dd`. On this macOS 27
machine the PyPI SciPy 1.15.2 wheel failed to load, so SciPy and NumPy 1.26.4
were installed from conda-forge. Other experiment dependencies were installed
in the environment, not in the system Python.

From the repository root:

```sh
python -m unittest experiments.comac95.test_data
# Optional: extract and check COMAC labels with Python's standard library only.
python -m experiments.comac95.data
python -m experiments.comac95.run prepare
python -m experiments.comac95.run train --epochs 100 --seed 2024
python -m experiments.comac95.run evaluate
```

The `prepare` stage uses the repository's existing XJTU cycle-organizing and
capacity-splitting functions on the local Batch-1/2 raw MAT files. If processed
XJTU pickle files in BatteryLife format are already available, pass
`--xjtu-processed /path/to/XJTU` to reuse them. It writes
`XJTU_labels_95.json` and `xjtu_eligibility_95.json` separately from the 80%
labels. It refuses to train if there are fewer than 5 eligible train cells, 2
eligible validation cells, or 1 eligible source test cell. A crossing at or
before the first 100 cycles, a missing crossing, or fewer than 90 valid early
cycles makes a source cell ineligible; exclusions are recorded.

The first local preflight found only **1/15 train, 1/4 validation, and 1/4 test**
eligible XJTU cells: 20 of 23 crossed at their first preprocessed cycle under
the stipulated discharge-Ah / 2.0-Ah definition. Consequently `prepare` stops
before COMAC feature extraction and no model is trained. See
`dataset/comac95/xjtu_eligibility_95.json` for the per-cell audit. Do not use
C1/C2 for training or silently replace the stipulated threshold/denominator.

The label is the first cycle with SOH <= 0.95. XJTU uses full-discharge Ah /
2.0 Ah. COMAC uses ordinary-cycle discharge Ah / (183 Ah × 0.94), omitting the
five RPT cycles. Its 95% proxy crossings must be C1 cycle 379 and C2 cycle
419. Those crossings are **not full-RPT measurements**. RPT step-8 discharge
Ah / 183 Ah is saved independently and shown on the QA plots. The target
model inputs use only physical cycles 1–100; cycle 1 is RPT and is masked.
No post-100 target curve is used for inference.

`evaluation_95.csv` contains source-test and target cell-level predictions,
absolute error and APE. The XJTU test result is an in-domain check. C1/C2
errors are against the proxy threshold labels, and results should not be
reported as errors against measured full-capacity 95% life.
