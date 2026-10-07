"""Run the repository's original trainer with the paper's Li-ion CPTransformer setup.

The full run intentionally requires two CUDA GPUs, as in the published
hyperparameter table. A one-epoch --smoke run checks memory/runtime without
producing a result that may be evaluated on C1/C2.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from experiments.mixed95.prepare import DEFAULT_OUTPUT, ROOT


SEEDS = (2021, 42, 2024)
MODEL_ARGS = {
    "model": "CPTransformer", "model_id": "CPTransformer", "dataset": "MIX_large",
    "data": "Dataset_original", "task_name": "classification", "is_training": 1,
    "features": "MS", "seq_len": 1, "label_len": 50, "factor": 3,
    "enc_in": 3, "dec_in": 1, "c_out": 1, "des": "Exp", "itr": 1,
    "d_model": 256, "d_ff": 64, "batch_size": 128, "learning_rate": 5e-5,
    "train_epochs": 100, "accumulation_steps": 1,
    "charge_discharge_length": 300, "num_workers": 32,
    "e_layers": 1, "lstm_layers": 6, "d_layers": 12,
    "patience": 5, "n_heads": 4, "early_cycle_threshold": 100,
    "dropout": 0, "lradj": "constant", "loss": "MSE",
    "model_comment": "mixed95_paper",
}


def _check_environment() -> None:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError(
            "the paper configuration needs two CUDA GPUs; this host has "
            f"{torch.cuda.device_count()}. No smaller or CPU training was substituted."
        )
    missing = [name for name in ("accelerate", "evaluate", "wandb", "peft",
                                     "denseweight", "deepspeed", "safetensors")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"missing training dependencies: {', '.join(missing)}")


def _source_metrics(log_text: str) -> dict[str, float]:
    match = re.search(
        r"Best model performance: Test MAE: ([0-9.eE+-]+) \| Test RMSE: ([0-9.eE+-]+) "
        r"\| Test MAPE: ([0-9.eE+-]+) \| Test 15%-accuracy: ([0-9.eE+-]+) "
        r"\| Test 10%-accuracy: ([0-9.eE+-]+)", log_text
    )
    if not match:
        raise RuntimeError("original trainer did not report source-test metrics")
    keys = ("mae_cycles", "rmse_cycles", "mape", "acc15", "acc10")
    return dict(zip(keys, map(float, match.groups())))


def train(output: Path = DEFAULT_OUTPUT, *, run_id: str = "paper95_v1", smoke: bool = False) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run-id must contain only letters, digits, underscore or hyphen")
    summary_path = output / "preflight_95.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"prepare source labels first: {summary_path}")
    preflight = json.loads(summary_path.read_text(encoding="utf-8"))
    if not preflight.get("passed"):
        raise RuntimeError("source 95% preflight did not pass")
    curve_audit = output / "source_curves_audit_95.json"
    if not curve_audit.is_file() or not json.loads(curve_audit.read_text(encoding="utf-8")).get("passed"):
        raise RuntimeError("run the original source-curve check before training")
    _check_environment()

    experiment_root = output / ("smoke" if smoke else "runs") / run_id
    if experiment_root.exists():
        raise FileExistsError(f"refusing to overwrite an existing run: {experiment_root}")
    experiment_root.mkdir(parents=True)
    manifest = {"run_id": run_id, "smoke_only": smoke,
                "source_input_root": preflight["input_root"], "seeds": {},
                "model_args": MODEL_ARGS}
    manifest_path = experiment_root / "manifest.json"
    for seed in ((2021,) if smoke else SEEDS):
        checkpoint_root = experiment_root / f"seed_{seed}"
        args = dict(MODEL_ARGS, root_path=preflight["input_root"],
                    checkpoints=str(checkpoint_root), seed=seed)
        if smoke:
            args["train_epochs"] = 1
            args["model_comment"] = "mixed95_smoke_not_for_test"
        command = [sys.executable, "-m", "accelerate.commands.launch", "--multi_gpu",
                   "--num_processes", "2", "--main_process_port", "25216",
                   str(ROOT / "run_main.py")]
        for key, value in args.items():
            command.extend((f"--{key}", str(value)))
        log_path = experiment_root / f"seed_{seed}.log"
        env = os.environ.copy()
        env["WANDB_MODE"] = "offline"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, env=env)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            status = process.wait()
        if status:
            manifest["seeds"][str(seed)] = {"status": "failed", "exit_code": status,
                                             "log": str(log_path)}
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            raise RuntimeError(f"seed {seed} failed; inspect {log_path}")
        candidates = list(checkpoint_root.glob("*/args.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"expected one checkpoint under {checkpoint_root}, found {len(candidates)}")
        record = {"status": "complete", "checkpoint": str(candidates[0].parent),
                  "log": str(log_path)}
        if not smoke:
            record["source_test"] = _source_metrics(log_path.read_text(encoding="utf-8"))
        manifest["seeds"][str(seed)] = record
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default="paper95_v1")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    train(args.output, run_id=args.run_id, smoke=args.smoke)


if __name__ == "__main__":
    main()
