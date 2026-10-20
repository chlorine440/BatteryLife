"""Frozen 13-model mixed95 suite; original trainer is launched unchanged.

dry-run creates only configuration/status records. smoke/train require the
audited data and two idle CUDA GPUs. Failed jobs are never retried implicitly.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from experiments.mixed95.benchmark_config import (
    ROOT, MODELS, SEEDS, canonical_hash, cli_args, provenance, resolve_config, sha256,
)
from experiments.mixed95.benchmark_inference import (
    METRICS, compare_metrics, load_checkpoint, metrics, reload_probe, source_evaluation,
    target_evaluation, validate_scaler,
)
from experiments.mixed95.prepare import DEFAULT_OUTPUT, label_file_and_key, source_folder
from experiments.mixed95.train import _check_environment, _source_metrics

BASELINE_PATH = Path(__file__).with_name("benchmark_baseline.json")
DEFAULT_SUITE = "benchmark95_20261011_v1"
TERMINAL = {"complete", "failed", "blocked_smoke"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def log_tail(path, limit=20000):
    """Seek bytes, then decode: a progress bar may start mid UTF-8 character."""
    path = Path(path)
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - limit))
        return handle.read().decode("utf-8", errors="replace")


@contextmanager
def suite_lock(suite):
    suite.mkdir(parents=True, exist_ok=True)
    with (suite / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another command already owns this suite") from exc
        yield


def suite_root(output, suite_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", suite_id):
        raise ValueError("suite-id must contain only letters, digits, underscore or hyphen")
    return output / "benchmarks" / suite_id


def checked_configs(baseline):
    configs = {model: resolve_config(model) for model in MODELS}
    sources = {model: provenance(model) for model in MODELS}
    if configs != baseline["configs"] or sources != baseline["config_provenance"]:
        raise RuntimeError("public configurations changed since freezing; do not silently change this suite")
    return configs, sources


def code_inventory():
    files = list((ROOT / "experiments/mixed95").glob("benchmark*.py"))
    files.append(BASELINE_PATH)
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def initialize(suite, baseline):
    configs, sources = checked_configs(baseline)
    identity = {"configs": configs, "config_provenance": sources,
                "baseline_sha256": sha256(BASELINE_PATH), "code_sha256": code_inventory()}
    path = suite / "manifest.json"
    if path.exists():
        manifest = read(path)
        if manifest["identity"] != identity:
            raise RuntimeError("suite identity/config/code changed; existing results are protected")
        # An acquired suite lock proves the previous coordinator has exited.
        # Interrupted work is a failure, never an automatic rerun (especially
        # for target evaluation, which may already have emitted predictions).
        changed = False
        for entry in manifest["models"].values():
            for record in [entry["smoke"], *entry["seeds"].values(), entry["target"]]:
                if record["status"] == "running":
                    record.update(status="failed", reason="coordinator interrupted; no automatic retry")
                    changed = True
        if changed:
            save(path, manifest)
        return manifest
    manifest = {"schema": 1, "suite_id": suite.name, "created_at": now(), "identity": identity,
        "protocol": {"source_cells": {"train": 283, "val": 89, "test": 96},
            "effective_source_datasets": 8, "world_size": 2, "precision": "FP32",
            "selection": "source validation MAPE only", "source_test_warning": baseline["warning_cell"],
            "target_labels": {"C1": 379, "C2": 419},
            "target_warning": "ordinary-cycle proxy crossings, not full RPT observed lives",
            "target_use": "previously observed; fixed comparison, not new blind testing; no target tuning",
            "not_strict_v4_reproduction": True},
        "models": {m: {"smoke": {"status": "not_applicable" if m == "CPTransformer" else "pending"},
            "seeds": {str(s): {"status": "pending"} for s in SEEDS},
            "target": {"status": "pending"}} for m in MODELS}}
    save(path, manifest)
    return manifest


def verify_data(output, baseline, *, full_hash=True):
    """No re-labeling, no source copying, no writes to historical artifacts."""
    from data_provider.data_split_recorder import split_recorder
    preflight = read(output / "preflight_95.json")
    if not preflight.get("passed") or preflight["eligible_cells"] != {"train": 283, "val": 89, "test": 96}:
        raise RuntimeError("95% source coverage changed")
    if not read(output / "source_curves_audit_95.json").get("passed"):
        raise RuntimeError("source curve audit did not pass")
    actual = read(output / "source_labels_audit_95.json")
    trimmed = [{k: r[k] for k in ("cell", "split", "dataset", "status", "life")} for r in actual]
    if trimmed != baseline["cells"]:
        raise RuntimeError("source label decisions changed")
    splits = {s: list(getattr(split_recorder, f"MIX_large_{s}_files")) for s in ("train", "val", "test")}
    ids = [n for names in splits.values() for n in names]
    if len(ids) != 843 or len(set(ids)) != 843:
        raise RuntimeError("MIX_large split overlap/count changed")
    for split, names in splits.items():
        if names != [r["cell"] for r in baseline["cells"] if r["split"] == split]:
            raise RuntimeError("source split membership/order changed")
    eligible = [r for r in actual if r["life"] is not None]
    if len({r["dataset"] for r in eligible}) != 8 or any(r["life"] <= 100 for r in eligible):
        raise RuntimeError("effective datasets or early-life exclusion changed")
    input_root = Path(preflight["input_root"])
    labels = input_root / "Life labels"
    for name, digest in baseline["label_sha256"].items():
        if sha256(labels / name) != digest:
            raise RuntimeError(f"label hash changed: {name}")
    for record in actual:
        filename, key = label_file_and_key(record["cell"])
        if read(labels / filename).get(key) != record["life"]:
            raise RuntimeError(f"loader label mismatch: {record['cell']}")
    for name, digest in baseline["core_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"audited original code changed: {name}")
    target = output / "comac_loader_v2"
    for name, digest in baseline["target_sha256"].items():
        if sha256(target / name) != digest:
            raise RuntimeError(f"canonical COMAC input changed: {name}")
    if read(target / "COMAC_proxy_labels_95.json") != {"C1": 379, "C2": 419}:
        raise RuntimeError("COMAC proxy labels changed")
    source = Path(preflight["source"])
    for j, record in enumerate(baseline["source_inventory"], 1):
        path = source / record["dataset"] / record["cell"]
        staged = input_root / ("total_MICH" if record["cell"].startswith("MICH_")
                               else record["dataset"]) / record["cell"]
        if staged.resolve() != path.resolve() or path.stat().st_size != record["size_bytes"]:
            raise RuntimeError(f"source symlink or file size changed: {record['cell']}")
        if full_hash and sha256(path) != record["sha256"]:
            raise RuntimeError(f"source SHA256 changed: {record['cell']}")
        if j % 50 == 0 or j == 843:
            print(f"Source {'SHA256' if full_hash else 'size'} check: {j}/843", flush=True)
    return {"passed": True, "checked_at": now(), "full_source_sha256": full_hash,
        "source_files": 843, "source_bytes": sum(r["size_bytes"] for r in baseline["source_inventory"]),
        "input_root": str(input_root), "split_sha256": canonical_hash(splits),
        "label_sha256": baseline["label_sha256"], "target_sha256": baseline["target_sha256"],
        "core_sha256": baseline["core_sha256"]}


def import_cp(suite, output, baseline, manifest):
    """Hash-gated reuse: no training and no repeat COMAC prediction."""
    import joblib
    record = manifest["models"]["CPTransformer"]
    run = output / "runs" / baseline["cp_run_id"]
    if not run.exists():
        run = output / "audits/20261009_v1/remote_run"  # audited local copies
    old = read(run / "manifest.json")
    if old.get("smoke_only") or any(old["seeds"][str(s)]["status"] != "complete" for s in SEEDS):
        raise RuntimeError("existing CPTransformer three-seed run is incomplete")
    for name, digest in baseline["cp_run_sha256"].items():
        path = run / name
        # Audit download did not copy redundant CSV; JSON and weights are mandatory.
        if name.endswith(".csv") and not path.exists():
            continue
        if sha256(path) != digest:
            raise RuntimeError(f"existing CPTransformer artifact changed: {name}")
    official = read(run / "held_out_comac_evaluation_95_loader_v2.json")
    if official["target_input_sha256"] != {c: baseline["target_sha256"][f"{c}_input.npz"] for c in ("C1", "C2")}:
        raise RuntimeError("existing CPTransformer target inputs do not match")
    results = []
    config = manifest["identity"]["configs"]["CPTransformer"]
    for seed in SEEDS:
        saved = old["seeds"][str(seed)]
        cp = run / f"seed_{seed}" / Path(saved["checkpoint"]).name
        args = read(cp / "args.json")
        for key, value in dict(config, seed=seed).items():
            if args.get(key) != value:
                raise RuntimeError(f"existing CPTransformer config mismatch: {key}")
        validate_scaler(joblib.load(cp / "label_scaler"), baseline["scaler"])
        verified = baseline["cp_verified"][str(seed)]
        compare_metrics(verified["source"]["test"]["all_windows"], saved["source_test"])
        record["seeds"][str(seed)] = dict(saved, checkpoint=str(cp),
            status="complete", imported=True, provenance="20261009 independent audit",
            source_verified=dict(verified["source"]["test"],
                sensitivity_without_flagged_cell=baseline["cp_sensitivity"][str(seed)]),
            resource_record="historical timing/peak memory not measured; unavailable, not fabricated")
        cell_rows = [r for r in official["comac"] if int(r["seed"]) == seed]
        if {r["cell"] for r in cell_rows} != {"C1", "C2"} or len(cell_rows) != 2:
            raise RuntimeError("existing CPTransformer target results incomplete")
        for row in cell_rows:
            truth = {"C1": 379, "C2": 419}[row["cell"]]
            pred = row["predicted_life_cycle"]
            if row["proxy_label_life_cycle"] != truth or not math.isfinite(pred) or not (
                    abs(abs(pred-truth)-row["absolute_error_cycles"]) < 1e-8
                    and abs(100*abs(pred-truth)/truth-row["ape_percent"]) < 1e-8):
                raise RuntimeError("existing COMAC numeric result inconsistent")
            results.append(dict(row, model="CPTransformer", imported=True))
    record["target"] = {"status": "complete", "imported": True, "rows": results}
    save(suite / "manifest.json", manifest)


def gpu_memory():
    proc = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return [[int(v.strip()) for v in row.split(",")] for row in proc.stdout.strip().splitlines()]


def environment_preflight(suite):
    import importlib.metadata
    import torch
    _check_environment()
    import_probe = subprocess.run([sys.executable,
        str(ROOT / "experiments/mixed95/benchmark_worker.py"), "--import-check"],
        cwd=ROOT, capture_output=True, text=True, check=True, timeout=60)
    trainer_imports = json.loads(import_probe.stdout.strip().splitlines()[-1])
    # --help exits at the original parser, before accelerator/data creation,
    # but imports every original training-entry dependency using the actual
    # wrapper invocation. A common import failure must stop the entire suite,
    # not consume 12 full data-loading jobs and masquerade as model failures.
    entry_probe = subprocess.run([sys.executable,
        str(ROOT / "experiments/mixed95/benchmark_worker.py"), "--help"],
        cwd=ROOT, capture_output=True, text=True, check=True, timeout=60)
    if "usage: run_main.py" not in entry_probe.stdout or "--early_cycle_threshold" not in entry_probe.stdout:
        raise RuntimeError("original training parser help/import probe did not execute")
    trainer_imports["original_entry_all_imports_passed"] = True
    trainer_imports["original_parser_help_sha256"] = canonical_hash(entry_probe.stdout)
    if torch.__version__ != "2.4.1+cu118":
        raise RuntimeError("use the existing audited torch 2.4.1+cu118 environment; no automatic upgrade")
    gpu = gpu_memory()
    if len(gpu) < 2 or any(row[1] > 1024 or row[3] > 5 for row in gpu[:2]):
        raise RuntimeError("two idle GPUs are required; another job appears to use them")
    free = shutil.disk_usage(suite).free
    if free < 8 * 1024**3:
        raise RuntimeError("less than 8 GiB disk headroom; no automatic expansion/deletion")
    packages = {n: importlib.metadata.version(n) for n in (
        "torch", "numpy", "scikit-learn", "accelerate", "deepspeed", "safetensors", "joblib")}
    return {"checked_at": now(), "gpu": gpu, "disk_free_bytes": free, "packages": packages,
            "direct_worker_import_check": trainer_imports}


def trial_args(manifest, model, seed, input_root, checkpoint_root, smoke):
    return dict(manifest["identity"]["configs"][model], seed=seed, root_path=input_root,
        checkpoints=str(checkpoint_root), train_epochs=1 if smoke else 100,
        model_comment=f"{manifest['suite_id']}_{'smoke_not_for_test' if smoke else 'formal'}")


def launch_trial(suite, manifest, baseline, model, seed, input_root, smoke):
    entry = manifest["models"][model]
    old = entry["smoke"] if smoke else entry["seeds"][str(seed)]
    if old["status"] != "pending":
        print(f"Skipping {model}/{seed}/{'smoke' if smoke else 'formal'}: {old['status']}", flush=True)
        return
    environment_preflight(suite)
    root = suite / ("smoke" if smoke else "runs") / model / f"seed_{seed}"
    if root.exists():
        raise RuntimeError(f"unrecorded trial directory; refusing original trainer overwrite: {root}")
    root.mkdir(parents=True)
    args = trial_args(manifest, model, seed, input_root, root / "checkpoint", smoke)
    command = [sys.executable, "-m", "accelerate.commands.launch", "--multi_gpu",
               "--num_processes", "2", "--mixed_precision", "no", "--main_process_port", "25216",
               str(ROOT / "experiments/mixed95/benchmark_worker.py"), *cli_args(args)]
    env = dict(os.environ, WANDB_MODE="offline", CUDA_VISIBLE_DEVICES="0,1", PYTHONUNBUFFERED="1")
    record = {"status": "running", "started_at": now(), "args": args, "command": command,
              "log": str(root / "train.log"), "smoke_only": smoke, "peak_gpu_used_mib": [0, 0],
              "peak_memory_method": "nvidia-smi device memory, polled every second (not allocator exact peak)"}
    if smoke:
        entry["smoke"] = record
    else:
        entry["seeds"][str(seed)] = record
    save(root / "resolved_args.json", args)
    save(suite / "manifest.json", manifest)
    started = time.monotonic()
    process = None
    try:
        with (root / "train.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            record["pid"] = process.pid
            save(suite / "manifest.json", manifest)
            while process.poll() is None:
                time.sleep(1)
                for i, row in enumerate(gpu_memory()[:2]):
                    record["peak_gpu_used_mib"][i] = max(record["peak_gpu_used_mib"][i], row[1])
                tail = log_tail(root / "train.log")
                if re.search(r"(?:Train Loss|Vali MAPE|Test MAPE|Test MAE|Test RMSE)[^\n]*\b(?:nan|inf)\b", tail, re.I):
                    raise RuntimeError("nonfinite training/validation metrics")
        if process.returncode:
            raise RuntimeError(f"original trainer exit code {process.returncode}; inspect train.log")
        candidates = list((root / "checkpoint").glob("*/args.json"))
        if len(candidates) != 1:
            raise RuntimeError("expected exactly one original checkpoint")
        cp = candidates[0].parent
        record["checkpoint"] = str(cp)
        probes = read(cp / "checkpoint_reload_probes.json")
        if not probes or not all(p["state_exactly_equal"] and p["prediction_exactly_equal"] for p in probes):
            raise RuntimeError("save-time checkpoint consistency diagnostics missing/failed")
        record["save_time_probes"] = probes
        record["reload_probe"] = reload_probe(cp, args, baseline["scaler"])
        if not smoke:
            logged = _source_metrics((root / "train.log").read_text())
            if not all(math.isfinite(v) for v in logged.values()):
                raise RuntimeError("nonfinite reported source-test metrics")
            record["source_test"] = logged
            record["source_verified"] = source_evaluation(cp, args, baseline,
                root / "source_test_predictions.npz", logged)
        record["checkpoint_sha256"] = {p.name: sha256(p) for p in cp.iterdir() if p.is_file()
                                         and p.name in {"model.safetensors", "pytorch_model.bin", "args.json",
                                                        "label_scaler", "life_class_scaler", "checkpoint_reload_probes.json"}}
        record["status"] = "complete"
    except BaseException as exc:
        if process is not None:
            # Kill only the process group started by this trial, including
            # accelerate workers. Never kill other users' training processes.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        record["status"] = "failed"
        record["reason"] = f"{type(exc).__name__}: {exc}"
        record["exit_code"] = process.returncode if process else None
        print(f"FAILED {model}/{seed}: {record['reason']}", flush=True)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # release this coordinator's diagnostic cache
        record["finished_at"] = now()
        record["elapsed_seconds"] = time.monotonic() - started
        save(suite / "manifest.json", manifest)
    print(f"{record['status'].upper()} {model}/{seed} in {record['elapsed_seconds']:.1f}s", flush=True)


def smoke_models(suite, manifest, baseline, models, input_root):
    for model in models:
        if model != "CPTransformer":
            launch_trial(suite, manifest, baseline, model, 2021, input_root, True)


def train_models(suite, manifest, baseline, models, input_root):
    for model in models:
        if model == "CPTransformer":
            continue
        if manifest["models"][model]["smoke"]["status"] != "complete":
            if manifest["models"][model]["smoke"]["status"] == "pending":
                raise RuntimeError(f"{model} smoke not run; use smoke or train --with-smoke")
            for record in manifest["models"][model]["seeds"].values():
                if record["status"] == "pending":
                    record.update(status="blocked_smoke", reason="full-configuration smoke failed; no automatic retry")
            save(suite / "manifest.json", manifest)
            continue
        for seed in SEEDS:
            launch_trial(suite, manifest, baseline, model, seed, input_root, False)


def evaluate_models(suite, output, baseline, manifest):
    records = {m: r["seeds"] for m, r in manifest["models"].items()}
    if any(r["status"] not in TERMINAL for rows in records.values() for r in rows.values()):
        raise RuntimeError("all 39 source trials must be terminal before any new COMAC prediction")
    if read(output / "comac_loader_v2/COMAC_proxy_labels_95.json") != {"C1": 379, "C2": 419}:
        raise RuntimeError("COMAC labels changed")
    for model in MODELS:
        entry = manifest["models"][model]
        if entry["target"]["status"] != "pending":
            continue  # imported, completed, interrupted, or failed: never implicitly rerun
        if any(r["status"] != "complete" for r in entry["seeds"].values()):
            entry["target"] = {"status": "blocked_incomplete", "reason": "not all three source seeds completed"}
            save(suite / "manifest.json", manifest)
            continue
        for record in entry["seeds"].values():
            for name, digest in record["checkpoint_sha256"].items():
                if sha256(Path(record["checkpoint"]) / name) != digest:
                    raise RuntimeError("selected source checkpoint changed before target evaluation")
        entry["target"] = {"status": "running", "started_at": now()}
        save(suite / "manifest.json", manifest)  # protects against repeat after interruption
        try:
            rows = target_evaluation({model: entry["seeds"]}, manifest["identity"]["configs"], baseline,
                output / "comac_loader_v2",
                lambda m, c, value: save(suite / "target_mapping" / f"{m}_{c}.json", value))
            if len(rows) != 6:
                raise RuntimeError("expected exactly three seeds times two target cells")
            entry["target"] = {"status": "complete", "rows": rows, "finished_at": now()}
        except Exception as exc:
            entry["target"] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        save(suite / "manifest.json", manifest)


def aggregate(values):
    import numpy as np
    if len(values) != 3:
        raise ValueError("only three-seed complete models may be aggregated")
    keys = set.intersection(*(set(v) for v in values)) & set(METRICS)
    return {k: {"mean": float(np.mean([v[k] for v in values])),
                "std_population": float(np.std([v[k] for v in values], ddof=0))} for k in sorted(keys)}


def report_models(suite, manifest):
    rows, predictions, detailed = [], [], {}
    for model in MODELS:
        entry = manifest["models"][model]
        seeds = list(entry["seeds"].values())
        counts = dict(Counter(r["status"] for r in seeds))
        complete = counts == {"complete": 3}
        reasons = [r.get("reason", "") for r in seeds if r["status"] != "complete"]
        if entry["target"].get("reason"):
            reasons.append(entry["target"]["reason"])
        row = {"model": model, "source_status": "complete" if complete else json.dumps(counts),
               "target_status": entry["target"]["status"], "failure_reason": "; ".join(filter(None, reasons))}
        detail = {"seed_status": entry["seeds"], "target_status": entry["target"]["status"]}
        if complete:
            for protocol in ("all_windows", "cycle100_only", "sensitivity_without_flagged_cell"):
                summary = aggregate([r["source_verified"][protocol] for r in seeds])
                detail[protocol] = summary
                for key, stats in summary.items():
                    row[f"source_{protocol}_{key}_mean"] = stats["mean"]
                    row[f"source_{protocol}_{key}_std"] = stats["std_population"]
        if entry["target"]["status"] == "complete":
            target = entry["target"]["rows"]
            if len({(r["model"], r["seed"], r["cell"]) for r in target}) != 6:
                raise ValueError("duplicate/incomplete target records")
            predictions.extend(target)
            summary = aggregate([metrics([r["predicted_life_cycle"] for r in target if r["seed"] == s],
                                         [r["proxy_label_life_cycle"] for r in target if r["seed"] == s]) for s in SEEDS])
            detail["target_metrics"] = summary
            for key, stats in summary.items():
                row[f"comac_{key}_mean"] = stats["mean"]
                row[f"comac_{key}_std"] = stats["std_population"]
            for cell in ("C1", "C2"):
                values = [r["predicted_life_cycle"] for r in target if r["cell"] == cell]
                import numpy as np
                row[f"{cell}_predicted_mean"] = float(np.mean(values))
                row[f"{cell}_predicted_std"] = float(np.std(values))
                row[f"{cell}_absolute_error_mean"] = float(np.mean(
                    [r["absolute_error_cycles"] for r in target if r["cell"] == cell]))
                row[f"{cell}_ape_percent_mean"] = float(np.mean(
                    [r["ape_percent"] for r in target if r["cell"] == cell]))
        measured = [r for r in seeds if "elapsed_seconds" in r]
        row["elapsed_seconds_sum"] = sum(r["elapsed_seconds"] for r in measured) if measured else None
        row["peak_gpu_used_mib"] = max((max(r["peak_gpu_used_mib"]) for r in measured), default=None)
        rows.append(row)
        detailed[model] = detail
    save(suite / "report.json", {"models": detailed, "protocol": manifest["protocol"],
        "summary": rows, "predictions": predictions, "seed_std": "population, ddof=0",
        "target_independent_cell_count": 2})
    fields = list(dict.fromkeys(k for r in rows for k in r))
    for name, data, keys in (("comparison.csv", rows, fields),
                             ("comac_predictions.csv", predictions, list(dict.fromkeys(k for r in predictions for k in r)))):
        if not keys:
            continue
        with (suite / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
    lines = ["# BatteryLife 13-model / 95% COMAC benchmark", "",
        "代理标签：C1=379、C2=419，并非完整 RPT 实测寿命。源数据仅 8 个有效数据集。",
        "C1/C2 已在此前实验观察过；本轮固定横向比较，不是新盲测，也不按商飞成绩调参。",
        "均值与总体标准差按三个种子计算，六次预测不等于六枚独立电芯。",
        "未完成模型不填成绩。历史 CPTransformer 未测运行时间/显存，不补造数值。", "",
        "|模型|源训练状态|商飞状态|源窗口 MAPE %|第100圈 MAPE %|商飞 MAPE %|", "|---|---|---|---:|---:|---:|"]
    def pct(row, key):
        return f"{100*row[key]:.4f}" if key in row else "—"
    for r in rows:
        lines.append(f"|{r['model']}|{r['source_status']}|{r['target_status']}|"
                     f"{pct(r, 'source_all_windows_mape_mean')}|{pct(r, 'source_cycle100_only_mape_mean')}|{pct(r, 'comac_mape_mean')}|")
    (suite / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if predictions:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        valid = [r for r in rows if "comac_mape_mean" in r]
        fig, ax = plt.subplots(figsize=(max(7, len(valid)*.65), 4))
        ax.bar([r["model"] for r in valid], [r["comac_mape_mean"]*100 for r in valid],
               yerr=[r["comac_mape_std"]*100 for r in valid])
        ax.set(ylabel="COMAC MAPE (%)", title="Fixed 95% proxy-label comparison (3 seeds)")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(suite / "comac_mape.png", dpi=150)
        plt.close(fig)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("dry-run", "smoke", "train", "evaluate", "report"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--suite-id", default=DEFAULT_SUITE)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--check-data", action="store_true", help="dry-run: include full source SHA256 checks")
    parser.add_argument("--with-smoke", action="store_true", help="train: first run all selected full-config smoke jobs")
    parser.add_argument("--then-evaluate", action="store_true", help="train: after all source tasks are terminal, evaluate/report")
    args = parser.parse_args()
    output = args.output.resolve()
    suite = suite_root(output, args.suite_id)
    baseline = read(BASELINE_PATH)
    with suite_lock(suite):
        manifest = initialize(suite, baseline)
        try:
            if args.command == "report":
                report_models(suite, manifest)
                return
            if args.command == "dry-run" and not args.check_data:
                for model in args.models:
                    print(json.dumps({"model": model, "args": manifest["identity"]["configs"][model],
                        "parameter_source": manifest["identity"]["config_provenance"][model]}, ensure_ascii=False))
                return
            verified = verify_data(output, baseline)
            save(suite / "data_preflight.json", verified)
            import_cp(suite, output, baseline, manifest)
            if args.command in {"smoke", "train"}:
                save(suite / "environment_preflight.json", environment_preflight(suite))
                if args.command == "smoke" or args.with_smoke:
                    smoke_models(suite, manifest, baseline, args.models, verified["input_root"])
                if args.command == "train":
                    train_models(suite, manifest, baseline, args.models, verified["input_root"])
            if args.command == "evaluate" or (args.command == "train" and args.then_evaluate):
                evaluate_models(suite, output, baseline, manifest)
            # Recheck immutable originals after all writes confined to this suite.
            verify_data(output, baseline, full_hash=False)
            import_cp(suite, output, baseline, manifest)
            report_models(suite, manifest)
        except BaseException as exc:
            manifest["last_blocker"] = {"at": now(), "reason": f"{type(exc).__name__}: {exc}"}
            save(suite / "manifest.json", manifest)
            report_models(suite, manifest)
            raise


if __name__ == "__main__":
    main()
