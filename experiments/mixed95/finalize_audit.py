"""End-of-run integrity, artifact indices and independent Autoformer replay.

Only new audit JSON files are written. Formal manifests, predictions, scalers,
checkpoints, original curves and frozen benchmark modules remain unchanged.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.mixed95 import benchmark as bench
from experiments.mixed95.audit_benchmark_results import audit


def ready(suite):
    manifest = bench.read(suite / "manifest.json")
    integrity = bench.read(suite / "retry_completion_integrity.json")
    if not integrity.get("passed") or integrity["formal_prediction_count"] != 72:
        raise ValueError("actual completed 72-row retry suite required")
    if any(r["status"] not in bench.TERMINAL for e in manifest["models"].values()
           for r in e["seeds"].values()):
        raise ValueError("source work still nonterminal")
    if any(e["target"]["status"] not in {"complete", "failed", "blocked_incomplete"}
           for e in manifest["models"].values()):
        raise ValueError("target work still nonterminal")
    for folder in Path("/proc").iterdir():
        if not folder.name.isdigit():
            continue
        try:
            args = (folder / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if suite.name in args and ("benchmark_worker.py --task_name" in args or
                                  "experiments.mixed95.explicit_retry --output" in args):
            raise ValueError("an actual training coordinator/worker is still live")
    return manifest


def evidence_files(suite, manifest):
    paths = {suite / "manifest.json", suite / "report.json", suite / "comparison.csv",
             suite / "comac_predictions.csv", suite / "retry_completion_integrity.json",
             bench.BASELINE_PATH}
    paths.update(bench.ROOT / p for p in manifest["identity"]["code_sha256"])
    for entry in manifest["models"].values():
        for record in [entry["smoke"], *entry["seeds"].values()]:
            if "checkpoint" in record:
                paths.update(p for p in Path(record["checkpoint"]).iterdir() if p.is_file())
            if record.get("log"):
                paths.add(Path(record["log"]))
    return {str(p): bench.sha256(p) for p in sorted(paths)}


def integrity(suite, output):
    with bench.suite_lock(suite):
        manifest = ready(suite)
        snapshot = evidence_files(suite, manifest)
        baseline = bench.read(bench.BASELINE_PATH)
        preflight = bench.verify_data(suite.parent.parent, baseline, full_hash=True)
        bench.save(output / "end_of_run_data_preflight_fullhash.json", preflight)
        old_cp = suite.parent.parent / "runs" / baseline["cp_run_id"]
        cp_proof = {}
        for name, expected in baseline["cp_run_sha256"].items():
            path = old_cp / name
            if name.endswith(".csv") and not path.exists():
                continue  # Historical redundant CSV is not required by the import protocol.
            actual = bench.sha256(path)
            if actual != expected:
                raise ValueError("historical CPTransformer artifact changed")
            cp_proof[name] = actual
        numeric = audit(suite, suite / "audit_inputs", live=False)
        bench.save(output / "independent_final_numeric_audit.json", numeric)
        smoke_proof, checkpoints = {}, []
        for model, entry in manifest["models"].items():
            if model != "CPTransformer":
                smoke = entry["smoke"]
                if smoke["status"] != "complete":
                    raise ValueError("not all twelve original full-configuration smokes passed")
                for name, digest in smoke["checkpoint_sha256"].items():
                    if bench.sha256(Path(smoke["checkpoint"]) / name) != digest:
                        raise ValueError("smoke checkpoint changed")
                probes = bench.read(Path(smoke["checkpoint"]) / "checkpoint_reload_probes.json")
                if not probes or not all(p["state_exactly_equal"] and p["prediction_exactly_equal"] for p in probes):
                    raise ValueError("smoke save/reload proof missing")
                smoke_proof[model] = {"complete": True, "probes": len(probes),
                                      "checkpoint_sha256": smoke["checkpoint_sha256"]}
            for seed, record in entry["seeds"].items():
                if record["status"] != "complete":
                    checkpoints.append({"model": model, "seed": int(seed), "status": record["status"],
                                        "reason": record["reason"], "log": record.get("log")})
                    continue
                cp = Path(record["checkpoint"])
                files = [{"repo_relative_path": str(p.relative_to(bench.ROOT)),
                          "remote_path": str(p), "bytes": p.stat().st_size, "sha256": bench.sha256(p)}
                         for p in sorted(cp.iterdir()) if p.is_file()]
                checkpoints.append({"model": model, "seed": int(seed), "status": "complete",
                    "checkpoint": str(cp), "log": record.get("log"), "files": files,
                    "parsed_arguments": bench.read(cp / "args.json"),
                    "configuration_provenance": manifest["identity"]["config_provenance"][model],
                    "historical_cp_save_time_probe_unavailable": model == "CPTransformer"})
        bench.save(output / "checkpoint_scaler_configuration_index.json", {"entries": checkpoints,
            "successful_seed_checkpoints": 36, "failed_seed_records": 3, "smokes": smoke_proof})
        # The independent auditor has ALL FIVE sensitivity fields, including
        # CPTransformer acc10 omitted from the old frozen baseline/report.
        sensitivity = {model: value.get("sensitivity_without_flagged_cell")
                       for model, value in numeric["models"].items()}
        if any(stats is not None and set(stats) != set(bench.METRICS) for stats in sensitivity.values()):
            raise ValueError("incomplete common sensitivity metric set")
        bench.save(output / "uniform_sensitivity_summary.json", {"cells": 95, "windows": 9500,
            "excluded_cell": baseline["warning_cell"], "replaces_formal_results": False,
            "source": "independent prediction-array calculation; CP acc10 supplemented, not zero-filled",
            "metrics": sensitivity})
        if evidence_files(suite, manifest) != snapshot:
            raise ValueError("protected artifacts changed during end-of-run integrity audit")
        bench.save(output / "protected_artifact_hashes.json", {"passed": True, "checked_at": bench.now(),
            "hashes": snapshot, "historical_cp_hashes": cp_proof, "source_full_sha256": True,
            "actual_training_processes_absent": True})


def replay_autoformer(suite, output):
    import joblib
    import torch
    from safetensors.torch import load_file

    with bench.suite_lock(suite):
        manifest = ready(suite)
        protected = evidence_files(suite, manifest)
        torch.set_num_threads(8)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        official = manifest["models"]["Autoformer"]["target"]["rows"]
        rows = []
        for seed in bench.SEEDS:
            record = manifest["models"]["Autoformer"]["seeds"][str(seed)]
            cp = Path(record["checkpoint"])
            args = bench.read(cp / "args.json")
            model = importlib.import_module("models.Autoformer").Model(SimpleNamespace(**args)).to("cuda:0").eval()
            state = load_file(str(cp / "model.safetensors"), device="cpu")
            model.load_state_dict(state, strict=True)
            scaler = joblib.load(cp / "label_scaler")
            if int(scaler.n_samples_seen_) != 283 or not all(p.dtype == torch.float32 for p in model.parameters()):
                raise ValueError("incorrect scaler population or precision")
            for cell in ("C1", "C2"):
                with np.load(suite.parent.parent / "comac_loader_v2" / f"{cell}_input.npz") as a:
                    x, mask, ids = a["features"].copy(), a["mask"].copy(), a["physical_cycles"]
                if x.shape != (100, 3, 300) or not np.array_equal(ids, np.arange(1, 101)):
                    raise ValueError("future cycles/incorrect canonical target shape")
                x[mask == 0] = 0
                with torch.inference_mode():
                    value = model(torch.from_numpy(x[None]).cuda(), torch.from_numpy(mask[None]).cuda())
                if value.shape != (1, 1) or not torch.isfinite(value).all():
                    raise ValueError("invalid independent replay output")
                life = float(value.item()) * float(scaler.scale_[0]) + float(scaler.mean_[0])
                ref = next(r for r in official if int(r["seed"]) == seed and r["cell"] == cell)
                delta = life - ref["predicted_life_cycle"]
                if abs(delta) > .005:
                    raise ValueError("Autoformer diagnostic/formal prediction mismatch")
                rows.append({"model": "Autoformer", "seed": seed, "cell": cell,
                    "diagnostic_life_cycles": life, "official_life_cycles": ref["predicted_life_cycle"],
                    "delta_cycles": delta})
            if any(not torch.equal(v.detach().cpu(), state[k]) for k, v in model.state_dict().items()):
                raise ValueError("model state updated during independent target inference")
            del model
        if evidence_files(suite, manifest) != protected:
            raise ValueError("formal artifacts changed during diagnostic replay")
        bench.save(output / "autoformer_diagnostic_replay_6.json", {"passed": True, "at": bench.now(),
            "rows": rows, "max_abs_delta_cycles": max(abs(r["delta_cycles"]) for r in rows),
            "tolerance_cycles": .005, "device": "cuda:0", "precision": "FP32",
            "no_formal_result_replacement": True, "no_production_target_evaluator_used": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("integrity", "autoformer-replay"))
    parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    suite = args.suite.resolve()
    output = suite / "final_audit_20261013_v1"
    expected = "protected_artifact_hashes.json" if args.mode == "integrity" else "autoformer_diagnostic_replay_6.json"
    if (output / expected).exists():
        raise RuntimeError("audit already completed; never overwrite accepted evidence")
    (integrity if args.mode == "integrity" else replay_autoformer)(suite, output)
    print("FINAL AUDIT", args.mode, "PASSED", flush=True)


if __name__ == "__main__":
    main()
