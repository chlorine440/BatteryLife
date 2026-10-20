"""Explicit, no-retraining recovery of the v3 post-training backend audit.

The old suite remains read-only. Checkpoint references, not weights/data, are
imported into a new suite. Every imported source checkpoint is re-audited with
the unchanged strict metric tolerances before remaining training is resumed.
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

from experiments.mixed95 import benchmark as bench
from experiments.mixed95.benchmark_config import MODELS, SEEDS, sha256
from experiments.mixed95.benchmark_inference import reload_probe, source_evaluation
from experiments.mixed95.train import _source_metrics

V3_INFERENCE_SHA = "5b6d08f650a4195fe8999508d115497a854e620f28805dbe414d4d35740301d4"
CHECKPOINT_FILES = {"model.safetensors", "pytorch_model.bin", "args.json", "label_scaler",
                    "life_class_scaler", "checkpoint_reload_probes.json"}


def eligible_for_reaudit(record):
    if record["status"] == "complete":
        return True
    return (record["status"] == "failed" and record.get("exit_code") == 0
            and record.get("reason", "").startswith(
                "ValueError: independent/logged metric mismatch:"))


def checkpoint_hashes(record):
    path = Path(record["checkpoint"])
    hashes = {p.name: sha256(p) for p in path.iterdir() if p.is_file() and p.name in CHECKPOINT_FILES}
    if not {"args.json", "label_scaler", "life_class_scaler", "checkpoint_reload_probes.json"} <= hashes.keys():
        raise RuntimeError("incomplete saved checkpoint; never import unfinished training")
    if not ({"model.safetensors", "pytorch_model.bin"} & hashes.keys()):
        raise RuntimeError("checkpoint weights missing")
    for name, digest in record.get("checkpoint_sha256", {}).items():
        if hashes.get(name) != digest:
            raise RuntimeError(f"historical checkpoint changed: {name}")
    probes = bench.read(path / "checkpoint_reload_probes.json")
    if not probes or not all(p["state_exactly_equal"] and p["prediction_exactly_equal"]
                              and p["cpu_rng_restored"] for p in probes):
        raise RuntimeError("historical save-time consistency proof failed")
    return hashes, probes


def validate_imported_args(previous, source, model, seed, record, input_root, smoke):
    root = source / ("smoke" if smoke else "runs") / model / f"seed_{seed}"
    expected = bench.trial_args(previous, model, seed, input_root, root / "checkpoint", smoke)
    if record["args"] != expected or Path(record["log"]) != root / "train.log":
        raise RuntimeError(f"historical trial is not the frozen training configuration: {model}/{seed}")
    if Path(record["checkpoint"]).parent != root / "checkpoint":
        raise RuntimeError("historical checkpoint outside its recorded trial")


def recover(output, source_id, suite_id):
    if source_id != "benchmark95_20261011_v3" or source_id == suite_id:
        raise ValueError("this explicit repair accepts only the preserved v3 suite and a new suite-id")
    source = bench.suite_root(output, source_id)
    suite = bench.suite_root(output, suite_id)
    baseline = bench.read(bench.BASELINE_PATH)
    # A lock, not a stale manifest, proves that the old coordinator has exited.
    with bench.suite_lock(source), bench.suite_lock(suite):
        previous = bench.read(source / "manifest.json")
        if any(r["status"] == "running" for e in previous["models"].values()
               for r in [e["smoke"], *e["seeds"].values(), e["target"]]):
            raise RuntimeError("previous trial still running; complete it before recovery")
        if any(e["target"]["status"] != "pending" for m, e in previous["models"].items()
               if m != "CPTransformer"):
            raise RuntimeError("v3 has already used new target checkpoints; refuse repeated target evaluation")
        manifest = bench.initialize(suite, baseline)
        if previous["identity"]["configs"] != manifest["identity"]["configs"] or previous[
                "identity"]["config_provenance"] != manifest["identity"]["config_provenance"]:
            raise RuntimeError("training configurations changed")
        if previous["identity"]["baseline_sha256"] != sha256(bench.BASELINE_PATH):
            raise RuntimeError("audited data/labels baseline changed")
        for path, digest in previous["identity"]["code_sha256"].items():
            if path.endswith("benchmark_inference.py"):
                if digest != V3_INFERENCE_SHA:
                    raise RuntimeError("not the identified v3 inference implementation")
            elif sha256(bench.ROOT / path) != digest:
                raise RuntimeError(f"training/config code changed: {path}")
        verified = bench.verify_data(output, baseline)
        bench.save(suite / "data_preflight.json", verified)
        bench.save(suite / "environment_preflight.json", bench.environment_preflight(suite))
        bench.import_cp(suite, output, baseline, manifest)
        manifest["recovery"] = {"source_suite": source_id, "started_at": bench.now(),
            "source_manifest_sha256": sha256(source / "manifest.json"),
            "source_suite_identity": previous["identity"],
            "change": "post-training inference now matches run_main.set_seed cuDNN settings",
            "training_unchanged": True, "metric_tolerances_unchanged": True,
            "weights_referenced_not_copied": True, "no_training_retry": True}
        bench.save(suite / "recovery" / "source_manifest_v3.json", previous)
        for model in MODELS:
            if model == "CPTransformer":
                continue
            old = previous["models"][model]["smoke"]
            if old["status"] != "complete":
                raise RuntimeError(f"{model}: a verified complete v3 smoke is required")
            validate_imported_args(previous, source, model, 2021, old, verified["input_root"], True)
            hashes, probes = checkpoint_hashes(old)
            record = copy.deepcopy(old)
            record["reload_probe"] = reload_probe(Path(old["checkpoint"]), old["args"], baseline["scaler"])
            record.update(checkpoint_sha256=hashes, save_time_probes=probes, imported_from=source_id)
            manifest["models"][model]["smoke"] = record
        bench.save(suite / "manifest.json", manifest)
        # Test the two identified failures first: no silently relaxed acceptance.
        pairs = [("CPBiGRU", 42), ("CPBiGRU", 2024)]
        pairs += [(m, s) for m in MODELS if m != "CPTransformer" for s in SEEDS
                  if (m, s) not in pairs]
        for model, seed in pairs:
            old = previous["models"][model]["seeds"][str(seed)]
            if old["status"] == "pending":
                continue
            destination = manifest["models"][model]["seeds"]
            if destination[str(seed)]["status"] != "pending":
                raise RuntimeError("recovery is explicit and one-shot; imported trials are not automatically re-audited")
            if not eligible_for_reaudit(old):
                destination[str(seed)] = copy.deepcopy(old)
                bench.save(suite / "manifest.json", manifest)
                continue  # Genuine training failures are preserved, never retried.
            validate_imported_args(previous, source, model, seed, old, verified["input_root"], False)
            hashes, probes = checkpoint_hashes(old)
            root = suite / "runs" / model / f"seed_{seed}"
            root.mkdir(parents=True, exist_ok=True)
            record = copy.deepcopy(old)
            record.update(status="running", imported_from=source_id,
                prior_audit_status=old["status"], prior_audit_reason=old.get("reason"),
                checkpoint_sha256=hashes, save_time_probes=probes)
            destination[str(seed)] = record
            bench.save(suite / "manifest.json", manifest)
            try:
                logged = _source_metrics(Path(old["log"]).read_text())
                if not all(math.isfinite(v) for v in logged.values()):
                    raise RuntimeError("nonfinite original source metrics")
                record["reload_probe"] = reload_probe(Path(old["checkpoint"]), old["args"], baseline["scaler"])
                record["source_verified"] = source_evaluation(Path(old["checkpoint"]), old["args"],
                    baseline, root / "source_test_predictions.npz", logged)
                after, _ = checkpoint_hashes(old)
                if after != hashes:
                    raise RuntimeError("checkpoint changed during read-only recovery")
                record.update(status="complete", source_test=logged, reaudited_at=bench.now())
                record.pop("reason", None)
                print(f"REAUDIT_COMPLETE {model}/{seed}: {record['source_verified']['logged_metric_deltas']}", flush=True)
            except BaseException as exc:
                record.update(status="failed", reason=f"recovery audit: {type(exc).__name__}: {exc}")
                raise
            finally:
                bench.save(suite / "manifest.json", manifest)
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
        manifest["recovery"]["finished_at"] = bench.now()
        bench.save(suite / "manifest.json", manifest)
        bench.report_models(suite, manifest)
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=bench.DEFAULT_OUTPUT)
    parser.add_argument("--source-suite", required=True)
    parser.add_argument("--suite-id", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    recover(output, args.source_suite, args.suite_id)
    # Continue only pending training with the original, unchanged trainer.
    sys.argv = [sys.argv[0], "train", "--output", str(output), "--suite-id", args.suite_id,
                "--with-smoke", "--then-evaluate"]
    bench.main()


if __name__ == "__main__":
    main()
