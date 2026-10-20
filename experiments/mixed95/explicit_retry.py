"""Explicitly authorized single Autoformer/2024 restart after long live pause.

Preserve the terminal v4 suite, failed record and existing 66 COMAC predictions.
Reference existing accepted checkpoints; only this one new attempt is pending.
The unmodified frozen benchmark then runs the original trainer and evaluator.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from experiments.mixed95 import benchmark as bench
from experiments.mixed95.resume_paused import STAGE_REPORT_SHA
from experiments.mixed95.staged_evaluate import STAGED_MODELS


def validate_previous(previous, control):
    if control.get("status") != "finished" or control.get("trainer_exit_code") != 1:
        raise ValueError("finished failed live-resume evidence required")
    failed = previous["models"]["Autoformer"]["seeds"]["2024"]
    if failed["status"] != "failed" or failed.get("exit_code") != 1 or not failed.get("live_pause_seconds", 0) > 600:
        raise ValueError("only the documented long-pause Autoformer/2024 failure may restart")
    for model, entry in previous["models"].items():
        if any(r["status"] not in bench.TERMINAL for r in entry["seeds"].values()):
            raise ValueError("previous source trials must all be terminal")
        if model in STAGED_MODELS and (entry["target"]["status"] != "complete" or
                len(entry["target"]["rows"]) != 6):
            raise ValueError("all eleven accepted target groups must be retained")
    if any(previous["models"]["Autoformer"]["seeds"][str(s)]["status"] != "complete"
           for s in (2021, 42)):
        raise ValueError("previous two Autoformer seeds must be accepted and reused")
    if previous["models"]["Autoformer"]["target"]["status"] != "blocked_incomplete":
        raise ValueError("Autoformer must not have a previous target attempt")
    return failed


def imported_manifest(previous, new_identity, suite_id, authorized_at):
    if previous["identity"] != new_identity:
        raise ValueError("frozen code/config identity changed")
    result = copy.deepcopy(previous)
    result.update(suite_id=suite_id, created_at=authorized_at)
    result["models"]["Autoformer"]["seeds"]["2024"] = {"status": "pending"}
    result["models"]["Autoformer"]["target"] = {"status": "pending"}
    result.pop("last_blocker", None)
    result["explicit_retry"] = {"authorized_at": authorized_at,
        "authorization": "user: 重训这个种子，完成 Autoformer",
        "previous_suite": previous["suite_id"], "model": "Autoformer", "seed": 2024,
        "failed_record": copy.deepcopy(previous["models"]["Autoformer"]["seeds"]["2024"]),
        "change": "single user-authorized restart from epoch 1; no optimizer checkpoint exists",
        "frozen_training_parameters_changed": False,
        "already_completed_training_and_target_evaluations_reused": True,
        "other_failed_seeds_not_retried": True}
    return result


def prepare(output, previous_id, suite_id):
    previous_path = bench.suite_root(output, previous_id)
    suite = bench.suite_root(output, suite_id)
    if previous_path == suite or suite.exists():
        raise RuntimeError("explicit restart requires a NEW, nonexistent suite; old results never overwritten")
    baseline = bench.read(bench.BASELINE_PATH)
    with bench.suite_lock(previous_path):
        before = bench.sha256(previous_path / "manifest.json")
        previous = bench.read(previous_path / "manifest.json")
        control = bench.read(previous_path / "resume_after_staged11_20261013.json")
        validate_previous(previous, control)
        if bench.sha256(previous_path / "staged11_20261013/report.json") != STAGE_REPORT_SHA:
            raise RuntimeError("staged results changed")
        log = Path(previous["models"]["Autoformer"]["seeds"]["2024"]["log"]).read_text(errors="replace")
        if "Watchdog caught collective operation timeout" not in log:
            raise RuntimeError("documented NCCL timeout evidence absent")
        bench.environment_preflight(previous_path)
        with bench.suite_lock(suite):
            initialized = bench.initialize(suite, baseline)
            imported = imported_manifest(previous, initialized["identity"], suite_id, bench.now())
            imported["explicit_retry"]["previous_manifest_sha256"] = before
            bench.save(suite / "explicit_retry_previous_manifest.json", previous)
            bench.save(suite / "explicit_retry_pause_resume_evidence.json", control)
            (suite / "audit_inputs").symlink_to(previous_path / "audit_inputs", target_is_directory=True)
            for mapping in (previous_path / "target_mapping").glob("*.json"):
                bench.save(suite / "target_mapping" / mapping.name, bench.read(mapping))
            for entry in previous["models"].values():
                for record in entry["seeds"].values():
                    for name, digest in record.get("checkpoint_sha256", {}).items():
                        if bench.sha256(Path(record["checkpoint"]) / name) != digest:
                            raise RuntimeError("previous checkpoint changed")
            bench.save(suite / "manifest.json", imported)
            bench.report_models(suite, imported)
            if bench.sha256(previous_path / "manifest.json") != before:
                raise RuntimeError("old suite unexpectedly changed")
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-suite", required=True)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--approve-autoformer-2024-retry", action="store_true", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    suite = prepare(output, args.previous_suite, args.suite_id)
    # Keep the original entry, all training flags/config and failure isolation.
    sys.argv = [sys.argv[0], "train", "--output", str(output), "--suite-id", suite.name,
                "--models", "Autoformer", "--then-evaluate"]
    bench.main()
    final = bench.read(suite / "manifest.json")
    previous = bench.read(suite / "explicit_retry_previous_manifest.json")
    for model in STAGED_MODELS:
        if final["models"][model]["target"]["rows"] != previous["models"][model]["target"]["rows"]:
            raise RuntimeError("previous eleven-model target predictions changed")
    previous_path = bench.suite_root(output, args.previous_suite)
    if bench.sha256(previous_path / "manifest.json") != final["explicit_retry"]["previous_manifest_sha256"]:
        raise RuntimeError("previous terminal suite changed")
    if bench.sha256(previous_path / "staged11_20261013/report.json") != STAGE_REPORT_SHA:
        raise RuntimeError("previous staged target report changed")
    from experiments.mixed95.audit_benchmark_results import audit
    bench.save(suite / "explicit_retry_final_numeric_audit.json",
               audit(suite, suite / "audit_inputs", live=False))
    bench.save(suite / "retry_completion_integrity.json", {"passed": True, "finished_at": bench.now(),
        "prior_eleven_model_predictions_unchanged": True, "prior_suite_manifest_unchanged": True,
        "retry_source_status": final["models"]["Autoformer"]["seeds"]["2024"]["status"],
        "autoformer_target_status": final["models"]["Autoformer"]["target"]["status"],
        "formal_prediction_count": len(bench.read(suite / "report.json")["predictions"])})


if __name__ == "__main__":
    main()
