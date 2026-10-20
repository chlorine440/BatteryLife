"""Resume the existing live Autoformer trial, without repeating staged targets.

The original coordinator remains SIGSTOP'd while its child resumes. This keeps
the launcher's real exit status available as a zombie owned by that coordinator.
Only after that exact child exits do we retire the scheduler, acquire its suite
lock, reuse staged targets and apply the original checkpoint/source audit.
No trainer, optimizer, model, data or frozen benchmark module is changed.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import math
import os
import signal
import time
from datetime import datetime
from pathlib import Path

from experiments.mixed95 import benchmark as bench
from experiments.mixed95.priority_handoff import acquire_released_suite_lock
from experiments.mixed95.staged_evaluate import STAGED_MODELS, validate_pause

STAGE_REPORT_SHA = "829afcc2dde74733d85a8bf243f6cb411810788dd8b9b9de6963dd9f8d23b192"


def process(pid, base=Path("/proc")):
    try:
        folder = base / str(pid)
        fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "state": fields[0], "ppid": int(fields[1]),
                "starttime": fields[19], "exit_status": int(fields[49]),
                "command": (folder / "cmdline").read_bytes().replace(b"\0", b" ").decode()}
    except FileNotFoundError:
        return None


def same_process(old):
    current = process(old["pid"])
    if current is None or current["starttime"] != old["starttime"]:
        raise RuntimeError(f"recorded process disappeared or PID was reused: {old['pid']}")
    return current


def zombie_returncode(current):
    if current["state"] != "Z":
        raise ValueError("a real zombie exit status is required; never guess trainer success")
    return os.waitstatus_to_exitcode(current["exit_status"])


def resume_order(members):
    parents = {p["pid"]: p["ppid"] for p in members}
    def depth(pid):
        seen = set()
        while pid in parents:
            if pid in seen:
                raise ValueError("cyclic recorded process tree")
            seen.add(pid)
            pid = parents[pid]
        return len(seen)
    # Wake loaders before ranks and the launcher, avoiding a resumed parent
    # tearing down still-stopped children on a rapid runtime failure.
    return sorted(members, key=lambda p: depth(p["pid"]), reverse=True)


def validate_stage(original, current, stage, report):
    if current != original:
        raise ValueError("original suite changed since the staged snapshot")
    if stage.get("stage_status") != "complete" or len(report["predictions"]) != 66:
        raise ValueError("completed 66-row staged report required")
    if set(stage["target_status"]) != set(STAGED_MODELS):
        raise ValueError("staged model scope changed")
    rows = []
    for model in STAGED_MODELS:
        source = current["models"][model]
        target = stage["target_status"][model]
        if any(r["status"] != "complete" for r in source["seeds"].values()):
            raise ValueError("staged source group no longer complete")
        if target["status"] != "complete" or len(target["rows"]) != 6:
            raise ValueError("staged target group incomplete")
        if {(r["model"], int(r["seed"]), r["cell"]) for r in target["rows"]} != {
                (model, seed, cell) for seed in bench.SEEDS for cell in ("C1", "C2")}:
            raise ValueError("staged target identity/uniqueness changed")
        if model == "CPTransformer":
            if source["target"]["rows"] != target["rows"]:
                raise ValueError("historical CPTransformer results changed")
        elif source["target"]["status"] != "pending":
            raise ValueError("do not overwrite an existing target attempt")
        rows.extend(target["rows"])
    if rows != report["predictions"]:
        raise ValueError("staged manifest/report predictions differ")
    running = [(m, s) for m, e in current["models"].items()
               for s, r in e["seeds"].items() if r["status"] == "running"]
    if running != [("Autoformer", "2024")]:
        raise ValueError("only the original Autoformer/2024 trial may resume")


def import_stage(manifest, stage, stage_path):
    result = copy.deepcopy(manifest)
    for model in STAGED_MODELS:
        if model != "CPTransformer":
            result["models"][model]["target"] = dict(copy.deepcopy(stage["target_status"][model]),
                staged_result_imported=True, staged_directory=str(stage_path),
                staged_report_sha256=STAGE_REPORT_SHA, no_repeat_target_inference=True)
    result["protocol"]["staged_target_amendment"] = stage["protocol_amendment"]
    result["protocol"]["staged_targets_reused_without_inference"] = list(STAGED_MODELS)
    return result


def verify_unchanged_checkpoints(manifest):
    for entry in manifest["models"].values():
        for record in entry["seeds"].values():
            for name, digest in record.get("checkpoint_sha256", {}).items():
                if bench.sha256(Path(record["checkpoint"]) / name) != digest:
                    raise RuntimeError("previously accepted/failed checkpoint changed")


def resume(suite, stage_path):
    control_path = suite / "resume_after_staged11_20261013.json"
    with (suite / "resume_after_staged11.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if control_path.exists():
            raise RuntimeError("resume already attempted; no implicit replay/retry")
        initial = bench.read(suite / "manifest.json")
        stage = bench.read(stage_path / "manifest.json")
        report = bench.read(stage_path / "report.json")
        original = bench.read(stage_path / "original_suite_snapshot.json")
        if bench.sha256(stage_path / "report.json") != STAGE_REPORT_SHA:
            raise RuntimeError("audited staged target report hash changed")
        validate_stage(original, initial, stage, report)
        if bench.sha256(suite / "manifest.json") != stage["original_manifest_sha256"]:
            raise RuntimeError("original suite snapshot hash changed")
        for name, digest in initial["identity"]["code_sha256"].items():
            if bench.sha256(bench.ROOT / name) != digest:
                raise RuntimeError("frozen benchmark code changed")
        baseline = bench.read(bench.BASELINE_PATH)
        bench.verify_data(suite.parent.parent, baseline, full_hash=False)
        verify_unchanged_checkpoints(initial)
        pause = bench.read(suite / "autoformer_live_pause_for_staged11_20261013.json")
        validate_pause(pause)
        coordinator = pause["coordinator"]
        members = pause["members_before_pause"]
        launcher = next(p for p in members if p["pid"] == pause["trial_pid"])
        if same_process(launcher)["ppid"] != coordinator["pid"]:
            raise RuntimeError("original launcher no longer belongs to paused coordinator")
        trial = initial["models"]["Autoformer"]["seeds"]["2024"]
        if trial["pid"] != launcher["pid"] or str(suite) not in trial["log"]:
            raise RuntimeError("trial identity mismatch")
        if not all(v in same_process(coordinator)["command"] for v in (
                "experiments.mixed95.benchmark", suite.name, "--then-evaluate")):
            raise RuntimeError("unexpected old coordinator command")
        control = {"status": "validated", "authorized_by": "user: ok，继续训模型吧",
            "started_at": bench.now(), "supervisor_pid": os.getpid(),
            "module_sha256": bench.sha256(Path(__file__)), "original_coordinator": coordinator,
            "trial_pid": launcher["pid"], "original_manifest_sha256": stage["original_manifest_sha256"],
            "staged_report_sha256": STAGE_REPORT_SHA, "training_parameters_changed": False,
            "resume_existing_memory_state": True, "no_new_training_attempt": True,
            "method": "resume exact worker descendants; keep scheduler stopped to retain actual child exit status; legal suite-lock handoff only after trainer exits"}
        bench.save(control_path, control)
        resumed_at = bench.now()
        disappeared = []
        for member in resume_order(members):
            actual = process(member["pid"])
            if actual is None:
                disappeared.append(member["pid"])
                continue  # Actual trainer exit status, never a guess, decides success.
            if actual["starttime"] != member["starttime"]:
                raise RuntimeError("PID reused during worker resume; do not signal it")
            if actual["state"] == "T":
                os.kill(member["pid"], signal.SIGCONT)
        control.update(status="training_resumed", resumed_at=resumed_at,
            resumed_processes=[p["pid"] for p in members if p["pid"] not in disappeared],
            vanished_during_resume=disappeared, peak_gpu_used_mib_after_resume=[0, 0])
        bench.save(control_path, control)
        print("RESUMED existing Autoformer/2024 trial", launcher["pid"], flush=True)
        last_write = 0
        while True:
            if same_process(coordinator)["state"] != "T":
                raise RuntimeError("old scheduler unexpectedly resumed; protect staged target results")
            current = same_process(launcher)
            if current["state"] == "Z":
                returncode = zombie_returncode(current)
                break
            for i, row in enumerate(bench.gpu_memory()[:2]):
                control["peak_gpu_used_mib_after_resume"][i] = max(
                    control["peak_gpu_used_mib_after_resume"][i], row[1])
            if time.monotonic() - last_write >= 30:
                control["last_observed_at"] = bench.now()
                bench.save(control_path, control)
                last_write = time.monotonic()
            time.sleep(1)
        control.update(status="trainer_exited", trainer_exit_code=returncode,
                       trainer_finished_at=bench.now(), kernel_wait_status=current["exit_status"])
        bench.save(control_path, control)
        # Never stop a live trainer at this boundary. Allow child teardown to finish.
        deadline = time.monotonic() + 30
        while any((p := process(old["pid"])) is not None and
                  p["starttime"] == old["starttime"] and p["state"] != "Z"
                  for old in members):
            if time.monotonic() > deadline:
                raise RuntimeError("training descendants remain live after launcher exit; no handoff")
            time.sleep(.1)
        if bench.read(suite / "manifest.json") != initial:
            raise RuntimeError("old manifest changed while coordinator was stopped")
        # Exact stopped scheduler only; the trial has already exited naturally.
        same_process(coordinator)
        os.kill(coordinator["pid"], signal.SIGKILL)
        with (suite / ".lock").open("a") as suite_guard:
            acquire_released_suite_lock(suite_guard)
            manifest = import_stage(initial, stage, stage_path)
            record = manifest["models"]["Autoformer"]["seeds"]["2024"]
            record["exit_code"] = returncode
            record["scheduler_handoff"] = str(control_path)
            pause_seconds = (datetime.fromisoformat(resumed_at) -
                             datetime.fromisoformat(pause["paused_at"])).total_seconds()
            record["live_pause_seconds"] = pause_seconds
            record["peak_gpu_used_mib"] = control["peak_gpu_used_mib_after_resume"]
            record["peak_memory_method"] = "nvidia-smi every second AFTER live resume; pre-pause peak unavailable"
            try:
                if returncode != 0:
                    raise RuntimeError(f"original trainer exit code {returncode} after live resume; no automatic retry")
                root = Path(record["log"]).parent
                candidates = list((root / "checkpoint").glob("*/args.json"))
                if len(candidates) != 1:
                    raise RuntimeError("expected exactly one original checkpoint")
                cp = candidates[0].parent
                record["checkpoint"] = str(cp)
                probes = bench.read(cp / "checkpoint_reload_probes.json")
                if not probes or not all(p["state_exactly_equal"] and p["prediction_exactly_equal"] for p in probes):
                    raise RuntimeError("save-time checkpoint consistency diagnostics missing/failed")
                record["save_time_probes"] = probes
                record["reload_probe"] = bench.reload_probe(cp, record["args"], baseline["scaler"])
                logged = bench._source_metrics(Path(record["log"]).read_text())
                if not all(math.isfinite(v) for v in logged.values()):
                    raise RuntimeError("nonfinite reported source-test metrics")
                record["source_test"] = logged
                record["source_verified"] = bench.source_evaluation(cp, record["args"], baseline,
                    root / "source_test_predictions.npz", logged)
                record["checkpoint_sha256"] = {p.name: bench.sha256(p) for p in cp.iterdir()
                    if p.is_file() and p.name in {"model.safetensors", "pytorch_model.bin", "args.json",
                        "label_scaler", "life_class_scaler", "checkpoint_reload_probes.json"}}
                record["status"] = "complete"
            except Exception as exc:
                record.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            record["finished_at"] = bench.now()
            record["elapsed_seconds"] = (datetime.fromisoformat(record["finished_at"]) -
                                         datetime.fromisoformat(record["started_at"])).total_seconds()
            record["elapsed_includes_live_pause"] = True
            bench.save(suite / "manifest.json", manifest)
            for mapping in (stage_path / "target_mapping").glob("*.json"):
                destination = suite / "target_mapping" / mapping.name
                if destination.exists() and bench.read(destination) != bench.read(mapping):
                    raise RuntimeError("existing target mapping disagrees with staged mapping")
                bench.save(destination, bench.read(mapping))
            control.update(status="source_audited_stage_imported", source_status=record["status"],
                suite_lock_acquired_legally=True, imported_target_models=list(STAGED_MODELS))
            bench.save(control_path, control)
            # The original evaluator skips imported/completed targets. Only a fully
            # successful Autoformer group can receive six new formal predictions.
            bench.evaluate_models(suite, suite.parent.parent, baseline, manifest)
            bench.verify_data(suite.parent.parent, baseline, full_hash=True)
            bench.import_cp(suite, suite.parent.parent, baseline, manifest)
            bench.report_models(suite, manifest)
            from experiments.mixed95.audit_benchmark_results import audit
            audit_path = suite / "independent_final_numeric_audit_after_resume.json"
            if audit_path.exists():
                raise RuntimeError("final audit already exists; refuse overwrite")
            bench.save(audit_path, audit(suite, suite / "audit_inputs", live=False))
            if bench.sha256(stage_path / "report.json") != STAGE_REPORT_SHA:
                raise RuntimeError("original staged target results changed")
            control.update(status="finished", finished_at=bench.now(), final_audit=str(audit_path),
                formal_target_predictions=len(bench.read(suite / "report.json")["predictions"]))
            bench.save(control_path, control)
            print("RESUME SUPERVISOR FINISHED", control["formal_target_predictions"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    suite = args.suite.resolve()
    try:
        resume(suite, args.stage.resolve())
    except BaseException as exc:
        control_path = suite / "resume_after_staged11_20261013.json"
        if control_path.exists():
            control = bench.read(control_path)
            if control.get("supervisor_pid") == os.getpid():
                control.update(status="supervisor_error", error_at=bench.now(),
                    reason=f"{type(exc).__name__}: {exc}", no_implicit_training_retry=True)
                bench.save(control_path, control)
        raise


if __name__ == "__main__":
    main()
