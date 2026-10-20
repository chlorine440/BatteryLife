"""User-requested queue reorder at a completed-trial boundary (Linux only).

Never signals a training worker. Waits for the current trial's original
coordinator to finish training AND its checkpoint/source audit, then stops
only that coordinator before the next trial is launched. The unmodified
benchmark entry resumes the same suite with a different model order.
This file is deliberately outside the frozen benchmark*.py code inventory.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORDER = ("iTransformer", "MICN", "Autoformer")
TERMINAL = {"complete", "failed", "blocked_smoke"}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(tmp, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def proc(pid, base=Path("/proc")):
    try:
        folder = base / str(pid)
        fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
        argv = (folder / "cmdline").read_bytes().split(b"\0")
        return {"pid": pid, "state": fields[0], "ppid": int(fields[1]),
                "start_ticks": fields[19],
                "argv": [x.decode() for x in argv if x]}
    except FileNotFoundError:
        return None


def same_process(pid, expected):
    value = proc(pid)
    if value is None or value["start_ticks"] != expected["start_ticks"]:
        raise RuntimeError("recorded process exited or PID was reused")
    return value


def is_preflight(argv):
    worker = str(ROOT / "experiments/mixed95/benchmark_worker.py")
    return (len(argv) == 3 and argv[1] == worker
            and argv[2] in {"--import-check", "--help"})


def validate_boundary(current, initial):
    if current["identity"] != initial["identity"]:
        raise RuntimeError("suite identity changed")
    trial = current["models"]["Autoformer"]["seeds"]["2021"]
    if trial["status"] not in TERMINAL:
        raise RuntimeError("current training/checkpoint audit is not terminal")
    for model, entry in current["models"].items():
        old = initial["models"][model]
        if entry["smoke"] != old["smoke"] or entry["target"] != old["target"]:
            raise RuntimeError("smoke or target records changed during handoff")
        for seed, record in entry["seeds"].items():
            if record["status"] == "running":
                raise RuntimeError("another source trial already started")
            if (model, seed) != ("Autoformer", "2021") and record != old["seeds"][seed]:
                raise RuntimeError("a trial other than the current seed changed")
    for seed in ("42", "2024"):
        if current["models"]["Autoformer"]["seeds"][seed]["status"] != "pending":
            raise RuntimeError("remaining Autoformer seeds must still be pending")
    return trial


def next_command(suite, python=sys.executable):
    return [python, "-u", "-m", "experiments.mixed95.benchmark", "train",
            "--output", str(suite.parent.parent), "--suite-id", suite.name,
            "--models", *ORDER, "--then-evaluate"]


def acquire_released_suite_lock(handle, timeout=30):
    """Allow a bounded lock-release delay; never remove or bypass the lock."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise RuntimeError("suite lock did not release; do not launch another queue") from None
            time.sleep(.1)


def terminate_coordinator(pid, identity):
    # Called only after the manifest boundary and child-process checks.
    same_process(pid, identity)
    os.kill(pid, signal.SIGTERM)
    os.kill(pid, signal.SIGCONT)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        value = proc(pid)
        if value is None or value["state"] == "Z":
            return
        if value["start_ticks"] != identity["start_ticks"]:
            raise RuntimeError("PID changed while waiting for coordinator exit")
        time.sleep(.1)
    same_process(pid, identity)
    os.kill(pid, signal.SIGKILL)
    for _ in range(50):
        value = proc(pid)
        if value is None or value["state"] == "Z":
            return
        time.sleep(.1)
    raise RuntimeError("old coordinator did not exit; do not start another queue")


def handoff(suite, coordinator_pid):
    control = suite / "priority_handoff.json"
    with (suite / "priority_handoff.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if control.exists():
            raise RuntimeError("handoff already recorded; never implicitly rerun it")
        initial = read(suite / "manifest.json")
        identity = proc(coordinator_pid)
        if identity is None or not any(x in identity["argv"] for x in (
                "experiments.mixed95.benchmark_recovery", "experiments.mixed95.benchmark")):
            raise RuntimeError("PID is not the expected benchmark coordinator")
        if suite.name not in identity["argv"]:
            raise RuntimeError("coordinator belongs to another suite")
        current = initial["models"]["Autoformer"]["seeds"]["2021"]
        if current["status"] != "running":
            raise RuntimeError("arm only while the current Autoformer trial is running")
        child = proc(current["pid"])
        if child is None or child["ppid"] != coordinator_pid:
            raise RuntimeError("live current trial does not belong to this coordinator")
        record = {"status": "waiting_current_trial", "requested_at": now(),
                  "request": "finish current Autoformer seed; iTransformer, MICN, then remaining Autoformer",
                  "observer_pid": os.getpid(), "old_coordinator": identity,
                  "current_trial_pid": current["pid"], "model_order": list(ORDER),
                  "script_sha256": sha(Path(__file__)), "training_unchanged": True,
                  "no_training_worker_signals": True, "no_target_prediction": True}
        write(control, record)
        signature = None
        frozen = False
        helpers = []
        try:
            while True:
                same_process(coordinator_pid, identity)
                stat = (suite / "manifest.json").stat()
                key = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
                if key != signature:
                    signature = key
                    manifest = read(suite / "manifest.json")
                    if manifest["models"]["Autoformer"]["seeds"]["2021"]["status"] in TERMINAL:
                        break
                time.sleep(.02)
            # Freeze only the scheduler; the original trial has already ended.
            same_process(coordinator_pid, identity)
            os.kill(coordinator_pid, signal.SIGSTOP)
            frozen = True
            for _ in range(100):
                if same_process(coordinator_pid, identity)["state"] == "T":
                    break
                time.sleep(.01)
            else:
                raise RuntimeError("coordinator did not reach its stopped boundary")
            manifest = read(suite / "manifest.json")
            finished = validate_boundary(manifest, initial)
            for model in ORDER:
                for seed, trial in manifest["models"][model]["seeds"].items():
                    if trial["status"] == "pending" and (suite / "runs" / model / f"seed_{seed}").exists():
                        raise RuntimeError("next trial directory exists; refuse unsafe boundary")
            children = Path(f"/proc/{coordinator_pid}/task/{coordinator_pid}/children").read_text().split()
            for pid in map(int, children):
                value = proc(pid)
                if value is None or value["state"] == "Z":
                    continue
                if not is_preflight(value["argv"]):
                    raise RuntimeError("live non-preflight child at boundary; never signal training workers")
                helpers.append(value)
            for path, expected in manifest["identity"]["code_sha256"].items():
                if sha(ROOT / path) != expected:
                    raise RuntimeError(f"frozen code changed: {path}")
            for name, expected in finished.get("checkpoint_sha256", {}).items():
                if sha(Path(finished["checkpoint"]) / name) != expected:
                    raise RuntimeError("finished checkpoint changed")
            write(suite / "priority_handoff_boundary_manifest.json", manifest)
            record.update(status="boundary_verified", verified_at=now(),
                          current_trial_status=finished["status"],
                          boundary_manifest_sha256=sha(suite / "priority_handoff_boundary_manifest.json"),
                          preflight_helpers=helpers)
            write(control, record)
            terminate_coordinator(coordinator_pid, identity)
            frozen = False
            # Only parser/import probes may exist; never a live trainer.
            for helper in helpers:
                value = proc(helper["pid"])
                if value and value["start_ticks"] == helper["start_ticks"] and value["state"] != "Z":
                    os.kill(helper["pid"], signal.SIGTERM)
            # The original suite lock must be free before resuming its entry.
            with (suite / ".lock").open("a") as suite_guard:
                acquire_released_suite_lock(suite_guard)
                command = next_command(suite)
                with (suite / "priority_queue_console.log").open("xb") as log:
                    process = subprocess.Popen(command, cwd=ROOT,
                        env=dict(os.environ, WANDB_MODE="offline", CUDA_VISIBLE_DEVICES="0,1"),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
            record.update(status="switched", switched_at=now(), new_coordinator_pid=process.pid,
                          new_command=command, queue_console=str(suite / "priority_queue_console.log"))
            write(control, record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        except BaseException as exc:
            if frozen:
                value = proc(coordinator_pid)
                if value and value["start_ticks"] == identity["start_ticks"]:
                    os.kill(coordinator_pid, signal.SIGCONT)
            record.update(status="failed_handoff", reason=f"{type(exc).__name__}: {exc}", at=now())
            write(control, record)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--coordinator-pid", type=int, required=True)
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        raise RuntimeError("process-boundary handoff requires Linux /proc")
    handoff(args.suite.resolve(), args.coordinator_pid)


if __name__ == "__main__":
    main()
