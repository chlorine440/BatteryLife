"""User-authorized 11-model target evaluation while Autoformer is live-paused.

Reads the original suite under its paused coordinator. Writes only a separate
stage directory, never the original manifest/checkpoints or training code.
Stage targets must be imported, not inferred again, before a later full report.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.mixed95.benchmark import (
    BASELINE_PATH, MODELS, SEEDS, checked_configs, now, read, save, sha256,
    suite_lock, verify_data,
)
from experiments.mixed95.benchmark_inference import target_evaluation
from experiments.mixed95.audit_benchmark_results import audit, calculate, summarize

STAGED_MODELS = tuple(m for m in MODELS if m not in {"Autoformer", "PatchTST"})


def validate_scope(manifest):
    if len(STAGED_MODELS) != 11:
        raise ValueError("expected exactly eleven selected models")
    for model in STAGED_MODELS:
        entry = manifest["models"][model]
        if set(entry["seeds"]) != {str(s) for s in SEEDS} or any(
                r["status"] != "complete" for r in entry["seeds"].values()):
            raise ValueError(f"three accepted source seeds required: {model}")
        if entry["target"]["status"] != ("complete" if model == "CPTransformer" else "pending"):
            raise ValueError(f"existing target trial cannot be repeated: {model}")
    return STAGED_MODELS


def process_info(pid):
    path = Path(f"/proc/{pid}")
    raw = (path / "stat").read_text()
    values = raw[raw.rfind(")") + 2:].split()
    return {"state": values[0], "starttime": values[19]}


def validate_pause(pause):
    for old in [pause["coordinator"], *pause["members_before_pause"]]:
        actual = process_info(old["pid"])
        if actual["starttime"] != old["starttime"] or actual["state"] != "T":
            raise ValueError(f"original training process not live-paused: {old['pid']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite-id", default="benchmark95_20261011_v4")
    parser.add_argument("--stage-id", default="staged11_20261013")
    args = parser.parse_args()
    output = args.output.resolve()
    suite = output / "benchmarks" / args.suite_id
    stage = suite / args.stage_id
    baseline = read(BASELINE_PATH)
    configs, _ = checked_configs(baseline)
    original_manifest = suite / "manifest.json"
    manifest_hash = sha256(original_manifest)
    original = read(original_manifest)
    models = validate_scope(original)
    pause = read(suite / "autoformer_live_pause_for_staged11_20261013.json")
    validate_pause(pause)
    # Independent stage lock. The original coordinator remains stopped and
    # retains its original suite lock. Never write through that lock.
    with suite_lock(stage):
        if (stage / "manifest.json").exists():
            raise RuntimeError("stage already attempted; do not repeat target predictions")
        save(stage / "original_suite_snapshot.json", original)
        identity = {
            "scope": "11-model user-authorized staged target evaluation",
            "created_at": now(), "models": list(models),
            "original_manifest_sha256": manifest_hash,
            "module_sha256": sha256(Path(__file__)),
            "source_cells": {"train": 283, "val": 89, "test": 96},
            "effective_datasets": 8, "target_independent_cells": 2,
            "proxy_labels": {"C1": 379, "C2": 419},
            "protocol_amendment": "User requested pause Autoformer and evaluate accepted eleven models before all 39 source trials terminate",
            "excluded": {"Autoformer": "user-requested live pause; source group incomplete", "PatchTST": "all three source checkpoint/log metric audits failed"},
            "training_changed": False, "original_suite_modified": False,
            "future_reuse_required": "Import completed stage predictions without new target inference before any later full-suite evaluation",
            "device": "cpu", "precision": "FP32", "target_status": {},
        }
        save(stage / "manifest.json", identity)
        save(stage / "data_preflight.json", verify_data(output, baseline, full_hash=True))
        save(stage / "source_independent_preflight.json",
             audit(suite, suite / "audit_inputs", live=True))
        predictions = []
        for model in models:
            validate_pause(pause)
            if sha256(original_manifest) != manifest_hash:
                raise RuntimeError("original suite manifest changed during pause")
            entry = original["models"][model]
            if model == "CPTransformer":
                result = entry["target"]["rows"]
                if len(result) != 6 or not entry["target"].get("imported"):
                    raise ValueError("expected six previously accepted imported CP results")
                state = {"status": "complete", "imported": True, "rows": result}
            else:
                for record in entry["seeds"].values():
                    for name, digest in record["checkpoint_sha256"].items():
                        if sha256(Path(record["checkpoint"]) / name) != digest:
                            raise ValueError("selected source checkpoint changed")
                identity["target_status"][model] = {"status": "running", "started_at": now()}
                save(stage / "manifest.json", identity)
                try:
                    result = target_evaluation(
                        {model: entry["seeds"]}, configs, baseline,
                        output / "comac_loader_v2",
                        lambda m, c, v: save(stage / "target_mapping" / f"{m}_{c}.json", v),
                        device="cpu")
                    if len(result) != 6:
                        raise ValueError("expected three seeds times two cells")
                    state = {"status": "complete", "imported": False,
                             "rows": result, "finished_at": now()}
                except Exception as exc:
                    state = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                             "finished_at": now()}
            identity["target_status"][model] = state
            save(stage / "manifest.json", identity)
            if state["status"] == "complete":
                predictions.extend(state["rows"])
            print(model, state["status"], flush=True)
        summary = []
        for model in models:
            state = identity["target_status"][model]
            row = {"model": model, "status": state["status"]}
            if state["status"] == "complete":
                values = state["rows"]
                row["metrics"] = summarize([
                    calculate([v["predicted_life_cycle"] for v in values if int(v["seed"]) == s],
                              [v["proxy_label_life_cycle"] for v in values if int(v["seed"]) == s])
                    for s in SEEDS])
                row["cells"] = {}
                for cell, truth in (("C1", 379), ("C2", 419)):
                    cell_rows = [v for v in values if v["cell"] == cell]
                    import numpy as np
                    row["cells"][cell] = {
                        "proxy_label_life_cycle": truth,
                        "predicted_mean": float(np.mean([v["predicted_life_cycle"] for v in cell_rows])),
                        "predicted_std_population": float(np.std([v["predicted_life_cycle"] for v in cell_rows], ddof=0)),
                        "ape_percent_mean": float(np.mean([v["ape_percent"] for v in cell_rows])),
                    }
            else:
                row["reason"] = state["reason"]
            summary.append(row)
        validate_pause(pause)
        if sha256(original_manifest) != manifest_hash:
            raise RuntimeError("original suite manifest changed")
        report = {"stage_id": args.stage_id, "completed_at": now(),
                  "scope": identity["scope"], "full_benchmark_complete": False,
                  "summary": summary, "predictions": predictions,
                  "independent_target_cell_count": 2, "std": "population, ddof=0",
                  "protocol_amendment": identity["protocol_amendment"]}
        save(stage / "report.json", report)
        identity["finished_at"] = now()
        identity["stage_status"] = "complete" if all(r["status"] == "complete" for r in summary) else "complete_with_failures"
        save(stage / "manifest.json", identity)
        print("STAGE", identity["stage_status"], "target rows", len(predictions), flush=True)


if __name__ == "__main__":
    main()
