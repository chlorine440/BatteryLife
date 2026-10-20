"""Reference retained source arrays in the explicit retry suite, never re-infer.

This only creates missing NPZ symlinks and a separate provenance record. It does
not write either manifest, accepted arrays, checkpoints or the active trial.
"""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path

from experiments.mixed95 import benchmark as bench
from experiments.mixed95.audit_benchmark_results import compare, source_arrays


def retained_pairs(previous, current):
    if current["identity"] != previous["identity"]:
        raise ValueError("frozen suite identity changed")
    if current["explicit_retry"]["model"] != "Autoformer" or current["explicit_retry"]["seed"] != 2024:
        raise ValueError("unexpected retry scope")
    pairs = []
    for model, entry in previous["models"].items():
        if model == "CPTransformer":
            continue  # Original CP arrays already have their separate audit_inputs reference.
        for seed, record in entry["seeds"].items():
            if record["status"] != "complete":
                continue
            if (model, seed) == ("Autoformer", "2024"):
                raise ValueError("active retry array must never be linked/overwritten")
            if current["models"][model]["seeds"][seed] != record:
                raise ValueError("retained source trial changed")
            pairs.append((model, seed, record))
    return pairs


def link_array(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise ValueError(f"retained source array absent: {source}")
    if destination.exists() or destination.is_symlink():
        if not destination.is_symlink() or destination.resolve() != source.resolve():
            raise ValueError("refuse to replace an existing array or different link")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())
    return True


def link_retained(suite, previous_suite):
    path = suite / "retained_source_array_links.json"
    with (suite / "retained_source_array_links.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            raise RuntimeError("retained source linking already recorded; refuse implicit repeat")
        old_hash = bench.sha256(previous_suite / "manifest.json")
        previous, current = bench.read(previous_suite / "manifest.json"), bench.read(suite / "manifest.json")
        if current["explicit_retry"]["previous_manifest_sha256"] != old_hash:
            raise RuntimeError("previous manifest changed")
        baseline = bench.read(bench.BASELINE_PATH)
        test = [r for r in baseline["cells"] if r["split"] == "test" and r["life"] is not None]
        records = []
        for model, seed, record in retained_pairs(previous, current):
            source = previous_suite / "runs" / model / f"seed_{seed}" / "source_test_predictions.npz"
            destination = suite / "runs" / model / f"seed_{seed}" / "source_test_predictions.npz"
            digest = bench.sha256(source)
            measured = source_arrays(source, test, baseline["warning_cell"])
            for protocol, values in measured.items():
                compare(values, record["source_verified"][protocol])
            compare(measured["all_windows"], record["source_test"])
            created = link_array(source, destination)
            if bench.sha256(destination) != digest:
                raise RuntimeError("linked array differs from original")
            records.append({"model": model, "seed": int(seed), "source": str(source),
                            "link": str(destination), "sha256": digest, "created": created,
                            "independent_metrics_match": True})
        if len(records) != 32:
            raise RuntimeError("expected exactly 32 retained accepted non-CP source arrays")
        if bench.sha256(previous_suite / "manifest.json") != old_hash:
            raise RuntimeError("original manifest changed")
        after = bench.read(suite / "manifest.json")
        retained_pairs(previous, after)
        bench.save(path, {"passed": True, "checked_at": bench.now(), "links": records,
            "previous_manifest_sha256": old_hash, "frozen_code_or_manifests_modified": False,
            "original_arrays_modified": False, "no_model_inference": True,
            "active_trial_excluded": "Autoformer/2024",
            "module_sha256": bench.sha256(Path(__file__))})
        print("RETAINED SOURCE ARRAYS VERIFIED AND LINKED", len(records), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--previous-suite", type=Path, required=True)
    args = parser.parse_args()
    link_retained(args.suite.resolve(), args.previous_suite.resolve())


if __name__ == "__main__":
    main()
