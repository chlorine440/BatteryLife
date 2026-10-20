"""Unchanged model inference, independent metrics, and target-only RNN adapter."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.mixed95.benchmark_config import instantiate, RNN_MODELS, SEEDS

METRICS = ("mae_cycles", "rmse_cycles", "mape", "acc15", "acc10")


def configure_inference_backend():
    """Match run_main.set_seed's backend, without seeding or importing training.

    Packed RNN cuDNN and native kernels can put borderline predictions on
    different sides of an accuracy threshold. This restores the original
    evaluation backend; it does not relax metric comparison tolerances.
    """
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    return {"cudnn_benchmark": False, "cudnn_deterministic": True,
            "cudnn_enabled": False, "source": "run_main.py:set_seed"}


def metrics(predicted, truth) -> dict:
    p, y = np.asarray(predicted, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if p.shape != y.shape or not p.size or not np.isfinite(p).all() or not (
            np.isfinite(y).all() and (y > 0).all()):
        raise ValueError("metrics require equal, nonempty finite arrays and positive labels")
    err = p - y
    rel = np.abs(err) / y
    return dict(mae_cycles=float(np.mean(np.abs(err))), rmse_cycles=float(np.sqrt(np.mean(err**2))),
                mape=float(np.mean(rel)), acc15=float(np.mean(rel <= .15) * 100),
                acc10=float(np.mean(rel <= .10) * 100))


def compare_metrics(actual: dict, logged: dict) -> dict:
    deltas = {k: actual[k] - logged[k] for k in METRICS}
    for key, delta in deltas.items():
        limit = .002 if key in {"mae_cycles", "rmse_cycles"} else .000051
        if abs(delta) > limit:
            raise ValueError(f"independent/logged metric mismatch: {key}: {delta}")
    return deltas


def adapt_input(model: str, features, mask, physical_cycles):
    """No future data; masked values zeroed identically to original collate.

    Only CP-RNNs compact gaps. Prefix input is bit-for-bit unchanged. Zero in
    the saved mapping denotes padding, not an additional physical cycle.
    """
    x, m, ids = np.asarray(features), np.asarray(mask), np.asarray(physical_cycles)
    if x.shape != (100, 3, 300) or m.shape != (100,) or ids.shape != (100,):
        raise ValueError("expected 100 physical slots with 3x300 features")
    if not np.isfinite(x).all() or not np.isin(m, [0, 1]).all():
        raise ValueError("nonfinite features or nonbinary mask")
    if not np.array_equal(ids, np.arange(1, 101)):
        raise ValueError("input physical IDs must be exactly 1..100")
    x, m, ids = x.copy(), m.copy(), ids.copy()
    x[m == 0] = 0
    if model not in RNN_MODELS:
        return x, m, ids
    valid = np.flatnonzero(m)
    if not len(valid):
        raise ValueError("RNN input must contain at least one observed cycle")
    out, out_mask, mapping = np.zeros_like(x), np.zeros_like(m), np.zeros_like(ids)
    out[:len(valid)] = x[valid]
    out_mask[:len(valid)] = 1
    mapping[:len(valid)] = ids[valid]
    return out, out_mask, mapping


def validate_scaler(scaler, expected):
    if not (np.isclose(scaler.mean_[0], expected["mean"], rtol=0, atol=1e-10)
            and np.isclose(scaler.scale_[0], expected["scale"], rtol=0, atol=1e-10)
            and int(scaler.n_samples_seen_) == 283):
        raise ValueError("label scaler is not fitted on the audited 283 training cells")


def load_checkpoint(checkpoint: Path, expected_args: dict, scaler_expected: dict, device="cpu"):
    import joblib
    import torch
    configure_inference_backend()
    args = json.loads((checkpoint / "args.json").read_text())
    for key, value in expected_args.items():
        if key in {"root_path", "checkpoints", "model_comment"}:
            continue
        if args.get(key) != value:
            raise ValueError(f"checkpoint config mismatch: {key}: {args.get(key)} != {value}")
    model = instantiate(args)
    safe = checkpoint / "model.safetensors"
    if safe.is_file():
        from safetensors.torch import load_file
        state = load_file(str(safe), device="cpu")
    else:
        state = torch.load(checkpoint / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("nonfinite checkpoint parameters")
    model = model.to(device).eval()
    scaler = joblib.load(checkpoint / "label_scaler")
    validate_scaler(scaler, scaler_expected)
    return model, scaler, args


def reload_probe(checkpoint, args, expected_scaler):
    """Two independent loads agree; smoke never reads COMAC curves/labels."""
    import torch
    device = "cuda:0" if args["model"] == "Autoformer" else "cpu"
    model, _, _ = load_checkpoint(checkpoint, args, expected_scaler, device)
    torch.manual_seed(0)
    x, m = torch.rand(1, 100, 3, 300).to(device), torch.ones(1, 100).to(device)
    with torch.inference_mode():
        first = model(x, m)
    if first.shape != (1, 1) or not torch.isfinite(first).all():
        raise ValueError(f"not a finite scalar lifespan: {first}")
    del model
    model, _, _ = load_checkpoint(checkpoint, args, expected_scaler, device)
    with torch.inference_mode():
        second = model(x, m)
    if not torch.equal(first, second):
        raise ValueError("checkpoint reloaded predictions changed")
    return {"scaled_prediction": first.item(), "two_loads_equal": True,
            "inference_backend": configure_inference_backend()}


def source_evaluation(checkpoint: Path, expected_args: dict, baseline: dict, destination: Path,
                      logged_metrics: dict, device="cuda:0"):
    import joblib
    import torch
    from argparse import Namespace
    from data_provider.data_factory import data_provider_baseline
    model, scaler, args = load_checkpoint(checkpoint, expected_args, baseline["scaler"], device)
    life_scaler = joblib.load(checkpoint / "life_class_scaler")
    # Feature/label computation is original. Workers=0 only for sequential,
    # read-only post-training audit; this is not a training configuration.
    dataset, loader = data_provider_baseline(Namespace(**dict(args, num_workers=0)),
                                            "test", None, scaler, life_class_scaler=life_scaler)
    cells = [r for r in baseline["cells"] if r["split"] == "test" and r["life"] is not None]
    if len(dataset) != 9600 or dataset.unique_labels != [r["life"] for r in cells]:
        raise ValueError("source evaluation order/count/labels changed")
    predictions, truths, windows = [], [], []
    with torch.inference_mode():
        for x, mask, y, *_ in loader:
            if x.shape[1:] != (100, 3, 300) or not torch.isfinite(x).all() or x[mask == 0].any():
                raise ValueError("source features violate original input protocol")
            output = model(x.float().to(device), mask.float().to(device))
            if output.shape != y.shape or not torch.isfinite(output).all():
                raise ValueError("nonfinite or incorrect source output")
            # Float32 inverse transform matches original training entry.
            p = output * float(scaler.scale_[0]) + float(scaler.mean_[0])
            t = y.float().to(device) * float(scaler.scale_[0]) + float(scaler.mean_[0])
            predictions.extend(p.cpu().numpy().reshape(-1))
            truths.extend(t.cpu().numpy().reshape(-1))
            windows.extend(mask.sum(1).numpy().astype(int))
    p, y, w = np.asarray(predictions), np.asarray(truths), np.asarray(windows)
    if not np.array_equal(w, np.tile(np.arange(1, 101), 96)):
        raise ValueError("source window order changed")
    cell_ids = np.repeat([r["cell"] for r in cells], 100)
    keep = cell_ids != baseline["warning_cell"]
    report = {"all_windows": metrics(p, y), "cycle100_only": metrics(p[w == 100], y[w == 100]),
              "sensitivity_without_flagged_cell": metrics(p[keep], y[keep]),
              "cell_count": 96, "window_count": 9600,
              "label_quality_warning": baseline["warning_cell"],
              "inference_backend": configure_inference_backend()}
    report["logged_metric_deltas"] = compare_metrics(report["all_windows"], logged_metrics)
    np.savez_compressed(destination, predicted=p, truth=y, observation_cycles=w, cell=cell_ids)
    return report


def target_evaluation(records: dict, configs: dict, baseline: dict, target_dir: Path,
                      save_mapping, device="cpu") -> list[dict]:
    import torch
    from experiments.mixed95.comac_input import load_target_input
    rows = []
    for model_name, model_records in records.items():
        if model_name == "CPTransformer":
            continue  # imported old official results, never rerun
        if set(model_records) != {str(s) for s in SEEDS} or any(
                r["status"] != "complete" for r in model_records.values()):
            continue
        for seed in SEEDS:
            record = model_records[str(seed)]
            model_device = "cuda:0" if model_name == "Autoformer" else device
            model, scaler, _ = load_checkpoint(Path(record["checkpoint"]),
                dict(configs[model_name], seed=seed), baseline["scaler"], model_device)
            for cell, truth in (("C1", 379), ("C2", 419)):
                features, mask = load_target_input(target_dir / f"{cell}_input.npz")
                x, m, mapping = adapt_input(model_name, features, mask, np.arange(1, 101))
                save_mapping(model_name, cell, {"physical_cycle_by_slot": mapping.tolist(),
                    "mask": m.tolist(), "target_only_RNN_compaction": model_name in RNN_MODELS})
                with torch.inference_mode():
                    output = model(torch.from_numpy(x[None]).to(model_device),
                                   torch.from_numpy(m[None]).to(model_device))
                if output.shape != (1, 1) or not torch.isfinite(output).all():
                    raise ValueError(f"nonfinite or incorrect target output: {model_name}/{seed}/{cell}")
                life = float(scaler.inverse_transform([[float(output.item())]])[0, 0])
                rows.append(dict(model=model_name, seed=seed, cell=cell,
                    predicted_life_cycle=life, proxy_label_life_cycle=truth,
                    absolute_error_cycles=abs(life-truth), ape_percent=100*abs(life-truth)/truth))
            del model
    return rows
