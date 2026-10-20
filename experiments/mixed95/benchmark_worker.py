"""Execute unmodified run_main.py with read-only checkpoint-save diagnostics.

The hook observes save_model only. No optimizer, data, forward-training,
loss, checkpoint selection or original source file is changed. CPU random
state is restored after diagnostics so data sampling is unaffected.
"""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def configure_import_path():
    """Match direct run_main imports; don't shadow Hugging Face evaluate.

    Launching this file by path adds experiments/mixed95 to sys.path. Its
    evaluate.py is a different module from the third-party evaluate package
    used by run_main, so the wrapper directory must not remain on the path.
    """
    wrapper_directory = Path(__file__).resolve().parent
    excluded = {ROOT, wrapper_directory}
    sys.path[:] = [str(ROOT), *[p for p in sys.path if Path(p or ".").resolve() not in excluded]]


configure_import_path()


def checkpoint_probe(before_state, directory):
    import torch
    from safetensors.torch import load_file
    from experiments.mixed95.benchmark_config import instantiate
    args = json.loads((directory / "args.json").read_text())
    if (directory / "model.safetensors").exists():
        after_state = load_file(str(directory / "model.safetensors"), device="cpu")
    else:
        after_state = torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True)
    if set(before_state) != set(after_state) or any(
            not torch.equal(before_state[k], after_state[k]) for k in before_state):
        raise RuntimeError("saved checkpoint state differs from in-memory selected model")
    # Diagnostic copies only, never the live model. Autoformer's original
    # AutoCorrelation uses hard-coded .cuda(); run that model natively on
    # CUDA rather than changing its implementation to enable CPU inference.
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator().manual_seed(0)
        x, mask = torch.rand(1, 100, 3, 300, generator=generator), torch.ones(1, 100)
        device = "cuda" if args["model"] == "Autoformer" else "cpu"
        x, mask = x.to(device), mask.to(device)
        model = instantiate(args).to(device).eval()
        model.load_state_dict(before_state)
        with torch.inference_mode():
            before = model(x, mask)
        model.load_state_dict(after_state)
        with torch.inference_mode():
            after = model(x, mask)
        if before.shape != (1, 1) or not torch.isfinite(before).all() or not torch.equal(before, after):
            raise RuntimeError("selected in-memory and reloaded checkpoint predictions differ")
    return {"state_exactly_equal": True, "prediction_exactly_equal": True,
            "synthetic_scaled_prediction": float(before.item()), "synthetic_input_only": True,
            "live_model_unchanged": True, "cpu_rng_restored": True, "diagnostic_device": device}


def main():
    import torch
    from accelerate import Accelerator
    original_save = Accelerator.save_model

    def observed_save(self, model, save_directory, *args, **kwargs):
        directory = Path(save_directory)
        before = None
        if self.is_main_process:
            before = {k: v.detach().cpu().clone() for k, v in self.unwrap_model(model).state_dict().items()}
        result = original_save(self, model, save_directory, *args, **kwargs)
        self.wait_for_everyone()
        if self.is_main_process:
            probe = checkpoint_probe(before, directory)
            path = directory / "checkpoint_reload_probes.json"
            records = json.loads(path.read_text()) if path.exists() else []
            records.append(probe)
            path.write_text(json.dumps(records, indent=2))
        self.wait_for_everyone()
        return result

    Accelerator.save_model = observed_save
    sys.argv[0] = str(ROOT / "run_main.py")
    runpy.run_path(str(ROOT / "run_main.py"), run_name="__main__")


if __name__ == "__main__":
    if sys.argv[1:] == ["--import-check"]:
        # Exercise the actual direct-script import path before loading data.
        import evaluate
        if not callable(getattr(evaluate, "load", None)):
            raise RuntimeError(f"wrong evaluate module: {getattr(evaluate, '__file__', None)}")
        print(json.dumps({"evaluate_module": evaluate.__file__, "evaluate_load_callable": True}))
    else:
        main()
