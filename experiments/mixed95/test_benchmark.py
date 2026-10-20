"""Offline benchmark protocol tests; never launch training or COMAC evaluation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.mixed95.benchmark_config import (
    MODELS, ROOT, RNN_MODELS, instantiate, resolve_config, provenance, cli_args,
)
from experiments.mixed95.benchmark_inference import (
    adapt_input, compare_metrics, metrics, validate_scaler, configure_inference_backend,
)
from experiments.mixed95.benchmark import (
    BASELINE_PATH, aggregate, checked_configs, evaluate_models, initialize, launch_trial,
    read, suite_root, log_tail,
)


class RegistryTests(unittest.TestCase):
    def test_backend_recovery_rejects_changed_training_args(self):
        from experiments.mixed95.benchmark_recovery import validate_imported_args
        from experiments.mixed95.benchmark import trial_args
        previous = {"suite_id": "v3", "identity": {"configs": {"CPBiGRU": resolve_config("CPBiGRU")}}}
        source = Path("/tmp/test-source")
        root = source / "runs/CPBiGRU/seed_42"
        row = {"args": trial_args(previous, "CPBiGRU", 42, "/audited/input", root / "checkpoint", False),
               "log": str(root / "train.log"), "checkpoint": str(root / "checkpoint/selected")}
        validate_imported_args(previous, source, "CPBiGRU", 42, row, "/audited/input", False)
        row["args"]["batch_size"] *= 2
        with self.assertRaises(RuntimeError):
            validate_imported_args(previous, source, "CPBiGRU", 42, row, "/audited/input", False)

    def test_backend_recovery_never_retries_training_failures_or_live_work(self):
        from experiments.mixed95.benchmark_recovery import eligible_for_reaudit
        self.assertTrue(eligible_for_reaudit({"status": "complete"}))
        self.assertTrue(eligible_for_reaudit({"status": "failed", "exit_code": 0,
            "reason": "ValueError: independent/logged metric mismatch: acc15: .02"}))
        for row in [
            {"status": "running"}, {"status": "pending"},
            {"status": "failed", "exit_code": 1, "reason": "out of memory"},
            {"status": "failed", "exit_code": 0, "reason": "nonfinite checkpoint"},
            {"status": "failed", "reason": "ValueError: independent/logged metric mismatch: acc15"},
        ]:
            self.assertFalse(eligible_for_reaudit(row))

    def test_inference_backend_matches_original_trainer(self):
        import ast
        import torch
        original = ast.parse((ROOT / "run_main.py").read_text())
        seed_fn = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "set_seed")
        expected = {}
        for n in seed_fn.body:
            if isinstance(n, ast.Assign) and len(n.targets) == 1:
                name = ast.unparse(n.targets[0])
                if name.startswith("torch.backends.cudnn."):
                    expected[name.rsplit(".", 1)[-1]] = ast.literal_eval(n.value)
        self.assertEqual(expected, {"benchmark": False, "deterministic": True, "enabled": False})
        prior = {k: getattr(torch.backends.cudnn, k) for k in expected}
        try:
            for k, v in expected.items():
                setattr(torch.backends.cudnn, k, not v)
            proof = configure_inference_backend()
            for k, v in expected.items():
                self.assertEqual(getattr(torch.backends.cudnn, k), v)
                self.assertEqual(proof[f"cudnn_{k}"], v)
        finally:
            for k, v in prior.items():
                setattr(torch.backends.cudnn, k, v)

    def test_original_thirteen_models_finite_scalar_and_state_reload(self):
        import io
        import torch
        torch.set_num_threads(2)
        x = torch.rand(1, 100, 3, 300)
        mask = torch.ones(1, 100)
        for name in MODELS:
            with self.subTest(model=name):
                # Original AutoCorrelation hardcodes .cuda(). The remote
                # CUDA test covers it; do not patch the original algorithm.
                if name == "Autoformer" and not torch.cuda.is_available():
                    continue
                device = "cuda:0" if name == "Autoformer" else "cpu"
                xx, mm = x.to(device), mask.to(device)
                model = instantiate(resolve_config(name)).to(device).eval()
                with torch.inference_mode():
                    prediction = model(xx, mm)
                self.assertEqual(tuple(prediction.shape), (1, 1))
                self.assertTrue(torch.isfinite(prediction).all())
                stream = io.BytesIO()
                torch.save(model.state_dict(), stream)
                stream.seek(0)
                del model
                model = instantiate(resolve_config(name)).to(device).eval()
                model.load_state_dict(torch.load(stream, weights_only=True))
                with torch.inference_mode():
                    reloaded = model(xx, mm)
                self.assertTrue(torch.equal(prediction, reloaded), name)

    def test_all_thirteen_frozen_configs(self):
        baseline = read(BASELINE_PATH)
        configs, sources = checked_configs(baseline)
        self.assertEqual(len(configs), 13)
        expected = {
            "CPTransformer": (128, 1, 5e-5, 256, 64, 1, 12, 0),
            "CPMLP": (16, 1, 5e-5, 32, 256, 12, 7, 0),
            "CPGRU": (8, 1, 5e-4, 32, 32, 9, 2, .05),
            "CPLSTM": (8, 1, 5e-5, 128, 256, 10, 2, .05),
            "CPBiGRU": (16, 1, 5e-5, 64, 256, 6, 2, .05),
            "CPBiLSTM": (32, 1, 1e-3, 64, 128, 3, 2, .1),
            "MLP": (32, 1, 5e-5, 64, 128, 0, 12, 0),
            "CNN": (4, 4, 5e-5, 128, 128, 6, 2, .1),
            "DLinear": (32, 2, 5e-4, 64, 64, 4, 2, 0),
            "PatchTST": (16, 1, 1e-3, 64, 64, 2, 2, .1),
            "Autoformer": (4, 4, 5e-5, 128, 256, 2, 2, .1),
            "iTransformer": (32, 1, 1e-3, 128, 256, 6, 2, 0),
            "MICN": (4, 4, 5e-5, 256, 64, 4, 2, 0),
        }
        keys = ("batch_size", "accumulation_steps", "learning_rate", "d_model", "d_ff",
                "e_layers", "d_layers", "dropout")
        for m, a in configs.items():
            self.assertEqual(tuple(a[k] for k in keys), expected[m], m)
            self.assertEqual(a["dataset"], "MIX_large")
            self.assertEqual(a["num_workers"], 32)
            self.assertEqual(a["least_epochs"], 5)
            self.assertFalse(a["use_amp"])
            self.assertNotIn("root_path", a)
            self.assertNotIn("--use_amp", cli_args(a))
        self.assertEqual(configs["CPGRU"]["lstm_layers"], 4)
        self.assertEqual(configs["CPLSTM"]["lstm_layers"], 2)
        self.assertEqual(configs["PatchTST"]["patch_len"], 150)
        self.assertEqual(configs["PatchTST"]["stride"], 150)
        self.assertEqual(configs["MICN"]["c_out"], 3)
        # MICN shell patch variables were never passed in its actual command.
        self.assertEqual(configs["MICN"]["patch_len"], 10)
        self.assertEqual(sources["CPMLP"]["parameter_source"], "paper_Li-ion_table")
        self.assertEqual(sources["MLP"]["parameter_source"], "repository_script")

    def test_invalid_model_is_not_imported(self):
        with self.assertRaises(ValueError):
            instantiate({"model": "Dummy"})

    def test_suite_isolated_and_identity_guarded(self):
        with tempfile.TemporaryDirectory() as d:
            p = suite_root(Path(d), "test")
            p.mkdir(parents=True)
            baseline = read(BASELINE_PATH)
            m = initialize(p, baseline)
            self.assertEqual(initialize(p, baseline), m)
            m["identity"]["configs"]["CNN"]["batch_size"] = 1
            (p / "manifest.json").write_text(json.dumps(m))
            with self.assertRaises(RuntimeError):
                initialize(p, baseline)
            with self.assertRaises(ValueError):
                suite_root(Path(d), "../old")


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(42).normal(size=(100, 3, 300)).astype(np.float32)
        self.ids = np.arange(1, 101)

    def test_rnn_gap_includes_physical_100(self):
        original = self.x.copy()
        mask = np.r_[0, np.ones(99)].astype(np.float32)
        for model in RNN_MODELS:
            x, m, ids = adapt_input(model, self.x, mask, self.ids)
            np.testing.assert_array_equal(x[:99], original[1:])
            np.testing.assert_array_equal(ids[:99], np.arange(2, 101))
            np.testing.assert_array_equal(m, np.r_[np.ones(99), 0])
            self.assertFalse(x[-1].any())
            self.assertEqual(ids[-1], 0)
        np.testing.assert_array_equal(self.x, original)

    def test_other_models_keep_physical_slots(self):
        for model in set(MODELS) - RNN_MODELS:
            x, m, ids = adapt_input(model, self.x, np.r_[0, np.ones(99)], self.ids)
            self.assertFalse(x[0].any())
            np.testing.assert_array_equal(x[1:], self.x[1:])
            np.testing.assert_array_equal(ids, self.ids)

    def test_contiguous_prefix_is_identical(self):
        mask = np.r_[np.ones(77), np.zeros(23)].astype(np.float32)
        self.x[77:] = 0
        for model in RNN_MODELS:
            x, m, _ = adapt_input(model, self.x, mask, self.ids)
            np.testing.assert_array_equal(x, self.x)
            np.testing.assert_array_equal(m, mask)

    def test_future_and_invalid_inputs_rejected(self):
        for ids in (np.arange(2, 102), np.arange(100)):
            with self.assertRaises(ValueError):
                adapt_input("CPGRU", self.x, np.ones(100), ids)
        with self.assertRaises(ValueError):
            adapt_input("CPGRU", self.x, np.zeros(100), self.ids)

    def test_rnn_prefix_predictions_unchanged(self):
        import torch
        torch.set_num_threads(2)
        self.x[77:] = 0
        mask = np.r_[np.ones(77), np.zeros(23)].astype(np.float32)
        for name in RNN_MODELS:
            torch.manual_seed(7)
            model = instantiate(resolve_config(name)).eval()
            x, m, _ = adapt_input(name, self.x, mask, self.ids)
            with torch.inference_mode():
                p = model(torch.from_numpy(self.x[None]), torch.from_numpy(mask[None]))
                q = model(torch.from_numpy(x[None]), torch.from_numpy(m[None]))
            self.assertTrue(torch.equal(p, q), name)


class MetricsAndSafetyTests(unittest.TestCase):
    def test_original_training_entry_all_imports_without_loading_data(self):
        import os
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, HF_HOME=d, MPLCONFIGDIR=d)
            result = subprocess.run([sys.executable,
                str(ROOT / "experiments/mixed95/benchmark_worker.py"), "--help"],
                cwd=ROOT, env=env, capture_output=True, text=True, check=True, timeout=60)
        self.assertIn("usage: run_main.py", result.stdout)
        self.assertIn("--early_cycle_threshold", result.stdout)
        self.assertNotIn("Loading training samples", result.stdout)

    def test_actual_direct_worker_import_check_before_data_loading(self):
        import subprocess
        import sys
        result = subprocess.run([sys.executable,
            str(ROOT / "experiments/mixed95/benchmark_worker.py"), "--import-check"],
            cwd=ROOT, capture_output=True, text=True, check=True)
        proof = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertTrue(proof["evaluate_load_callable"])
        self.assertIn("site-packages", proof["evaluate_module"])

    def test_worker_script_path_cannot_shadow_third_party_evaluate(self):
        import sys
        import importlib.util
        from experiments.mixed95.benchmark_worker import configure_import_path
        wrapper = ROOT / "experiments/mixed95"
        # Reproduce Python's direct-script sys.path[0], not just -m import.
        original_path = list(sys.path)
        with patch.object(sys, "path", [str(wrapper), str(ROOT), *original_path]):
            configure_import_path()
            self.assertEqual(sys.path[0], str(ROOT))
            self.assertNotIn(str(wrapper), sys.path)
            spec = importlib.util.find_spec("evaluate")
            self.assertIsNotNone(spec, "original trainer requires Hugging Face evaluate")
            self.assertNotEqual(Path(spec.origin).resolve(), wrapper / "evaluate.py")
            self.assertIn("site-packages", spec.origin)

    def test_progress_log_tail_can_start_inside_utf8_character(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "progress.log"
            path.write_bytes("█".encode("utf-8") * 10000 + b"\nTrain Loss: 0.25\n")
            text = log_tail(path, limit=20000)
            self.assertIn("Train Loss: 0.25", text)
            self.assertIn("\ufffd", text)  # incomplete leading codepoint safely replaced

    def test_full_trial_monitor_with_long_progress_log(self):
        from types import SimpleNamespace
        b = read(BASELINE_PATH)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            manifest = initialize(p, b)
            process = SimpleNamespace(pid=999999, returncode=0)
            process.poll = iter([None, 0]).__next__
            def start(command, **kwargs):
                root = p / "smoke/CPMLP/seed_2021"
                args = read(root / "resolved_args.json")
                cp = root / "checkpoint/setting"
                cp.mkdir(parents=True)
                (cp / "args.json").write_text(json.dumps(args))
                (cp / "checkpoint_reload_probes.json").write_text(json.dumps([
                    {"state_exactly_equal": True, "prediction_exactly_equal": True}]))
                kwargs["stdout"].write("█" * 10000 + "\nTrain Loss: 0.25\n")
                kwargs["stdout"].flush()
                return process
            with patch("experiments.mixed95.benchmark.environment_preflight"), \
                 patch("experiments.mixed95.benchmark.gpu_memory", return_value=[[0, 2000, 32000, 20], [1, 2000, 32000, 20]]), \
                 patch("experiments.mixed95.benchmark.time.sleep"), \
                 patch("experiments.mixed95.benchmark.reload_probe", return_value={"two_loads_equal": True}), \
                 patch("experiments.mixed95.benchmark.subprocess.Popen", side_effect=start), \
                 patch("experiments.mixed95.benchmark.os.killpg"):
                launch_trial(p, manifest, b, "CPMLP", 2021, "unused", True)
            self.assertEqual(manifest["models"]["CPMLP"]["smoke"]["status"], "complete")

    def test_save_time_probe_state_prediction_and_rng(self):
        import torch
        from safetensors.torch import save_file
        from experiments.mixed95.benchmark_worker import checkpoint_probe
        args = resolve_config("CPMLP")
        model = instantiate(args).eval()
        state = model.state_dict()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "args.json").write_text(json.dumps(args))
            save_file(state, str(p / "model.safetensors"))
            before = torch.get_rng_state().clone()
            proof = checkpoint_probe(state, p)
            self.assertTrue(proof["prediction_exactly_equal"])
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            bad = {k: v.clone() for k, v in state.items()}
            key = next(iter(bad))
            bad[key] += 1
            with self.assertRaises(RuntimeError):
                checkpoint_probe(bad, p)

    def test_metrics_units_and_population_std(self):
        result = metrics([90, 120], [100, 100])
        self.assertAlmostEqual(result["mae_cycles"], 15)
        self.assertAlmostEqual(result["rmse_cycles"], np.sqrt(250))
        self.assertAlmostEqual(result["mape"], .15)
        self.assertEqual(result["acc15"], 50)
        self.assertEqual(result["acc10"], 50)
        summary = aggregate([result, result, result])
        self.assertEqual(summary["mape"]["std_population"], 0)
        with self.assertRaises(ValueError):
            aggregate([result])
        with self.assertRaises(ValueError):
            metrics([np.nan], [379])
        bad = dict(result, mape=.18)
        with self.assertRaises(ValueError):
            compare_metrics(result, bad)

    def test_scaler_train_only_count(self):
        from types import SimpleNamespace
        baseline = read(BASELINE_PATH)["scaler"]
        scaler = SimpleNamespace(mean_=[baseline["mean"]], scale_=[baseline["scale"]], n_samples_seen_=283)
        validate_scaler(scaler, baseline)
        scaler.n_samples_seen_ = 468
        with self.assertRaises(ValueError):
            validate_scaler(scaler, baseline)

    def test_target_waits_for_all_source_trials(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            b = read(BASELINE_PATH)
            m = initialize(p, b)
            with self.assertRaises(RuntimeError):
                evaluate_models(p, p, b, m)

    def test_completed_and_failed_trials_never_relaunch(self):
        b = read(BASELINE_PATH)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            m = initialize(p, b)
            for status in ("complete", "failed"):
                m["models"]["CPGRU"]["seeds"]["2021"] = {"status": status}
                with patch("experiments.mixed95.benchmark.subprocess.Popen") as start:
                    launch_trial(p, m, b, "CPGRU", 2021, "unused", False)
                    start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
