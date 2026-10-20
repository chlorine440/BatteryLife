import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.mixed95.priority_handoff import (
    ORDER, ROOT, is_preflight, next_command, proc, validate_boundary,
    acquire_released_suite_lock,
)


def manifest():
    models = {m: {"smoke": {"status": "complete"},
                  "target": {"status": "pending"},
                  "seeds": {str(s): {"status": "pending"} for s in (2021, 42, 2024)}}
              for m in ("Autoformer", "iTransformer", "MICN", "CPMLP")}
    models["CPMLP"]["seeds"] = {str(s): {"status": "complete", "checkpoint": str(s)}
                                 for s in (2021, 42, 2024)}
    models["Autoformer"]["seeds"]["2021"] = {"status": "running", "pid": 123}
    return {"identity": {"code_sha256": {"frozen": "hash"}}, "models": models}


class PriorityHandoffTests(unittest.TestCase):
    def test_transient_lock_release_delay_waits_without_bypassing(self):
        prefix = "experiments.mixed95.priority_handoff."
        with patch(prefix+"fcntl.flock", side_effect=[BlockingIOError(), BlockingIOError(), None]) as lock, \
                patch(prefix+"time.monotonic", return_value=0), patch(prefix+"time.sleep") as sleep:
            acquire_released_suite_lock(123)
            self.assertEqual(lock.call_count, 3)
            self.assertEqual(sleep.call_count, 2)

    def test_unreleased_lock_times_out_without_launch(self):
        prefix = "experiments.mixed95.priority_handoff."
        with patch(prefix+"fcntl.flock", side_effect=BlockingIOError()), \
                patch(prefix+"time.monotonic", side_effect=[0, 31]), patch(prefix+"subprocess.Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "do not launch another queue"):
                acquire_released_suite_lock(123)
            launch.assert_not_called()

    def test_queue_order_uses_original_entry_no_parameter_overrides(self):
        suite = Path("/root/task/dataset/mixed95/benchmarks/suite_v4")
        command = next_command(suite, "/env/python")
        self.assertEqual(ORDER, ("iTransformer", "MICN", "Autoformer"))
        self.assertEqual(command, ["/env/python", "-u", "-m", "experiments.mixed95.benchmark",
            "train", "--output", "/root/task/dataset/mixed95", "--suite-id", "suite_v4",
            "--models", "iTransformer", "MICN", "Autoformer", "--then-evaluate"])

    def test_only_import_parser_probes_not_training_workers_allowed(self):
        worker = str(ROOT/"experiments/mixed95/benchmark_worker.py")
        for flag in ("--import-check", "--help"):
            self.assertTrue(is_preflight(["python", worker, flag]))
        for argv in (["python", worker, "--model", "Autoformer"],
                     ["python", worker, "--help", "--model", "Autoformer"],
                     ["python", "/another/benchmark_worker.py", "--help"],
                     ["python", worker]):
            self.assertFalse(is_preflight(argv))

    def test_current_training_or_audit_must_finish(self):
        initial = manifest()
        with self.assertRaisesRegex(RuntimeError, "not terminal"):
            validate_boundary(copy.deepcopy(initial), initial)
        for status in ("complete", "failed"):
            current = copy.deepcopy(initial)
            current["models"]["Autoformer"]["seeds"]["2021"]["status"] = status
            self.assertEqual(validate_boundary(current, initial)["status"], status)

    def test_other_trial_start_or_completed_result_mutation_rejected(self):
        initial = manifest()
        for model, seed, field, value in (("Autoformer", "42", "status", "running"),
                                        ("iTransformer", "2021", "status", "running"),
                                        ("CPMLP", "42", "checkpoint", "changed")):
            current = copy.deepcopy(initial)
            current["models"]["Autoformer"]["seeds"]["2021"]["status"] = "complete"
            current["models"][model]["seeds"][seed][field] = value
            with self.assertRaises(RuntimeError):
                validate_boundary(current, initial)

    def test_identity_smoke_and_target_are_preserved(self):
        initial = manifest()
        for kind in ("identity", "smoke", "target"):
            current = copy.deepcopy(initial)
            current["models"]["Autoformer"]["seeds"]["2021"]["status"] = "complete"
            if kind == "identity":
                current["identity"] = {}
            else:
                current["models"]["iTransformer"][kind] = {"status": "changed"}
            with self.assertRaises(RuntimeError):
                validate_boundary(current, initial)

    def test_proc_identity_handles_spaces_and_start_time_not_pid_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            folder = base/"123"
            folder.mkdir()
            fields = ["S", "1"] + ["0"]*17 + ["987654"] + ["0"]*5
            (folder/"stat").write_text("123 (python worker) " + " ".join(fields))
            (folder/"cmdline").write_bytes(b"python\0-m\0module\0")
            value = proc(123, base)
            self.assertEqual(value["start_ticks"], "987654")
            self.assertEqual(value["argv"], ["python", "-m", "module"])
            self.assertEqual(value["ppid"], 1)
            self.assertIsNone(proc(456, base))


if __name__ == "__main__":
    unittest.main()
