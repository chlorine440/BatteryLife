import copy
import tempfile
import unittest
from pathlib import Path

from experiments.mixed95.resume_paused import (
    STAGED_MODELS, import_stage, process, resume_order, validate_stage, zombie_returncode,
)


def fixture():
    models, targets, rows = {}, {}, []
    for model in STAGED_MODELS:
        values = [{"model": model, "seed": seed, "cell": cell,
                   "predicted_life_cycle": 123.0, "proxy_label_life_cycle": truth}
                  for seed in (2021, 42, 2024) for cell, truth in (("C1", 379), ("C2", 419))]
        targets[model] = {"status": "complete", "rows": values}
        models[model] = {"seeds": {str(s): {"status": "complete"} for s in (2021, 42, 2024)},
                         "target": targets[model] if model == "CPTransformer" else {"status": "pending"}}
        rows.extend(values)
    models["Autoformer"] = {"seeds": {"2021": {"status": "complete"},
        "42": {"status": "complete"}, "2024": {"status": "running"}}, "target": {"status": "pending"}}
    manifest = {"models": models, "protocol": {"original": True}, "identity": {"frozen": "hash"}}
    stage = {"stage_status": "complete", "target_status": targets, "protocol_amendment": "user authorized"}
    return manifest, stage, {"predictions": rows}


class ResumeTests(unittest.TestCase):
    def test_resume_children_before_their_parents(self):
        members = [{"pid": 4, "ppid": 2}, {"pid": 1, "ppid": 0},
                   {"pid": 3, "ppid": 1}, {"pid": 2, "ppid": 1}]
        result = [p["pid"] for p in resume_order(members)]
        self.assertLess(result.index(4), result.index(2))
        self.assertLess(result.index(2), result.index(1))
        self.assertLess(result.index(3), result.index(1))
        with self.assertRaises(ValueError):
            resume_order([{"pid": 1, "ppid": 2}, {"pid": 2, "ppid": 1}])

    def test_exit_status_only_from_real_zombie(self):
        self.assertEqual(zombie_returncode({"state": "Z", "exit_status": 0}), 0)
        self.assertEqual(zombie_returncode({"state": "Z", "exit_status": 256}), 1)
        self.assertEqual(zombie_returncode({"state": "Z", "exit_status": 9}), -9)
        with self.assertRaises(ValueError):
            zombie_returncode({"state": "T", "exit_status": 0})

    def test_process_parses_kernel_exit_and_start_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "123"
            folder.mkdir()
            fields = ["Z", "456"] + ["0"] * 17 + ["98765"] + ["0"] * 29 + ["256"]
            (folder / "stat").write_text("123 (name with spaces) " + " ".join(fields))
            (folder / "cmdline").write_bytes(b"python\0-m\0worker\0")
            parsed = process(123, Path(tmp))
            self.assertEqual((parsed["state"], parsed["ppid"], parsed["starttime"], parsed["exit_status"]),
                             ("Z", 456, "98765", 256))
            self.assertIsNone(process(124, Path(tmp)))

    def test_complete_stage_validated_against_unchanged_snapshot(self):
        manifest, stage, report = fixture()
        validate_stage(copy.deepcopy(manifest), manifest, stage, report)
        manifest["identity"] = {}
        with self.assertRaises(ValueError):
            validate_stage(fixture()[0], manifest, stage, report)

    def test_duplicate_or_inconsistent_target_rows_rejected(self):
        manifest, stage, report = fixture()
        stage["target_status"]["MICN"]["rows"][0]["cell"] = "C2"
        with self.assertRaises(ValueError):
            validate_stage(manifest, manifest, stage, report)
        manifest, stage, report = fixture()
        report["predictions"] = report["predictions"][:-1]
        with self.assertRaises(ValueError):
            validate_stage(manifest, manifest, stage, report)

    def test_existing_target_attempt_and_additional_running_trial_rejected(self):
        manifest, stage, report = fixture()
        manifest["models"]["MICN"]["target"] = {"status": "running"}
        with self.assertRaises(ValueError):
            validate_stage(manifest, manifest, stage, report)
        manifest, stage, report = fixture()
        manifest["models"]["Autoformer"]["seeds"]["42"]["status"] = "running"
        with self.assertRaises(ValueError):
            validate_stage(manifest, manifest, stage, report)

    def test_import_only_targets_and_protocol_preserves_source_and_cp(self):
        manifest, stage, _ = fixture()
        old = copy.deepcopy(manifest)
        imported = import_stage(manifest, stage, Path("/stage"))
        self.assertEqual(manifest, old)
        self.assertEqual(imported["identity"], old["identity"])
        self.assertEqual(imported["models"]["CPTransformer"], old["models"]["CPTransformer"])
        self.assertEqual(imported["models"]["Autoformer"], old["models"]["Autoformer"])
        for model in STAGED_MODELS:
            self.assertEqual(imported["models"][model]["seeds"], old["models"][model]["seeds"])
            self.assertEqual(imported["models"][model]["target"]["rows"], stage["target_status"][model]["rows"])
            if model != "CPTransformer":
                self.assertTrue(imported["models"][model]["target"]["no_repeat_target_inference"])


if __name__ == "__main__":
    unittest.main()
