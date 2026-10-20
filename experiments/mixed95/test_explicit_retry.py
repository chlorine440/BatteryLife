import copy
import unittest

from experiments.mixed95.explicit_retry import imported_manifest, validate_previous
from experiments.mixed95.test_resume_paused import fixture


def previous_fixture():
    previous, _, _ = fixture()
    previous["suite_id"] = "old_v4"
    previous["models"]["Autoformer"]["seeds"]["2024"] = {
        "status": "failed", "exit_code": 1, "live_pause_seconds": 15161,
        "reason": "NCCL timeout", "log": "/old/train.log"}
    previous["models"]["Autoformer"]["target"] = {"status": "blocked_incomplete"}
    for entry in previous["models"].values():
        if entry["target"]["status"] == "pending":
            entry["target"] = {"status": "complete", "rows": [None] * 6}
    previous["models"]["PatchTST"] = {"seeds": {str(s): {"status": "failed", "reason": "audit mismatch"}
                                                       for s in (2021, 42, 2024)},
                                        "target": {"status": "blocked_incomplete"}}
    return previous, {"status": "finished", "trainer_exit_code": 1}


class ExplicitRetryTests(unittest.TestCase):
    def test_only_documented_pause_failure_can_be_authorized(self):
        previous, control = previous_fixture()
        validate_previous(previous, control)
        for key, value in (("status", "complete"), ("exit_code", 0), ("live_pause_seconds", 10)):
            changed = copy.deepcopy(previous)
            changed["models"]["Autoformer"]["seeds"]["2024"][key] = value
            with self.assertRaises(ValueError):
                validate_previous(changed, control)

    def test_live_or_previous_target_attempt_rejected(self):
        previous, control = previous_fixture()
        previous["models"]["Autoformer"]["target"]["status"] = "complete"
        with self.assertRaises(ValueError):
            validate_previous(previous, control)
        previous, control = previous_fixture()
        previous["models"]["PatchTST"]["seeds"]["42"]["status"] = "running"
        with self.assertRaises(ValueError):
            validate_previous(previous, control)

    def test_only_one_seed_and_unevaluated_target_reset(self):
        previous, _ = previous_fixture()
        old = copy.deepcopy(previous)
        new = imported_manifest(previous, previous["identity"], "retry_v1", "now")
        self.assertEqual(previous, old)
        self.assertEqual(new["models"]["Autoformer"]["seeds"]["2024"], {"status": "pending"})
        self.assertEqual(new["models"]["Autoformer"]["target"], {"status": "pending"})
        for seed in ("2021", "42"):
            self.assertEqual(new["models"]["Autoformer"]["seeds"][seed], old["models"]["Autoformer"]["seeds"][seed])
        for model in old["models"]:
            if model != "Autoformer":
                self.assertEqual(new["models"][model], old["models"][model])
        self.assertEqual(new["explicit_retry"]["failed_record"], old["models"]["Autoformer"]["seeds"]["2024"])

    def test_new_suite_keeps_frozen_identity(self):
        previous, _ = previous_fixture()
        with self.assertRaises(ValueError):
            imported_manifest(previous, {}, "retry_v1", "now")


if __name__ == "__main__":
    unittest.main()
