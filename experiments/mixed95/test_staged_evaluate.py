"""Stage authorization/gates only; no model or target inference."""
import unittest
from unittest.mock import patch

from experiments.mixed95.staged_evaluate import STAGED_MODELS, validate_scope, validate_pause


def fixture():
    return {"models": {m: {"seeds": {str(s): {"status": "complete"}
                                   for s in (2021, 42, 2024)},
                           "target": {"status": "complete" if m == "CPTransformer" else "pending"}}
                       for m in STAGED_MODELS}}


class StagedEvaluationTests(unittest.TestCase):
    def test_eleven_accepted_groups_without_autofomer_or_patchtst(self):
        self.assertEqual(len(validate_scope(fixture())), 11)
        self.assertIn("CPTransformer", STAGED_MODELS)
        self.assertFalse({"Autoformer", "PatchTST"} & set(STAGED_MODELS))

    def test_incomplete_source_seed_rejected(self):
        m = fixture()
        m["models"]["MICN"]["seeds"]["2024"]["status"] = "running"
        with self.assertRaises(ValueError):
            validate_scope(m)

    def test_previous_target_attempt_cannot_repeat(self):
        m = fixture()
        m["models"]["CPMLP"]["target"]["status"] = "complete"
        with self.assertRaises(ValueError):
            validate_scope(m)

    def test_pause_requires_same_live_stopped_processes(self):
        pause = {"coordinator": {"pid": 1, "starttime": "10"},
                 "members_before_pause": [{"pid": 2, "starttime": "20"}]}
        with patch("experiments.mixed95.staged_evaluate.process_info",
                   side_effect=[{"state": "T", "starttime": "10"},
                                {"state": "T", "starttime": "20"}]):
            validate_pause(pause)
        with patch("experiments.mixed95.staged_evaluate.process_info",
                   return_value={"state": "S", "starttime": "10"}):
            with self.assertRaises(ValueError):
                validate_pause(pause)
        with patch("experiments.mixed95.staged_evaluate.process_info",
                   return_value={"state": "T", "starttime": "999"}):
            with self.assertRaises(ValueError):
                validate_pause(pause)


if __name__ == "__main__":
    unittest.main()
