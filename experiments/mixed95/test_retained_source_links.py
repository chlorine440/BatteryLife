import copy
import tempfile
import unittest
from pathlib import Path

from experiments.mixed95.explicit_retry import imported_manifest
from experiments.mixed95.retained_source_links import link_array, retained_pairs
from experiments.mixed95.test_explicit_retry import previous_fixture


class RetainedSourceLinksTests(unittest.TestCase):
    def test_only_accepted_retained_arrays_not_cp_retry_or_failures(self):
        previous, _ = previous_fixture()
        current = imported_manifest(previous, previous["identity"], "retry", "now")
        pairs = [(m, s) for m, s, _ in retained_pairs(previous, current)]
        self.assertEqual(len(pairs), 32)
        self.assertNotIn(("Autoformer", "2024"), pairs)
        self.assertFalse(any(m in {"CPTransformer", "PatchTST"} for m, _ in pairs))

    def test_changed_source_record_rejected(self):
        previous, _ = previous_fixture()
        current = imported_manifest(previous, previous["identity"], "retry", "now")
        current["models"]["MICN"]["seeds"]["42"]["status"] = "failed"
        with self.assertRaises(ValueError):
            retained_pairs(previous, current)

    def test_current_running_retry_is_never_linked(self):
        previous, _ = previous_fixture()
        current = imported_manifest(previous, previous["identity"], "retry", "now")
        current["models"]["Autoformer"]["seeds"]["2024"] = {"status": "running"}
        snapshot = copy.deepcopy(current)
        retained_pairs(previous, current)
        self.assertEqual(current, snapshot)

    def test_links_only_no_copy_or_original_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "old.npz"
            source.write_bytes(b"unchanged original array")
            destination = Path(tmp) / "new" / "source_test_predictions.npz"
            self.assertTrue(link_array(source, destination))
            self.assertTrue(destination.is_symlink())
            self.assertEqual(source.read_bytes(), b"unchanged original array")
            self.assertFalse(link_array(source, destination))

    def test_existing_regular_file_or_different_link_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, destination, other = [Path(tmp) / n for n in ("old", "new", "other")]
            source.write_bytes(b"original")
            destination.write_bytes(b"protected")
            with self.assertRaises(ValueError):
                link_array(source, destination)
            self.assertEqual(destination.read_bytes(), b"protected")
            other.symlink_to(destination)
            with self.assertRaises(ValueError):
                link_array(source, other)


if __name__ == "__main__":
    unittest.main()
