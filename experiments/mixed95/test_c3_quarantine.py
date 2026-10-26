"""A narrowly scoped non-feature record exception, never blanket recovery."""
import hashlib
import unittest

from experiments.mixed95.c3_quarantine import remove_exact_nonfeature_row


class QuarantineTests(unittest.TestCase):
    def bad_row(self, step=15):
        return (f'<row r="7"><c r="C7"><v>{step}</v></c>'
                '<c r="K7</v></broken></row>').encode()

    def test_only_one_exact_row_is_removed(self):
        row=self.bad_row();before=b'<sheet><row r="6"/></sheetData>'
        after=b'<row r="8"/><row r="9"/></sheet>'
        got,log=remove_exact_nonfeature_row(before+row+after,7,hashlib.sha256(row).hexdigest())
        self.assertEqual(got,before+after)
        self.assertEqual(log['removed_bytes'],len(row))
        self.assertTrue(log['bytes_outside_row_unchanged'])

    def test_changed_row_refused(self):
        row=self.bad_row()
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            remove_exact_nonfeature_row(row,7,'0'*64)

    def test_duplicate_row_refused(self):
        row=self.bad_row()
        with self.assertRaisesRegex(ValueError,'exactly one'):
            remove_exact_nonfeature_row(row+row,7,hashlib.sha256(row).hexdigest())

    def test_missing_row_refused(self):
        with self.assertRaisesRegex(ValueError,'exactly one'):
            remove_exact_nonfeature_row(b'<sheet/>',7,'0'*64)

    def test_charging_row_never_removed(self):
        row=self.bad_row(step=10)
        with self.assertRaisesRegex(ValueError,'NON-charging'):
            remove_exact_nonfeature_row(row,7,hashlib.sha256(row).hexdigest(),expected_step=10)

    def test_unexpected_step_refused(self):
        row=self.bad_row(step=19)
        with self.assertRaisesRegex(ValueError,'NON-charging'):
            remove_exact_nonfeature_row(row,7,hashlib.sha256(row).hexdigest())


if __name__=='__main__':
    unittest.main()
