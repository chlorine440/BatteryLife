"""Small-data reconstruction tests, independent of raw workbook availability."""
import unittest
from pathlib import Path
import struct
import tempfile
import zipfile
import json

import numpy as np

from experiments.mixed95.c3_reconstruct import (
    WINDOW, OUT, calibrate, estimate, first_estimated_crossing, indicators, partial_q, workbook_integrity,
)
from experiments.mixed95.c3_evaluate import sha


class ReconstructionTests(unittest.TestCase):
    def trace(self):
        v = np.linspace(2.9,4.1,100)
        return [(i,float(i),float(t),91.8,float(100*(t-2.9))) for i,t in enumerate(v)]

    def test_window_capacity_and_dedup(self):
        p=self.trace()
        dq,info=partial_q(p+p[-5:])
        self.assertEqual(info['duplicates_removed'],5)
        self.assertEqual(len(dq),len(WINDOW)-1)
        self.assertAlmostEqual(dq.sum(),100*(WINDOW[-1]-WINDOW[0]),places=10)

    def test_missing_window_not_extrapolated(self):
        with self.assertRaisesRegex(ValueError,'window not covered'):
            partial_q(self.trace()[60:])

    def test_conflicting_export_is_rejected(self):
        p=self.trace()
        with self.assertRaisesRegex(ValueError,'conflicting'):
            partial_q(p+[(1,1.,3.,91.8,900.)])

    def test_capacity_counter_reset_is_rejected(self):
        p=self.trace();p[50]=(*p[50][:4],0.)
        with self.assertRaisesRegex(ValueError,'counter reset'):
            partial_q(p)

    def test_entropy_and_training_only_pca(self):
        q=np.array([[1.,1.,1.],[2.,2.,2.],[9.,4.,8.]])
        x,fit=indicators(q,[0,1])
        self.assertAlmostEqual(x[0,1],np.log(3))
        np.testing.assert_allclose(fit['mean'],[1.5]*3)
        q[2]*=10
        _,fit2=indicators(q,[0,1])
        np.testing.assert_array_equal(fit['loading'],fit2['loading'])

    def test_scaler_uses_only_training_anchors(self):
        q=np.array([[1.,2.,3.],[2.,3.,4.],[3.,4.,5.],[8.,1.,2.]])
        result,fit=estimate(q,[0,1,2],[1.,.98,.96])
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(fit['hi_mean'],np.asarray(fit['hi'])[:3].mean(axis=0))

    def test_calibration_exact_anchor_no_extrapolation(self):
        c=np.arange(1,8);raw=np.full(7,.97)
        got=calibrate(c,raw,[2,6],[1.,.94])
        self.assertTrue(np.isnan(got[0]) and np.isnan(got[-1]))
        self.assertEqual(got[1],1.)
        self.assertAlmostEqual(got[5],.94)
        np.testing.assert_array_equal(raw,np.full(7,.97))

    def test_estimated_crossing_boundary_and_missing(self):
        self.assertEqual(first_estimated_crossing([1,2,3,4],np.array([np.nan,.951,.95,.94])),3)
        self.assertIsNone(first_estimated_crossing([1,2],np.array([1.,np.nan])))

    def test_crc_is_checked_not_bypassed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'book.xlsx'
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('xl/worksheets/sheet6.xml','<sheet/>')
            self.assertTrue(workbook_integrity(path)['all_parts_pass_crc'])
            data=bytearray(path.read_bytes())
            central=data.index(b'PK\x01\x02')
            struct.pack_into('<I',data,central+16,0)
            path.write_bytes(data)
            bad=workbook_integrity(path)
            self.assertFalse(bad['all_parts_pass_crc'])
            self.assertEqual(bad['bad_parts'][0]['part'],'xl/worksheets/sheet6.xml')


@unittest.skipUnless((OUT/'reconstruction.json').exists(),'real reconstruction not generated yet')
class ReconstructionArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit=json.loads((OUT/'extraction_audit.json').read_text())
        cls.report=json.loads((OUT/'reconstruction.json').read_text())

    def test_404_physical_cycles_complete(self):
        with np.load(OUT/'charge_features.npz',allow_pickle=False) as data:
            np.testing.assert_array_equal(data['cycles'],np.arange(1,405))
            self.assertEqual(data['delta_q'].shape,(404,len(WINDOW)-1))
            self.assertTrue(np.isfinite(data['delta_q']).all())
            self.assertTrue((data['delta_q']>=0).all())
        self.assertTrue(self.audit['full_charge_step_reconciled_to_summary'])

    def test_only_known_nonfeature_record_quarantined(self):
        self.assertEqual(len(self.audit['quarantine']),1)
        row=self.audit['quarantine'][0]
        self.assertEqual((row['xml_row'],row['step'],row['physical_cycle']),(56394,15,257))
        self.assertEqual(row['remaining_rows'],200041)
        self.assertEqual(row['rows_after_quarantine'],143648)
        self.assertGreater(row['feature_records_after_quarantine'],0)
        self.assertFalse(row['raw_crc_pass'])
        self.assertTrue(row['strict_remaining_xml_pass'])

    def test_raw_and_benchmark_remain_unchanged(self):
        for key in ['raw_sha256','protected_sha256']:
            for path,expected in self.audit[key].items():
                self.assertEqual(sha(Path(path)),expected)

    def test_interpolation_and_pseudolabel_not_truth(self):
        r=self.report
        self.assertIsNone(r['exact95_life_cycle'])
        self.assertFalse(r['suitable_as_measured_benchmark_truth'])
        self.assertFalse(r['strict_paper_reproduction'])
        self.assertEqual(r['original_rpt95_bracket'],[202,301])
        self.assertEqual([v['physical_cycle'] for v in r['rows']],list(range(1,405)))
        self.assertTrue(all(v['reconstructed_pseudo_soh'] is None for v in r['rows'][-2:]))
        self.assertEqual(sum(v['measured_rpt_soh'] is not None for v in r['rows']),6)

    def test_group_holdout_no_neighbor_leak(self):
        for row in self.report['held_out']:
            self.assertNotIn(row['cycle'],row['training_rpt_cycles'])
            if row['cycle'] in [401,402]:
                self.assertFalse(set(row['training_rpt_cycles'])&{401,402})

    def test_saved_metrics_independently_recomputed(self):
        for key,field in [('hi_all_checkpoints','hi_prediction'),
                          ('anchored_hi_interior','corrected_hi_prediction'),
                          ('pchip_interior','pchip_prediction')]:
            errors=np.array([100*(r[field]-r['measured_soh']) for r in self.report['held_out'] if r[field] is not None])
            actual=self.report['validation'][key]
            self.assertEqual(actual['n'],len(errors))
            self.assertAlmostEqual(actual['mae_pp'],np.abs(errors).mean(),places=12)
            self.assertAlmostEqual(actual['rmse_pp'],np.sqrt(np.mean(errors**2)),places=12)

    def test_rpt_values_retained(self):
        rows={r['physical_cycle']:r for r in self.report['rows']}
        for rpt in self.report['rpt']:
            self.assertEqual(rows[rpt['cycle']]['measured_rpt_soh'],rpt['soh'])
            self.assertAlmostEqual(rows[rpt['cycle']]['reconstructed_pseudo_soh'],rpt['soh'],places=12)


if __name__=='__main__':
    unittest.main()
