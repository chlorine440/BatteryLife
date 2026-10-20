"""Focused CPU/C3 checks; no training and no changes to original benchmark."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.mixed95.c3_evaluate import ROOT, OUTPUT, SUITE, interval_errors, cpu_autoformer_aggregation, sha


class C3Tests(unittest.TestCase):
    def test_interval_discrete_boundaries(self):
        self.assertEqual(interval_errors(202,201,301)['absolute_error_cycles_min'],0)
        self.assertEqual(interval_errors(301,201,301)['absolute_error_cycles_min'],0)
        self.assertEqual(interval_errors(201,201,301)['absolute_error_cycles_min'],1)
        self.assertEqual(interval_errors(350,201,301)['absolute_error_cycles_min'],49)
        self.assertEqual(interval_errors(100,201,301)['absolute_error_cycles_min'],102)

    def test_interval_not_exact_label(self):
        audit=json.loads((OUTPUT/'input_audit.json').read_text())
        self.assertIsNone(audit['label']['exact_life_cycle'])
        self.assertEqual(audit['label']['integer_crossing_range_assuming_monotone'],[202,301])
        self.assertTrue(audit['label']['no_daily_div094_label'])

    def test_mapping_ignores_only_empty_export_tails(self):
        a=json.loads((OUTPUT/'input_audit.json').read_text())
        self.assertEqual([r['physical_cycle'] for r in a['mapping']],list(range(1,405)))
        self.assertEqual([(r['file'],r['local_cycle']) for r in a['excluded']],
                         [('C3-100-1.xlsx',101),('C3-200-1.xlsx',101),('C3-400.xlsx',101)])
        self.assertEqual([r['cycle'] for r in a['rpt']],[1,101,201,301,401,402])
        self.assertTrue(a['independent_rpt_openpyxl_exact'])

    def test_features_no_future(self):
        with np.load(OUTPUT/'C3_input.npz',allow_pickle=False) as a:
            self.assertEqual(a['features'].shape,(100,3,300))
            np.testing.assert_array_equal(a['physical_cycles'],np.arange(1,101))
            np.testing.assert_array_equal(a['mask'],np.r_[0,np.ones(99)])
            self.assertFalse(a['features'][0].any())
            self.assertTrue(np.isfinite(a['features']).all())
        a=json.loads((OUTPUT/'input_audit.json').read_text())
        self.assertEqual(set(a['cycle_stats']),set(map(str,range(2,101))))

    def test_cpu_autoformer_math(self):
        g=torch.Generator().manual_seed(7)
        v=torch.randn(2,2,3,37,generator=g);c=torch.randn(2,2,3,37,generator=g)
        actual=cpu_autoformer_aggregation(SimpleNamespace(factor=3),v,c)
        weights,delay=torch.topk(c.mean((1,2)),int(3*math.log(37)),dim=-1)
        weights=weights.softmax(-1);expected=torch.zeros_like(v)
        for b in range(2):
            for i in range(delay.shape[1]):
                expected[b]+=torch.roll(v[b],-int(delay[b,i]),-1)*weights[b,i]
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)

    def test_frozen_hashes_preserved(self):
        a=json.loads((OUTPUT/'input_audit.json').read_text())
        self.assertTrue(all(sha(Path(p))==v for p,v in a['protected_input_sha256'].items()))
        original=json.loads((SUITE/'report.json').read_text())
        self.assertEqual(len(original['predictions']),72)

    def test_accepted_cpu_replays_and_masks(self):
        r=json.loads((OUTPUT/'predictions.json').read_text())['results']
        good=[v for v in r if v['status']=='complete']
        self.assertEqual(len(good),33)
        for v in good:
            self.assertLessEqual(max(abs(p['delta']) for p in v['replay']),.005)
            self.assertLessEqual(max(v['physical_cycle_by_slot']),100)
            self.assertEqual(sum(v['mask']),99)
            if v['model'] in ('CPGRU','CPLSTM','CPBiGRU','CPBiLSTM'):
                self.assertEqual(v['physical_cycle_by_slot'][:99],list(range(2,101)))
                self.assertEqual(v['physical_cycle_by_slot'][-1],0)


if __name__=='__main__':unittest.main()
