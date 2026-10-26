"""Same-condition MICH experiment: independent split/label/numerical checks."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.mixed95.mich_che_control import (
    CELLS, OUT, PROTOCOL, checkpoint_indices, verify_condition, screen,
    model_inputs, score, features, audit,
)
from experiments.mixed95.c3_che_data import read


class ControlTests(unittest.TestCase):
    def fixture(self):
        arrays={'target_x':np.linspace(1.,.6,180)[:,None],'target_cycles':np.arange(1,181)}
        for c in range(4):
            x=np.linspace(1.,.55,180)
            arrays[f'c{c}_x']=x[:,None].copy();arrays[f'c{c}_soh']=x.copy();arrays[f'c{c}_cycles']=np.arange(1,181)
        rows=checkpoint_indices(180)
        cal=dict(points=[dict(row=int(j),cycle=int(j+1),soh=float(arrays['target_x'][j,0])) for j in rows],
                 group_rows=[[int(j)] for j in rows[:4]]+[rows[4:].tolist()])
        meta=dict(descriptors=[dict(id='window:total',window='window',kind='total')])
        return arrays,meta,cal

    def test_exactly_six_checkpoints_cover_endpoints_without_truth_input(self):
        for n in (150,180,437):
            idx=checkpoint_indices(n)
            self.assertEqual(len(set(idx)),6)
            self.assertEqual(idx[[0,-2,-1]].tolist(),[0,n-2,n-1])
        with self.assertRaises(ValueError):checkpoint_indices(6)

    def test_cells_disjoint_and_same_labelled_cohort(self):
        self.assertEqual(len(set(CELLS)),5)
        self.assertTrue(all('_45C_0-100_1-1C_' in s and s.startswith('MICH_MCForm') for s in CELLS))
        self.assertEqual(PROTOCOL['source_cells'],list(CELLS[:3]))
        self.assertEqual(PROTOCOL['validation_cell'],CELLS[3]);self.assertEqual(PROTOCOL['test_cell'],CELLS[4])

    def test_condition_mismatch_is_not_silently_accepted(self):
        records=[]
        for name in CELLS:
            records.append(dict(cell_id=name[:-4],form_factor='pouch',anode_material='graphite',
                cathode_material='NMC111',nominal_capacity_in_Ah=2.36,SOC_interval=[0,1],
                charge_protocol=[dict(rate_in_C=1.)],discharge_protocol=[dict(rate_in_C=1.)],
                min_voltage_limit_in_V=3.,max_voltage_limit_in_V=4.2))
        verify_condition(records)
        records[-1]['SOC_interval']=[.2,.8]
        with self.assertRaisesRegex(ValueError,'protocol differs'):verify_condition(records)

    def test_selector_has_no_dense_truth_argument_and_ignores_injected_truth(self):
        arrays,meta,cal=self.fixture();first=screen(arrays,meta,cal)
        altered=copy.deepcopy(arrays);altered['sealed_test_soh']=np.full(180,999.)
        changed_meta=copy.deepcopy(meta);changed_meta['sealed_test_soh']=np.full(180,-999.).tolist()
        self.assertEqual(first,screen(altered,changed_meta,cal))
        self.assertEqual(first['selected_indices'],[0]);self.assertFalse(first['dense_test_labels_used'])

    def test_calibration_is_not_part_of_test_mask(self):
        arrays,meta,cal=self.fixture();anchor=[q['row'] for q in cal['points']]
        mask=np.ones(180,bool);mask[anchor]=False
        self.assertEqual(int(mask.sum()),174)
        self.assertTrue(all(not mask[i] for i in anchor))
        self.assertEqual(len(cal['group_rows']),5)

    def test_scaler_ignores_target_and_validation(self):
        a,m,c=self.fixture();s=screen(a,m,c);_,state=model_inputs(a,s)
        a['target_x']*=1000;a['c3_x']*=100;a['c3_soh']*=.5
        _,other=model_inputs(a,s);self.assertEqual(state,other)
        self.assertEqual(state['fitted_cells'],list(CELLS[:3]))

    def test_direction_mismatch_stops_instead_of_lowering_threshold(self):
        a,m,c=self.fixture();a['c0_x']*=-1
        s=screen(a,m,c);self.assertEqual(s['status'],'stopped_no_qualified_features')
        with self.assertRaisesRegex(ValueError,'no qualified'):model_inputs(a,s)

    def test_metric_units_are_capacity_not_lifetime(self):
        s=score([1.,.9],[.99,.91])
        self.assertAlmostEqual(s['mae_pp'],1.);self.assertAlmostEqual(s['rmse_pp'],1.)
        self.assertAlmostEqual(s['mae_Ah'],.0236)


@unittest.skipUnless((OUT/'features.json').exists(),'MICH input artifacts not prepared')
class ControlInputTests(unittest.TestCase):
    def test_features_file_has_no_hidden_test_capacity_array(self):
        arrays,meta,cal=features(OUT)
        self.assertNotIn('target_y',arrays);self.assertNotIn('test_soh',arrays)
        self.assertTrue(all(not k.endswith('_soh') or k.startswith(('c0','c1','c2','c3')) for k in arrays))
        self.assertEqual(len(cal['points']),6)
        self.assertEqual(len(arrays['target_cycles'])-6,meta['hidden_capacity_count'])
        self.assertTrue(all(t['fitted_cells']==list(CELLS[:3]) for t in meta['pca'].values()))

    def test_condition_and_input_quality_audit(self):
        arrays,meta,cal=features(OUT);verify_condition(meta['cells_metadata'])
        self.assertEqual(len(meta['target_excluded_cycles'])+len(meta['target_eligible_cycles']),meta['target_total_preprocessed_cycles'])
        selection=screen(arrays,meta,cal)
        self.assertEqual(selection['status'],'selected')
        self.assertTrue(np.isfinite(arrays['target_x']).all())


@unittest.skipUnless((OUT/'results.json').exists(),'MICH experiment not yet evaluated')
class ControlResultTests(unittest.TestCase):
    def test_independent_saved_weight_and_metric_audit(self):
        self.assertEqual(audit(OUT)['status'],'passed')

    def test_dense_test_more_informative_than_three_C3_points(self):
        r=read(OUT/'results.json')
        self.assertGreater(r['hidden_capacity_points'],100)
        self.assertEqual(sum(q['calibration_point'] for q in r['rows']),6)
        self.assertTrue(all(q['evaluated']!=q['calibration_point'] for q in r['rows']))
        self.assertFalse(r['paired_C3_domain_shift_ablation'])


if __name__=='__main__':unittest.main()
